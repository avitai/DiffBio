"""Tests for the single-cell analysis pipeline.

Following TDD principles, these tests define the expected behavior
of the SingleCellPipeline before implementation.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


class TestSingleCellPipelineConfig:
    """Tests for SingleCellPipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from diffbio.pipelines.single_cell import SingleCellPipelineConfig

        config = SingleCellPipelineConfig()

        assert config.n_genes == 2000
        assert config.n_clusters == 10
        assert config.latent_dim == 64
        assert config.umap_n_components == 2
        assert config.enable_ambient_removal is True
        assert config.enable_batch_correction is True
        assert config.enable_dim_reduction is True
        assert config.enable_clustering is True

    def test_custom_config(self):
        """Test custom configuration values."""
        from diffbio.pipelines.single_cell import SingleCellPipelineConfig

        config = SingleCellPipelineConfig(
            n_genes=5000,
            n_clusters=20,
            latent_dim=128,
            umap_n_components=3,
            enable_ambient_removal=False,
        )

        assert config.n_genes == 5000
        assert config.n_clusters == 20
        assert config.latent_dim == 128
        assert config.umap_n_components == 3
        assert config.enable_ambient_removal is False


class TestSingleCellPipeline:
    """Tests for SingleCellPipeline operator."""

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.pipelines.single_cell import SingleCellPipelineConfig

        return SingleCellPipelineConfig(
            n_genes=100,  # Small for testing
            n_clusters=5,
            latent_dim=16,
            umap_n_components=2,
        )

    @pytest.fixture
    def pipeline(self, config, rngs):
        """Create pipeline instance."""
        from diffbio.pipelines.single_cell import SingleCellPipeline

        return SingleCellPipeline(config, rngs=rngs)

    @pytest.fixture
    def sample_data(self, config):
        """Create sample single-cell data."""
        n_cells = 50
        n_genes = config.n_genes

        # Simulate count data with some structure
        key = jax.random.key(42)
        counts = jax.random.poisson(key, lam=5.0, shape=(n_cells, n_genes)).astype(jnp.float32)

        # Ambient profile (average expression)
        ambient_profile = counts.mean(axis=0)

        # Batch labels
        batch_labels = jnp.zeros(n_cells, dtype=jnp.int32)
        batch_labels = batch_labels.at[n_cells // 2 :].set(1)

        return {
            "counts": counts,
            "ambient_profile": ambient_profile,
            "batch_labels": batch_labels,
        }

    def test_initialization(self, config, rngs):
        """Test pipeline initialization."""
        from diffbio.pipelines.single_cell import SingleCellPipeline

        pipeline = SingleCellPipeline(config, rngs=rngs)

        assert pipeline is not None
        assert hasattr(pipeline, "vae_normalizer")
        assert hasattr(pipeline, "clustering")

    def test_initialization_without_optional_components(self, rngs):
        """Test initialization with optional components disabled."""
        from diffbio.pipelines.single_cell import (
            SingleCellPipeline,
            SingleCellPipelineConfig,
        )

        config = SingleCellPipelineConfig(
            n_genes=100,
            enable_ambient_removal=False,
            enable_batch_correction=False,
            enable_dim_reduction=False,
        )
        pipeline = SingleCellPipeline(config, rngs=rngs)

        assert pipeline.ambient_removal is None
        assert pipeline.batch_correction is None
        assert pipeline.dim_reduction is None

    def test_apply_full_pipeline(self, pipeline, sample_data):
        """Test full pipeline apply method."""
        result, state, metadata = pipeline.apply(sample_data, {}, None)

        # Check output keys
        assert "normalized" in result
        assert "latent" in result
        assert "cluster_assignments" in result
        assert "embeddings_2d" in result

        # Check shapes
        n_cells = sample_data["counts"].shape[0]
        assert result["normalized"].shape[0] == n_cells
        assert result["latent"].shape[0] == n_cells
        assert result["cluster_assignments"].shape == (n_cells, pipeline.config.n_clusters)
        assert result["embeddings_2d"].shape == (n_cells, pipeline.config.umap_n_components)

    def test_apply_without_ambient_removal(self, sample_data, rngs):
        """Test pipeline without ambient removal."""
        from diffbio.pipelines.single_cell import (
            SingleCellPipeline,
            SingleCellPipelineConfig,
        )

        config = SingleCellPipelineConfig(
            n_genes=100,
            n_clusters=5,
            latent_dim=16,
            enable_ambient_removal=False,
        )
        pipeline = SingleCellPipeline(config, rngs=rngs)

        result, _, _ = pipeline.apply(sample_data, {}, None)

        # Should still produce outputs
        assert "normalized" in result
        assert "cluster_assignments" in result

    def test_apply_without_batch_correction(self, sample_data, rngs):
        """Test pipeline without batch correction."""
        from diffbio.pipelines.single_cell import (
            SingleCellPipeline,
            SingleCellPipelineConfig,
        )

        config = SingleCellPipelineConfig(
            n_genes=100,
            n_clusters=5,
            latent_dim=16,
            enable_batch_correction=False,
        )
        pipeline = SingleCellPipeline(config, rngs=rngs)

        result, _, _ = pipeline.apply(sample_data, {}, None)

        assert "normalized" in result
        assert "cluster_assignments" in result

    def test_output_finite(self, pipeline, sample_data):
        """Test that all outputs are finite."""
        result, _, _ = pipeline.apply(sample_data, {}, None)

        for key in ["normalized", "latent", "cluster_assignments", "embeddings_2d"]:
            assert jnp.all(jnp.isfinite(result[key])), f"{key} contains non-finite values"

    def test_cluster_assignments_valid(self, pipeline, sample_data):
        """Test that cluster assignments are valid probabilities."""
        result, _, _ = pipeline.apply(sample_data, {}, None)

        assignments = result["cluster_assignments"]

        # Should be non-negative
        assert jnp.all(assignments >= 0.0)

        # Should sum to approximately 1 for soft assignments
        row_sums = assignments.sum(axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)

    def test_preserves_original_data(self, pipeline, sample_data):
        """Test that original data is preserved in output."""
        result, _, _ = pipeline.apply(sample_data, {}, None)

        assert "counts" in result
        assert jnp.allclose(result["counts"], sample_data["counts"])


class TestSingleCellPipelineDifferentiability:
    """Tests for gradient flow through the pipeline."""

    @pytest.fixture
    def config(self):
        """Provide config for gradient tests."""
        from diffbio.pipelines.single_cell import SingleCellPipelineConfig

        return SingleCellPipelineConfig(
            n_genes=50,
            n_clusters=3,
            latent_dim=8,
        )

    def test_gradient_flow_through_pipeline(self, config, rngs):
        """Test that gradients flow through the full pipeline."""
        from diffbio.pipelines.single_cell import SingleCellPipeline

        pipeline = SingleCellPipeline(config, rngs=rngs)

        def loss_fn(pipe, counts, ambient, batch):
            data = {
                "counts": counts,
                "ambient_profile": ambient,
                "batch_labels": batch,
            }
            result, _, _ = pipe.apply(data, {}, None)
            return result["cluster_assignments"].sum()

        n_cells = 20
        n_genes = config.n_genes

        key = jax.random.key(0)
        counts = jax.random.poisson(key, lam=5.0, shape=(n_cells, n_genes)).astype(jnp.float32)
        ambient = counts.mean(axis=0)
        batch = jnp.zeros(n_cells, dtype=jnp.int32)

        grads = nnx.grad(loss_fn)(pipeline, counts, ambient, batch)

        # Gradients should exist
        assert grads is not None

    def test_gradient_wrt_input_counts(self, config, rngs):
        """Test gradient with respect to input counts."""
        from diffbio.pipelines.single_cell import SingleCellPipeline

        pipeline = SingleCellPipeline(config, rngs=rngs)

        def loss_fn(counts, ambient, batch):
            data = {
                "counts": counts,
                "ambient_profile": ambient,
                "batch_labels": batch,
            }
            result, _, _ = pipeline.apply(data, {}, None)
            return result["normalized"].sum()

        n_cells = 20
        n_genes = config.n_genes

        key = jax.random.key(0)
        counts = jax.random.uniform(key, shape=(n_cells, n_genes), minval=1, maxval=10)
        ambient = counts.mean(axis=0)
        batch = jnp.zeros(n_cells, dtype=jnp.int32)

        grad = jax.grad(loss_fn)(counts, ambient, batch)

        assert grad.shape == counts.shape
        assert jnp.all(jnp.isfinite(grad))


class TestSingleCellPipelineJITCompatibility:
    """Tests for JIT compilation compatibility."""

    @pytest.fixture
    def config(self):
        """Provide config for JIT tests."""
        from diffbio.pipelines.single_cell import SingleCellPipelineConfig

        return SingleCellPipelineConfig(
            n_genes=50,
            n_clusters=3,
            latent_dim=8,
        )

    def test_jit_apply(self, config, rngs):
        """Test JIT compilation of apply method."""
        from diffbio.pipelines.single_cell import SingleCellPipeline

        pipeline = SingleCellPipeline(config, rngs=rngs)

        @jax.jit
        def jit_apply(counts, ambient, batch):
            data = {
                "counts": counts,
                "ambient_profile": ambient,
                "batch_labels": batch,
            }
            result, _, _ = pipeline.apply(data, {}, None)
            return result["cluster_assignments"]

        n_cells = 20
        n_genes = config.n_genes

        key = jax.random.key(0)
        counts = jax.random.poisson(key, lam=5.0, shape=(n_cells, n_genes)).astype(jnp.float32)
        ambient = counts.mean(axis=0)
        batch = jnp.zeros(n_cells, dtype=jnp.int32)

        # Should compile and run without error
        result = jit_apply(counts, ambient, batch)
        assert result.shape == (n_cells, config.n_clusters)


class TestCreateSingleCellPipeline:
    """Tests for factory function."""

    def test_create_pipeline_default(self):
        """Test factory function with defaults."""
        from diffbio.pipelines.single_cell import create_single_cell_pipeline

        pipeline = create_single_cell_pipeline()
        assert pipeline is not None

    def test_create_pipeline_custom(self):
        """Test factory function with custom parameters."""
        from diffbio.pipelines.single_cell import create_single_cell_pipeline

        pipeline = create_single_cell_pipeline(
            n_genes=1000,
            n_clusters=15,
            latent_dim=32,
            seed=123,
        )

        assert pipeline.config.n_genes == 1000
        assert pipeline.config.n_clusters == 15
        assert pipeline.config.latent_dim == 32
