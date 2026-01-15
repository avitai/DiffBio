"""Tests for differentiable differential expression pipeline.

Following TDD principles, these tests define the expected behavior
of the DifferentialExpressionPipeline.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest


class TestDEPipelineConfig:
    """Tests for DEPipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from diffbio.pipelines.differential_expression import DEPipelineConfig

        config = DEPipelineConfig(stream_name=None)

        assert config.n_genes == 1000
        assert config.n_conditions == 2
        assert config.alpha == 0.05
        assert config.stochastic is False

    def test_custom_config(self):
        """Test custom configuration values."""
        from diffbio.pipelines.differential_expression import DEPipelineConfig

        config = DEPipelineConfig(
            n_genes=5000,
            n_conditions=3,
            alpha=0.01,
            stream_name=None,
        )

        assert config.n_genes == 5000
        assert config.n_conditions == 3
        assert config.alpha == 0.01


class TestDifferentialExpressionPipeline:
    """Tests for DifferentialExpressionPipeline."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.pipelines.differential_expression import DEPipelineConfig

        return DEPipelineConfig(
            n_genes=100,
            n_conditions=2,
            alpha=0.05,
            stream_name=None,
        )

    @pytest.fixture
    def pipeline(self, config, rngs):
        """Create pipeline instance."""
        from diffbio.pipelines.differential_expression import (
            DifferentialExpressionPipeline,
        )

        return DifferentialExpressionPipeline(config, rngs=rngs)

    def test_initialization(self, config, rngs):
        """Test pipeline initialization."""
        from diffbio.pipelines.differential_expression import (
            DifferentialExpressionPipeline,
        )

        pipeline = DifferentialExpressionPipeline(config, rngs=rngs)

        assert pipeline.config == config
        assert hasattr(pipeline, "nb_glm")

    def test_initialization_without_rngs(self, config):
        """Test initialization without providing RNGs."""
        from diffbio.pipelines.differential_expression import (
            DifferentialExpressionPipeline,
        )

        pipeline = DifferentialExpressionPipeline(config, rngs=None)
        assert pipeline is not None

    def test_apply_basic(self, pipeline, config):
        """Test basic apply with count matrix."""
        n_samples = 10
        n_genes = config.n_genes

        # Create synthetic count data
        counts = jax.random.poisson(
            jax.random.key(0),
            lam=50.0,
            shape=(n_samples, n_genes),
        ).astype(jnp.float32)

        # Design matrix: simple case/control
        design = jnp.array(
            [
                [1, 0],  # intercept, treatment
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
            ],
            dtype=jnp.float32,
        )

        data = {"counts": counts, "design": design}
        result, state, metadata = pipeline.apply(data, {}, None)

        assert "log_fold_change" in result
        assert "p_values" in result
        assert "significant" in result

        assert result["log_fold_change"].shape == (n_genes,)
        assert result["p_values"].shape == (n_genes,)
        assert result["significant"].shape == (n_genes,)

    def test_p_values_in_range(self, pipeline, config):
        """Test that p-values are in [0, 1]."""
        n_samples = 10
        counts = jax.random.poisson(
            jax.random.key(0),
            lam=50.0,
            shape=(n_samples, config.n_genes),
        ).astype(jnp.float32)

        design = jnp.concatenate(
            [
                jnp.ones((n_samples, 1)),
                jnp.array([[0]] * 5 + [[1]] * 5),
            ],
            axis=1,
        ).astype(jnp.float32)

        data = {"counts": counts, "design": design}
        result, _, _ = pipeline.apply(data, {}, None)

        p_values = result["p_values"]
        assert jnp.all(p_values >= 0.0)
        assert jnp.all(p_values <= 1.0)

    def test_output_finite(self, pipeline, config):
        """Test that all outputs are finite."""
        n_samples = 10
        counts = jax.random.poisson(
            jax.random.key(0),
            lam=50.0,
            shape=(n_samples, config.n_genes),
        ).astype(jnp.float32)

        design = jnp.concatenate(
            [
                jnp.ones((n_samples, 1)),
                jnp.array([[0]] * 5 + [[1]] * 5),
            ],
            axis=1,
        ).astype(jnp.float32)

        data = {"counts": counts, "design": design}
        result, _, _ = pipeline.apply(data, {}, None)

        assert jnp.all(jnp.isfinite(result["log_fold_change"]))
        assert jnp.all(jnp.isfinite(result["p_values"]))

    def test_preserves_original_data(self, pipeline, config):
        """Test that original data is preserved in output."""
        n_samples = 10
        counts = jax.random.poisson(
            jax.random.key(0),
            lam=50.0,
            shape=(n_samples, config.n_genes),
        ).astype(jnp.float32)

        design = jnp.concatenate(
            [
                jnp.ones((n_samples, 1)),
                jnp.array([[0]] * 5 + [[1]] * 5),
            ],
            axis=1,
        ).astype(jnp.float32)

        extra_data = jnp.array([1.0, 2.0, 3.0])

        data = {"counts": counts, "design": design, "extra": extra_data}
        result, _, _ = pipeline.apply(data, {}, None)

        assert "extra" in result
        assert jnp.allclose(result["extra"], extra_data)


class TestDEPipelineDifferentiability:
    """Tests for gradient flow through the DE pipeline."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.pipelines.differential_expression import DEPipelineConfig

        return DEPipelineConfig(
            n_genes=50,
            n_conditions=2,
            stream_name=None,
        )

    def test_gradient_flow_through_pipeline(self, config, rngs):
        """Test that gradients flow through the pipeline."""
        from diffbio.pipelines.differential_expression import (
            DifferentialExpressionPipeline,
        )

        pipeline = DifferentialExpressionPipeline(config, rngs=rngs)

        def loss_fn(pipe, counts, design):
            data = {"counts": counts, "design": design}
            result, _, _ = pipe.apply(data, {}, None)
            return result["log_fold_change"].sum()

        n_samples = 10
        counts = jax.random.poisson(
            jax.random.key(0),
            lam=50.0,
            shape=(n_samples, config.n_genes),
        ).astype(jnp.float32)

        design = jnp.concatenate(
            [
                jnp.ones((n_samples, 1)),
                jnp.array([[0]] * 5 + [[1]] * 5),
            ],
            axis=1,
        ).astype(jnp.float32)

        grads = nnx.grad(loss_fn)(pipeline, counts, design)
        assert grads is not None

    def test_gradient_wrt_input_counts(self, config, rngs):
        """Test gradient with respect to input counts."""
        from diffbio.pipelines.differential_expression import (
            DifferentialExpressionPipeline,
        )

        pipeline = DifferentialExpressionPipeline(config, rngs=rngs)

        def loss_fn(counts, design):
            data = {"counts": counts, "design": design}
            result, _, _ = pipeline.apply(data, {}, None)
            # Use predicted mean for loss to ensure gradient flow
            return result["predicted_mean"].sum()

        n_samples = 10
        counts = jax.random.poisson(
            jax.random.key(0),
            lam=50.0,
            shape=(n_samples, config.n_genes),
        ).astype(jnp.float32)

        design = jnp.concatenate(
            [
                jnp.ones((n_samples, 1)),
                jnp.array([[0]] * 5 + [[1]] * 5),
            ],
            axis=1,
        ).astype(jnp.float32)

        grad = jax.grad(loss_fn)(counts, design)
        assert grad.shape == counts.shape


class TestDEPipelineJITCompatibility:
    """Tests for JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide default config."""
        from diffbio.pipelines.differential_expression import DEPipelineConfig

        return DEPipelineConfig(
            n_genes=50,
            n_conditions=2,
            stream_name=None,
        )

    def test_jit_apply(self, config, rngs):
        """Test JIT compilation of apply method."""
        from diffbio.pipelines.differential_expression import (
            DifferentialExpressionPipeline,
        )

        pipeline = DifferentialExpressionPipeline(config, rngs=rngs)

        @jax.jit
        def jit_apply(counts, design):
            data = {"counts": counts, "design": design}
            result, _, _ = pipeline.apply(data, {}, None)
            return result["log_fold_change"]

        n_samples = 10
        counts = jax.random.poisson(
            jax.random.key(0),
            lam=50.0,
            shape=(n_samples, config.n_genes),
        ).astype(jnp.float32)

        design = jnp.concatenate(
            [
                jnp.ones((n_samples, 1)),
                jnp.array([[0]] * 5 + [[1]] * 5),
            ],
            axis=1,
        ).astype(jnp.float32)

        # Should compile and run without error
        result = jit_apply(counts, design)
        assert result.shape == (config.n_genes,)


class TestSizeFactor:
    """Tests for size factor computation."""

    @pytest.fixture
    def rngs(self):
        """Provide RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def config(self):
        """Provide config."""
        from diffbio.pipelines.differential_expression import DEPipelineConfig

        return DEPipelineConfig(
            n_genes=100,
            n_conditions=2,
            stream_name=None,
        )

    def test_size_factors_positive(self, config, rngs):
        """Test that size factors are positive."""
        from diffbio.pipelines.differential_expression import (
            DifferentialExpressionPipeline,
        )

        pipeline = DifferentialExpressionPipeline(config, rngs=rngs)

        n_samples = 10
        counts = jax.random.poisson(
            jax.random.key(0),
            lam=50.0,
            shape=(n_samples, config.n_genes),
        ).astype(jnp.float32)

        design = jnp.concatenate(
            [
                jnp.ones((n_samples, 1)),
                jnp.array([[0]] * 5 + [[1]] * 5),
            ],
            axis=1,
        ).astype(jnp.float32)

        data = {"counts": counts, "design": design}
        result, _, _ = pipeline.apply(data, {}, None)

        if "size_factors" in result:
            assert jnp.all(result["size_factors"] > 0)
