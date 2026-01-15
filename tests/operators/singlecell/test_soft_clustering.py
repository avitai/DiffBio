"""Tests for diffbio.operators.singlecell.soft_clustering module.

These tests define the expected behavior of the SoftKMeansClustering
operator for differentiable cell clustering.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.soft_clustering import (
    SoftKMeansClustering,
    SoftClusteringConfig,
)


class TestSoftClusteringConfig:
    """Tests for SoftClusteringConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SoftClusteringConfig()
        assert config.n_clusters == 10
        assert config.temperature == 1.0
        assert config.learnable_centroids is True
        assert config.stochastic is False

    def test_custom_clusters(self):
        """Test custom number of clusters."""
        config = SoftClusteringConfig(n_clusters=20)
        assert config.n_clusters == 20

    def test_custom_temperature(self):
        """Test custom temperature."""
        config = SoftClusteringConfig(temperature=0.5)
        assert config.temperature == 0.5


class TestSoftKMeansClustering:
    """Tests for SoftKMeansClustering operator."""

    @pytest.fixture
    def sample_cells(self):
        """Provide sample cell embeddings."""
        # Simulate cells in latent space
        key = jax.random.key(0)
        n_cells = 500
        n_features = 50
        embeddings = jax.random.normal(key, (n_cells, n_features))
        return {"embeddings": embeddings}

    @pytest.fixture
    def small_config(self):
        """Provide small config for faster tests."""
        return SoftClusteringConfig(
            n_clusters=5,
            n_features=50,
            temperature=1.0,
        )

    def test_initialization(self, rngs, small_config):
        """Test operator initialization."""
        op = SoftKMeansClustering(small_config, rngs=rngs)
        assert op is not None
        assert op.n_clusters == 5

    def test_output_contains_assignments(self, rngs, small_config, sample_cells):
        """Test that output contains soft cluster assignments."""
        op = SoftKMeansClustering(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_cells, {}, None, None)

        assert "cluster_assignments" in transformed
        assert transformed["cluster_assignments"].shape == (500, 5)

    def test_assignments_sum_to_one(self, rngs, small_config, sample_cells):
        """Test that cluster assignments sum to 1 per cell."""
        op = SoftKMeansClustering(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_cells, {}, None, None)

        assignments_sum = jnp.sum(transformed["cluster_assignments"], axis=-1)
        assert jnp.allclose(assignments_sum, 1.0, atol=1e-5)

    def test_output_contains_centroids(self, rngs, small_config, sample_cells):
        """Test that output contains cluster centroids."""
        op = SoftKMeansClustering(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_cells, {}, None, None)

        assert "centroids" in transformed
        assert transformed["centroids"].shape == (5, 50)

    def test_output_contains_hard_labels(self, rngs, small_config, sample_cells):
        """Test that output contains hard cluster labels."""
        op = SoftKMeansClustering(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_cells, {}, None, None)

        assert "cluster_labels" in transformed
        assert transformed["cluster_labels"].shape == (500,)

    def test_hard_labels_valid(self, rngs, small_config, sample_cells):
        """Test that hard labels are valid cluster indices."""
        op = SoftKMeansClustering(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_cells, {}, None, None)

        labels = transformed["cluster_labels"]
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < 5)


class TestGradientFlow:
    """Tests for gradient flow through clustering."""

    @pytest.fixture
    def small_config(self):
        return SoftClusteringConfig(
            n_clusters=5,
            n_features=50,
        )

    def test_gradient_flows_through_assignments(self, rngs, small_config):
        """Test that gradients flow through cluster assignments."""
        op = SoftKMeansClustering(small_config, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (100, 50))

        def loss_fn(emb):
            data = {"embeddings": emb}
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["cluster_assignments"][:, 0].sum()

        grad = jax.grad(loss_fn)(embeddings)
        assert grad is not None
        assert grad.shape == embeddings.shape
        assert jnp.isfinite(grad).all()

    def test_centroids_are_learnable(self, rngs, small_config):
        """Test that centroid parameters are learnable."""
        op = SoftKMeansClustering(small_config, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (100, 50))
        data = {"embeddings": embeddings}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["cluster_assignments"][:, 0].sum()

        loss, grads = loss_fn(op)

        assert hasattr(grads, "centroids")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def small_config(self):
        return SoftClusteringConfig(
            n_clusters=5,
            n_features=50,
        )

    def test_apply_is_jit_compatible(self, rngs, small_config):
        """Test that apply method works with JIT."""
        op = SoftKMeansClustering(small_config, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (100, 50))
        data = {"embeddings": embeddings}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["cluster_assignments"]).all()


class TestTemperatureControl:
    """Tests for temperature control."""

    def test_low_temperature_sharper_assignments(self, rngs):
        """Test that low temperature gives sharper assignments."""
        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (100, 50))
        data = {"embeddings": embeddings}

        config_high = SoftClusteringConfig(n_clusters=5, n_features=50, temperature=10.0)
        op_high = SoftKMeansClustering(config_high, rngs=rngs)

        config_low = SoftClusteringConfig(n_clusters=5, n_features=50, temperature=0.1)
        op_low = SoftKMeansClustering(config_low, rngs=nnx.Rngs(42))

        trans_high, _, _ = op_high.apply(data, {}, None, None)
        trans_low, _, _ = op_low.apply(data, {}, None, None)

        # Low temperature should have higher max assignment (more certain)
        max_high = jnp.max(trans_high["cluster_assignments"], axis=-1).mean()
        max_low = jnp.max(trans_low["cluster_assignments"], axis=-1).mean()
        assert max_low > max_high


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_cell(self, rngs):
        """Test with single cell."""
        config = SoftClusteringConfig(n_clusters=3, n_features=20)
        op = SoftKMeansClustering(config, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (1, 20))
        data = {"embeddings": embeddings}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["cluster_assignments"]).all()

    def test_many_clusters(self, rngs):
        """Test with many clusters."""
        config = SoftClusteringConfig(n_clusters=50, n_features=20)
        op = SoftKMeansClustering(config, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (100, 20))
        data = {"embeddings": embeddings}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["cluster_assignments"].shape == (100, 50)

    def test_high_dimensional(self, rngs):
        """Test with high-dimensional embeddings."""
        config = SoftClusteringConfig(n_clusters=5, n_features=500)
        op = SoftKMeansClustering(config, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (100, 500))
        data = {"embeddings": embeddings}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["cluster_assignments"]).all()
