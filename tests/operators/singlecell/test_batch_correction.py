"""Tests for diffbio.operators.singlecell.batch_correction module.

These tests define the expected behavior of the DifferentiableHarmony
operator for differentiable batch correction.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.batch_correction import (
    DifferentiableHarmony,
    BatchCorrectionConfig,
)


class TestBatchCorrectionConfig:
    """Tests for BatchCorrectionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BatchCorrectionConfig()
        assert config.n_clusters == 100
        assert config.n_iterations == 10
        assert config.theta == 2.0
        assert config.stochastic is False

    def test_custom_clusters(self):
        """Test custom number of clusters."""
        config = BatchCorrectionConfig(n_clusters=50)
        assert config.n_clusters == 50

    def test_custom_iterations(self):
        """Test custom number of iterations."""
        config = BatchCorrectionConfig(n_iterations=20)
        assert config.n_iterations == 20


class TestDifferentiableHarmony:
    """Tests for DifferentiableHarmony operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNGs for operator initialization."""
        return nnx.Rngs(42)

    @pytest.fixture
    def sample_data(self):
        """Provide sample batch data."""
        key = jax.random.key(0)
        n_cells = 200
        n_features = 50
        n_batches = 3

        # Generate embeddings with batch effect
        embeddings = jax.random.normal(key, (n_cells, n_features))

        # Generate batch labels
        key, subkey = jax.random.split(key)
        batch_labels = jax.random.randint(subkey, (n_cells,), 0, n_batches)

        return {"embeddings": embeddings, "batch_labels": batch_labels}

    @pytest.fixture
    def small_config(self):
        """Provide small config for faster tests."""
        return BatchCorrectionConfig(
            n_clusters=10,
            n_features=50,
            n_batches=3,
            n_iterations=3,
        )

    def test_initialization(self, rngs, small_config):
        """Test operator initialization."""
        op = DifferentiableHarmony(small_config, rngs=rngs)
        assert op is not None

    def test_output_contains_corrected(self, rngs, small_config, sample_data):
        """Test that output contains corrected embeddings."""
        op = DifferentiableHarmony(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "corrected_embeddings" in transformed
        assert transformed["corrected_embeddings"].shape == (200, 50)

    def test_corrected_preserves_dimension(self, rngs, small_config, sample_data):
        """Test that correction preserves embedding dimension."""
        op = DifferentiableHarmony(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert transformed["corrected_embeddings"].shape == sample_data["embeddings"].shape

    def test_output_contains_cluster_assignments(self, rngs, small_config, sample_data):
        """Test that output contains cluster assignments."""
        op = DifferentiableHarmony(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "cluster_assignments" in transformed
        assert transformed["cluster_assignments"].shape[0] == 200

    def test_output_is_finite(self, rngs, small_config, sample_data):
        """Test that output values are finite."""
        op = DifferentiableHarmony(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert jnp.isfinite(transformed["corrected_embeddings"]).all()


class TestGradientFlow:
    """Tests for gradient flow through batch correction."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def small_config(self):
        return BatchCorrectionConfig(
            n_clusters=10,
            n_features=50,
            n_batches=3,
            n_iterations=2,
        )

    def test_gradient_flows_through_correction(self, rngs, small_config):
        """Test that gradients flow through batch correction."""
        op = DifferentiableHarmony(small_config, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (50, 50))
        key, subkey = jax.random.split(key)
        batch_labels = jax.random.randint(subkey, (50,), 0, 3)

        def loss_fn(emb):
            data = {"embeddings": emb, "batch_labels": batch_labels}
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["corrected_embeddings"].sum()

        grad = jax.grad(loss_fn)(embeddings)
        assert grad is not None
        assert grad.shape == embeddings.shape
        assert jnp.isfinite(grad).all()

    def test_cluster_centroids_learnable(self, rngs, small_config):
        """Test that cluster centroids are learnable."""
        op = DifferentiableHarmony(small_config, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (50, 50))
        key, subkey = jax.random.split(key)
        batch_labels = jax.random.randint(subkey, (50,), 0, 3)
        data = {"embeddings": embeddings, "batch_labels": batch_labels}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["corrected_embeddings"].sum()

        loss, grads = loss_fn(op)

        assert hasattr(grads, "cluster_centroids")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def small_config(self):
        return BatchCorrectionConfig(
            n_clusters=10,
            n_features=50,
            n_batches=3,
            n_iterations=2,
        )

    def test_apply_is_jit_compatible(self, rngs, small_config):
        """Test that apply method works with JIT."""
        op = DifferentiableHarmony(small_config, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (50, 50))
        key, subkey = jax.random.split(key)
        batch_labels = jax.random.randint(subkey, (50,), 0, 3)
        data = {"embeddings": embeddings, "batch_labels": batch_labels}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["corrected_embeddings"]).all()


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_single_batch(self, rngs):
        """Test with single batch (should be no-op)."""
        config = BatchCorrectionConfig(
            n_clusters=5,
            n_features=20,
            n_batches=1,
            n_iterations=2,
        )
        op = DifferentiableHarmony(config, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (50, 20))
        batch_labels = jnp.zeros(50, dtype=jnp.int32)
        data = {"embeddings": embeddings, "batch_labels": batch_labels}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["corrected_embeddings"]).all()

    def test_many_batches(self, rngs):
        """Test with many batches."""
        config = BatchCorrectionConfig(
            n_clusters=10,
            n_features=20,
            n_batches=10,
            n_iterations=2,
        )
        op = DifferentiableHarmony(config, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (100, 20))
        key, subkey = jax.random.split(key)
        batch_labels = jax.random.randint(subkey, (100,), 0, 10)
        data = {"embeddings": embeddings, "batch_labels": batch_labels}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["corrected_embeddings"]).all()

    def test_few_cells(self, rngs):
        """Test with few cells."""
        config = BatchCorrectionConfig(
            n_clusters=5,
            n_features=20,
            n_batches=2,
            n_iterations=2,
        )
        op = DifferentiableHarmony(config, rngs=rngs)

        key = jax.random.key(0)
        embeddings = jax.random.normal(key, (10, 20))
        key, subkey = jax.random.split(key)
        batch_labels = jax.random.randint(subkey, (10,), 0, 2)
        data = {"embeddings": embeddings, "batch_labels": batch_labels}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["corrected_embeddings"]).all()
