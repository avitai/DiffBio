"""Tests for diffbio.operators.singlecell.ambient_removal module.

These tests define the expected behavior of the DifferentiableAmbientRemoval
operator for removing ambient RNA contamination from single-cell data.
"""

import jax
import jax.numpy as jnp
import pytest
from artifex.generative_models.core.base import MLP
from flax import nnx

from diffbio.operators.singlecell.ambient_removal import (
    AmbientRemovalConfig,
    DifferentiableAmbientRemoval,
)


class TestAmbientRemovalConfig:
    """Tests for AmbientRemovalConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AmbientRemovalConfig()
        assert config.n_genes == 2000
        assert config.latent_dim == 64
        assert config.ambient_prior == 0.01
        assert config.stochastic is True
        assert config.stream_name == "sample"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AmbientRemovalConfig(
            n_genes=5000,
            latent_dim=128,
            ambient_prior=0.05,
        )
        assert config.n_genes == 5000
        assert config.latent_dim == 128
        assert config.ambient_prior == 0.05

    def test_hidden_dims_must_be_non_empty(self):
        """Ambient removal should fail fast when no hidden backbone is configured."""
        with pytest.raises(ValueError, match="hidden_dims"):
            AmbientRemovalConfig(hidden_dims=[])


class TestDifferentiableAmbientRemoval:
    """Tests for DifferentiableAmbientRemoval operator."""

    @pytest.fixture
    def small_config(self):
        """Provide small config for faster tests."""
        return AmbientRemovalConfig(
            n_genes=100,
            latent_dim=32,
            hidden_dims=[64, 32],
            stochastic=False,
            stream_name=None,
        )

    @pytest.fixture
    def sample_data(self):
        """Provide sample single-cell count data."""
        key = jax.random.key(0)
        n_cells = 50
        n_genes = 100

        # Simulated count data (Poisson-like)
        key, subkey = jax.random.split(key)
        counts = jax.random.poisson(subkey, 10.0, (n_cells, n_genes)).astype(jnp.float32)

        # Ambient profile (normalized gene expression in empty droplets)
        key, subkey = jax.random.split(key)
        ambient_profile = jax.nn.softmax(jax.random.normal(subkey, (n_genes,)))

        return {
            "counts": counts,
            "ambient_profile": ambient_profile,
        }

    def test_initialization(self, rngs, small_config):
        """Test operator initialization."""
        op = DifferentiableAmbientRemoval(small_config, rngs=rngs)
        assert op is not None
        assert isinstance(op.encoder.backbone, MLP)
        assert isinstance(op.decoder.backbone, MLP)
        assert len(op.encoder.backbone.layers) == len(small_config.hidden_dims)
        assert len(op.decoder.backbone.layers) == len(small_config.hidden_dims)

    def test_output_contains_decontaminated(self, rngs, small_config, sample_data):
        """Test that output contains decontaminated counts."""
        op = DifferentiableAmbientRemoval(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "decontaminated_counts" in transformed
        assert transformed["decontaminated_counts"].shape == (50, 100)

    def test_output_contains_contamination_fraction(self, rngs, small_config, sample_data):
        """Test that output contains estimated contamination fraction."""
        op = DifferentiableAmbientRemoval(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "contamination_fraction" in transformed
        # One contamination fraction per cell
        assert transformed["contamination_fraction"].shape == (50,)

    def test_contamination_fraction_bounded(self, rngs, small_config, sample_data):
        """Test that contamination fraction is between 0 and 1."""
        op = DifferentiableAmbientRemoval(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        frac = transformed["contamination_fraction"]
        assert jnp.all(frac >= 0)
        assert jnp.all(frac <= 1)

    def test_output_contains_latent(self, rngs, small_config, sample_data):
        """Test that output contains latent representation."""
        op = DifferentiableAmbientRemoval(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "latent" in transformed
        assert transformed["latent"].shape == (50, 32)  # latent_dim

    def test_output_finite(self, rngs, small_config, sample_data):
        """Test that outputs are finite."""
        op = DifferentiableAmbientRemoval(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert jnp.isfinite(transformed["decontaminated_counts"]).all()
        assert jnp.isfinite(transformed["contamination_fraction"]).all()
        assert jnp.isfinite(transformed["latent"]).all()

    def test_decontaminated_non_negative(self, rngs, small_config, sample_data):
        """Test that decontaminated counts are non-negative."""
        op = DifferentiableAmbientRemoval(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert jnp.all(transformed["decontaminated_counts"] >= 0)


class TestGradientFlow:
    """Tests for gradient flow through ambient removal."""

    @pytest.fixture
    def small_config(self):
        return AmbientRemovalConfig(
            n_genes=50,
            latent_dim=16,
            hidden_dims=[32],
            stochastic=False,
            stream_name=None,
        )

    def test_gradient_flows_through_removal(self, rngs, small_config):
        """Test that gradients flow through ambient removal."""
        op = DifferentiableAmbientRemoval(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_cells = 20
        n_genes = 50

        key, subkey = jax.random.split(key)
        counts = jax.random.poisson(subkey, 10.0, (n_cells, n_genes)).astype(jnp.float32)

        key, subkey = jax.random.split(key)
        ambient = jax.nn.softmax(jax.random.normal(subkey, (n_genes,)))

        def loss_fn(c):
            data = {"counts": c, "ambient_profile": ambient}
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["decontaminated_counts"].sum()

        grad = jax.grad(loss_fn)(counts)
        assert grad is not None
        assert grad.shape == counts.shape
        assert jnp.isfinite(grad).all()

    def test_model_is_learnable(self, rngs, small_config):
        """Test that model parameters are learnable."""
        op = DifferentiableAmbientRemoval(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_cells = 20
        n_genes = 50

        key, subkey = jax.random.split(key)
        counts = jax.random.poisson(subkey, 10.0, (n_cells, n_genes)).astype(jnp.float32)

        key, subkey = jax.random.split(key)
        ambient = jax.nn.softmax(jax.random.normal(subkey, (n_genes,)))

        data = {"counts": counts, "ambient_profile": ambient}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["reconstructed"].sum()

        loss, grads = loss_fn(op)

        assert loss is not None
        assert hasattr(grads, "encoder")
        assert hasattr(grads, "decoder")
        assert jnp.any(grads.encoder.backbone.layers[0].kernel[...] != 0.0)
        assert jnp.any(grads.decoder.backbone.layers[0].kernel[...] != 0.0)


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def small_config(self):
        return AmbientRemovalConfig(
            n_genes=50,
            latent_dim=16,
            hidden_dims=[32],
            stochastic=False,
            stream_name=None,
        )

    def test_apply_is_jit_compatible(self, rngs, small_config):
        """Test that apply method works with JIT."""
        op = DifferentiableAmbientRemoval(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_cells = 20
        n_genes = 50

        key, subkey = jax.random.split(key)
        counts = jax.random.poisson(subkey, 10.0, (n_cells, n_genes)).astype(jnp.float32)

        key, subkey = jax.random.split(key)
        ambient = jax.nn.softmax(jax.random.normal(subkey, (n_genes,)))

        data = {"counts": counts, "ambient_profile": ambient}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["decontaminated_counts"]).all()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_few_cells(self, rngs):
        """Test with few cells."""
        config = AmbientRemovalConfig(
            n_genes=30,
            latent_dim=8,
            hidden_dims=[16],
            stochastic=False,
            stream_name=None,
        )
        op = DifferentiableAmbientRemoval(config, rngs=rngs)

        key = jax.random.key(0)
        counts = jax.random.poisson(key, 10.0, (5, 30)).astype(jnp.float32)
        ambient = jax.nn.softmax(jax.random.normal(key, (30,)))

        data = {"counts": counts, "ambient_profile": ambient}
        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["decontaminated_counts"]).all()

    def test_high_ambient_prior(self, rngs):
        """Test with high ambient prior."""
        config = AmbientRemovalConfig(
            n_genes=30,
            latent_dim=8,
            hidden_dims=[16],
            ambient_prior=0.5,
            stochastic=False,
            stream_name=None,
        )
        op = DifferentiableAmbientRemoval(config, rngs=rngs)

        key = jax.random.key(0)
        counts = jax.random.poisson(key, 10.0, (10, 30)).astype(jnp.float32)
        ambient = jax.nn.softmax(jax.random.normal(key, (30,)))

        data = {"counts": counts, "ambient_profile": ambient}
        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["decontaminated_counts"]).all()
