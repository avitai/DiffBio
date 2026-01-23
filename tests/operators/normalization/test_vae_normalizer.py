"""Tests for diffbio.operators.normalization.vae_normalizer module.

These tests define the expected behavior of the VAENormalizer
operator. Implementation should be written to pass these tests.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.normalization.vae_normalizer import (
    VAENormalizer,
    VAENormalizerConfig,
)


class TestVAENormalizerConfig:
    """Tests for VAENormalizerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VAENormalizerConfig()
        assert config.latent_dim == 10
        assert config.n_genes == 2000
        assert config.stochastic is True
        assert config.stream_name == "sample"

    def test_custom_latent_dim(self):
        """Test custom latent dimension."""
        config = VAENormalizerConfig(latent_dim=32)
        assert config.latent_dim == 32

    def test_custom_architecture(self):
        """Test custom architecture parameters."""
        config = VAENormalizerConfig(hidden_dims=[256, 128, 64], n_genes=5000)
        assert config.hidden_dims == [256, 128, 64]
        assert config.n_genes == 5000


class TestVAENormalizer:
    """Tests for VAENormalizer operator."""

    @pytest.fixture
    def sample_counts(self):
        """Provide sample count data."""
        # Simulate gene expression counts for a single cell
        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        return {"counts": counts, "library_size": jnp.sum(counts)}

    @pytest.fixture
    def batch_counts(self):
        """Provide batch of count data."""
        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(8, 100)).astype(jnp.float32)
        library_sizes = jnp.sum(counts, axis=1)
        return {"counts": counts, "library_size": library_sizes}

    def test_initialization(self, rngs):
        """Test operator initialization."""
        config = VAENormalizerConfig(n_genes=100)
        op = VAENormalizer(config, rngs=rngs)
        assert op is not None
        assert op.latent_dim == 10

    def test_initialization_custom_architecture(self, rngs):
        """Test initialization with custom architecture."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=20, hidden_dims=[64, 32])
        op = VAENormalizer(config, rngs=rngs)
        assert op.latent_dim == 20

    def test_encode_output_shape(self, rngs, sample_counts):
        """Test that encoder produces correct latent shape."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        mean, logvar = op.encode(sample_counts["counts"])

        assert mean.shape == (10,)
        assert logvar.shape == (10,)

    def test_decode_output_shape(self, rngs, sample_counts):
        """Test that decoder produces correct output shape."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        z = jnp.ones(10)
        library_size = sample_counts["library_size"]
        log_rate = op.decode(z, library_size)

        assert log_rate.shape == (100,)

    def test_reparameterize(self, rngs):
        """Test reparameterization produces valid samples."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        mean = jnp.zeros(10)
        logvar = jnp.zeros(10)

        # Uses inherited reparameterize() from EncoderDecoderOperator
        # (uses self.rngs internally, no key argument)
        z = op.reparameterize(mean, logvar)

        assert z.shape == (10,)
        # z should be different from mean due to sampling
        # (unless logvar is very negative)

    def test_apply_returns_normalized(self, rngs, sample_counts):
        """Test that apply returns normalized expression."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_counts, {}, None, None)

        assert "normalized" in transformed_data
        assert transformed_data["normalized"].shape == sample_counts["counts"].shape

    def test_apply_returns_latent(self, rngs, sample_counts):
        """Test that apply returns latent representation."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_counts, {}, None, None)

        assert "latent_z" in transformed_data
        assert transformed_data["latent_z"].shape == (10,)

    def test_apply_returns_reconstruction_params(self, rngs, sample_counts):
        """Test that apply returns reconstruction parameters."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_counts, {}, None, None)

        assert "log_rate" in transformed_data
        assert "latent_mean" in transformed_data
        assert "latent_logvar" in transformed_data

    def test_apply_preserves_counts(self, rngs, sample_counts):
        """Test that apply preserves original counts."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_counts, {}, None, None)

        assert "counts" in transformed_data
        assert jnp.allclose(transformed_data["counts"], sample_counts["counts"])


class TestVAELoss:
    """Tests for VAE loss computation."""

    def test_elbo_loss_computable(self, rngs):
        """Test that ELBO loss can be computed."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)

        # Uses inherited reparameterize() internally (no key argument needed)
        loss = op.compute_elbo_loss(counts, library_size)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_reconstruction_loss_non_negative(self, rngs):
        """Test that reconstruction loss is non-negative."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)

        # Get reconstruction
        mean, logvar = op.encode(counts)
        # Uses inherited reparameterize() from EncoderDecoderOperator (no key argument)
        z = op.reparameterize(mean, logvar)
        log_rate = op.decode(z, library_size)

        recon_loss = op.reconstruction_loss(counts, log_rate)

        assert recon_loss >= 0


class TestGradientFlow:
    """Tests for gradient flow through VAE normalizer."""

    def test_gradient_flows_through_apply(self, rngs):
        """Test that gradients flow through the apply method."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)
        state = {}

        def loss_fn(c):
            data = {"counts": c, "library_size": library_size}
            transformed, _, _ = op.apply(data, state, None, None)
            return jnp.sum(transformed["normalized"])

        grad = jax.grad(loss_fn)(counts)
        assert grad is not None
        assert grad.shape == counts.shape

    def test_gradient_flows_through_elbo(self, rngs):
        """Test that gradients flow through ELBO loss."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)

        # Uses inherited reparameterize() internally (no key argument needed)
        def loss_fn(c):
            return op.compute_elbo_loss(c, library_size)

        grad = jax.grad(loss_fn)(counts)
        assert grad is not None
        assert grad.shape == counts.shape

    def test_encoder_is_learnable(self, rngs):
        """Test that encoder parameters are learnable."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["normalized"])

        loss, grads = loss_fn(op)

        assert hasattr(grads, "encoder_layers")

    def test_decoder_is_learnable(self, rngs):
        """Test that decoder parameters are learnable."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["normalized"])

        loss, grads = loss_fn(op)

        assert hasattr(grads, "decoder_layers")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_apply_is_jit_compatible(self, rngs):
        """Test that apply method works with JIT."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, new_state, metadata = jit_apply(data, state)
        assert transformed["normalized"].shape == counts.shape

    def test_encode_is_jit_compatible(self, rngs):
        """Test that encode method works with JIT."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)

        @jax.jit
        def jit_encode(x):
            return op.encode(x)

        mean, logvar = jit_encode(counts)
        assert mean.shape == (10,)
        assert logvar.shape == (10,)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_counts(self, rngs):
        """Test with all zero counts."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jnp.zeros(100)
        library_size = jnp.array(1.0)  # Avoid division by zero
        data = {"counts": counts, "library_size": library_size}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["normalized"].shape == (100,)
        assert jnp.all(jnp.isfinite(transformed["normalized"]))

    def test_sparse_counts(self, rngs):
        """Test with sparse (mostly zero) counts."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jnp.zeros(100).at[0].set(100.0).at[50].set(50.0)
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["normalized"].shape == (100,)
        assert jnp.all(jnp.isfinite(transformed["normalized"]))

    def test_high_counts(self, rngs):
        """Test with high count values."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jnp.ones(100) * 10000.0
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["normalized"].shape == (100,)
        assert jnp.all(jnp.isfinite(transformed["normalized"]))

    def test_small_latent_dim(self, rngs):
        """Test with very small latent dimension."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=2)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["latent_z"].shape == (2,)

    def test_large_latent_dim(self, rngs):
        """Test with large latent dimension."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=50)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["latent_z"].shape == (50,)
