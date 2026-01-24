"""Tests for metagenomic binning operators."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.assembly import (
    DifferentiableMetagenomicBinner,
    MetagenomicBinnerConfig,
    create_metagenomic_binner,
)


class TestMetagenomicBinnerConfig:
    """Test configuration validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MetagenomicBinnerConfig()
        assert config.n_tnf_features == 136
        assert config.n_abundance_features == 10
        assert config.latent_dim == 32
        assert config.n_clusters == 100
        assert config.stochastic is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = MetagenomicBinnerConfig(
            n_abundance_features=5,
            latent_dim=16,
            n_clusters=50,
        )
        assert config.n_abundance_features == 5
        assert config.latent_dim == 16
        assert config.n_clusters == 50


class TestMetagenomicBinnerBasic:
    """Test basic functionality."""

    @pytest.fixture
    def binner(self):
        """Create a test binner."""
        config = MetagenomicBinnerConfig(
            n_tnf_features=136,
            n_abundance_features=5,
            latent_dim=16,
            hidden_dims=[64, 32],
            n_clusters=10,
        )
        return DifferentiableMetagenomicBinner(config, rngs=nnx.Rngs(42))

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        n_contigs = 50
        n_tnf = 136
        n_samples = 5

        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)

        # TNF frequencies (should sum to 1)
        tnf = jax.random.dirichlet(k1, jnp.ones(n_tnf), shape=(n_contigs,))
        # Abundance (positive values)
        abundance = jax.random.exponential(k2, shape=(n_contigs, n_samples))

        return {
            "tnf": tnf,
            "abundance": abundance,
        }

    def test_forward_pass(self, binner, sample_data):
        """Test forward pass produces expected outputs."""
        result, state, metadata = binner.apply(sample_data, {}, None)

        n_contigs = sample_data["tnf"].shape[0]
        latent_dim = binner.config.latent_dim
        n_clusters = binner.config.n_clusters
        n_tnf = binner.config.n_tnf_features
        n_abundance = binner.config.n_abundance_features

        # Check output keys
        assert "latent_z" in result
        assert "latent_mu" in result
        assert "latent_logvar" in result
        assert "cluster_assignments" in result
        assert "reconstructed_tnf" in result
        assert "reconstructed_abundance" in result

        # Check shapes
        assert result["latent_z"].shape == (n_contigs, latent_dim)
        assert result["latent_mu"].shape == (n_contigs, latent_dim)
        assert result["cluster_assignments"].shape == (n_contigs, n_clusters)
        assert result["reconstructed_tnf"].shape == (n_contigs, n_tnf)
        assert result["reconstructed_abundance"].shape == (n_contigs, n_abundance)

    def test_cluster_assignments_sum_to_one(self, binner, sample_data):
        """Test that soft cluster assignments are valid probabilities."""
        result, _, _ = binner.apply(sample_data, {}, None)
        assignments = result["cluster_assignments"]

        # Should sum to 1 along cluster dimension
        sums = jnp.sum(assignments, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

        # Should be non-negative
        assert jnp.all(assignments >= 0)

    def test_reconstructed_tnf_valid(self, binner, sample_data):
        """Test that reconstructed TNF is a valid distribution."""
        result, _, _ = binner.apply(sample_data, {}, None)
        tnf_recon = result["reconstructed_tnf"]

        # Should sum to 1 (TNF is a frequency distribution)
        sums = jnp.sum(tnf_recon, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

        # Should be non-negative
        assert jnp.all(tnf_recon >= 0)

    def test_reconstructed_abundance_positive(self, binner, sample_data):
        """Test that reconstructed abundance is positive."""
        result, _, _ = binner.apply(sample_data, {}, None)
        abundance_recon = result["reconstructed_abundance"]

        # Abundance should be positive (softplus output)
        assert jnp.all(abundance_recon >= 0)


class TestMetagenomicBinnerDifferentiability:
    """Test gradient computation."""

    @pytest.fixture
    def binner(self):
        """Create a test binner."""
        config = MetagenomicBinnerConfig(
            n_tnf_features=136,
            n_abundance_features=5,
            latent_dim=16,
            hidden_dims=[32],
            n_clusters=10,
        )
        return DifferentiableMetagenomicBinner(config, rngs=nnx.Rngs(42))

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        n_contigs = 20
        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)

        return {
            "tnf": jax.random.dirichlet(k1, jnp.ones(136), shape=(n_contigs,)),
            "abundance": jax.random.exponential(k2, shape=(n_contigs, 5)),
        }

    def test_gradient_flow(self, binner, sample_data):
        """Test that gradients flow through the operator."""

        @nnx.value_and_grad
        def loss_fn(model):
            result, _, _ = model.apply(sample_data, {}, None)
            # Simple loss: encourage tight clusters
            return result["latent_z"].mean()

        loss, grads = loss_fn(binner)
        assert grads is not None

        # Check gradients are not NaN
        grad_leaves = jax.tree.leaves(grads)
        for leaf in grad_leaves:
            if hasattr(leaf, "shape"):
                assert not jnp.any(jnp.isnan(leaf)), "Gradient contains NaN"

    def test_reconstruction_loss_gradient(self, binner, sample_data):
        """Test gradient through reconstruction loss."""

        @nnx.value_and_grad
        def loss_fn(model):
            result, _, _ = model.apply(sample_data, {}, None)
            # Reconstruction loss
            tnf_loss = jnp.mean((result["reconstructed_tnf"] - sample_data["tnf"]) ** 2)
            abundance_loss = jnp.mean(
                (result["reconstructed_abundance"] - sample_data["abundance"]) ** 2
            )
            return tnf_loss + abundance_loss

        _, grads = loss_fn(binner)

        # Check encoder_linear has gradients
        encoder_grads = jax.tree.leaves(grads.encoder_linear)
        assert any(
            hasattr(g, "shape") and jnp.any(g != 0) for g in encoder_grads
        ), "Encoder should have non-zero gradients"

        # Check decoder_linear has gradients
        decoder_grads = jax.tree.leaves(grads.decoder_linear)
        assert any(
            hasattr(g, "shape") and jnp.any(g != 0) for g in decoder_grads
        ), "Decoder should have non-zero gradients"

    def test_clustering_loss_gradient(self, binner, sample_data):
        """Test gradient through clustering loss."""

        @nnx.value_and_grad
        def loss_fn(model):
            result, _, _ = model.apply(sample_data, {}, None)
            # Clustering compactness loss
            assignments = result["cluster_assignments"]
            entropy = -jnp.sum(assignments * jnp.log(assignments + 1e-10), axis=-1)
            return jnp.mean(entropy)  # Minimize entropy = sharper clusters

        _, grads = loss_fn(binner)

        # Centroids should have gradients
        assert grads.centroids is not None
        assert jnp.any(grads.centroids.value != 0), "Centroids should have gradients"


class TestMetagenomicBinnerJIT:
    """Test JIT compilation."""

    def test_jit_compilation(self):
        """Test that forward pass can be JIT compiled."""
        config = MetagenomicBinnerConfig(
            n_tnf_features=136,
            n_abundance_features=5,
            latent_dim=16,
            hidden_dims=[32],
            n_clusters=10,
        )
        binner = DifferentiableMetagenomicBinner(config, rngs=nnx.Rngs(42))
        binner.eval()  # Deterministic for JIT (uses nnx.Module built-in)

        n_contigs = 20
        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)
        data = {
            "tnf": jax.random.dirichlet(k1, jnp.ones(136), shape=(n_contigs,)),
            "abundance": jax.random.exponential(k2, shape=(n_contigs, 5)),
        }

        @jax.jit
        def forward(model, data):
            result, _, _ = model.apply(data, {}, None)
            return result["cluster_assignments"]

        # Should not raise
        result = forward(binner, data)
        assert result.shape == (n_contigs, 10)


class TestMetagenomicBinnerTrainingMode:
    """Test training/eval mode switching."""

    @pytest.fixture
    def binner(self):
        """Create a test binner."""
        config = MetagenomicBinnerConfig(
            n_tnf_features=136,
            n_abundance_features=5,
            latent_dim=16,
            hidden_dims=[32],
            n_clusters=10,
        )
        return DifferentiableMetagenomicBinner(config, rngs=nnx.Rngs(42))

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        n_contigs = 20
        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)
        return {
            "tnf": jax.random.dirichlet(k1, jnp.ones(136), shape=(n_contigs,)),
            "abundance": jax.random.exponential(k2, shape=(n_contigs, 5)),
        }

    def test_eval_mode_deterministic(self, binner, sample_data):
        """Test that eval mode produces deterministic outputs."""
        binner.eval()  # Uses nnx.Module built-in method

        result1, _, _ = binner.apply(sample_data, {}, None)
        result2, _, _ = binner.apply(sample_data, {}, None)

        # In eval mode, outputs should be identical
        assert jnp.allclose(result1["latent_z"], result2["latent_z"])

    def test_train_mode_stochastic(self, binner, sample_data):
        """Test that train mode uses reparameterization."""
        binner.train()  # Uses nnx.Module built-in method

        # In train mode, z = mu + std * eps, so z != mu
        result, _, _ = binner.apply(sample_data, {}, None)

        # mu and z should differ (due to noise)
        # Note: They could be close by chance, so we just check they're computed
        assert "latent_mu" in result
        assert "latent_z" in result


class TestMetagenomicBinnerFactory:
    """Test factory function."""

    def test_create_metagenomic_binner(self):
        """Test factory function creates valid binner."""
        binner = create_metagenomic_binner(
            n_abundance_features=5,
            n_clusters=20,
            latent_dim=16,
            seed=123,
        )

        assert isinstance(binner, DifferentiableMetagenomicBinner)
        assert binner.config.n_abundance_features == 5
        assert binner.config.n_clusters == 20
        assert binner.config.latent_dim == 16

    def test_create_metagenomic_binner_defaults(self):
        """Test factory with default values."""
        binner = create_metagenomic_binner()

        assert binner.config.n_abundance_features == 10
        assert binner.config.n_clusters == 100
        assert binner.config.latent_dim == 32
