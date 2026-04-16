"""Tests for diffbio.operators.multiomics.hic_contact module.

These tests define the expected behavior of the HiCContactAnalysis
operator for differentiable chromatin contact analysis.
"""

import jax
import jax.numpy as jnp
import pytest
from artifex.generative_models.core.base import MLP
from flax import nnx

from diffbio.operators.multiomics.hic_contact import (
    HiCContactAnalysis,
    HiCContactAnalysisConfig,
)


class TestHiCContactAnalysisConfig:
    """Tests for HiCContactAnalysisConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HiCContactAnalysisConfig()
        assert config.n_bins == 1000
        assert config.hidden_dim == 128
        assert config.num_layers == 3
        assert config.stochastic is False

    def test_custom_bins(self):
        """Test custom number of bins."""
        config = HiCContactAnalysisConfig(n_bins=2000)
        assert config.n_bins == 2000

    def test_custom_hidden_dim(self):
        """Test custom hidden dimension."""
        config = HiCContactAnalysisConfig(hidden_dim=256)
        assert config.hidden_dim == 256

    def test_num_layers_must_be_positive(self):
        """Hi-C contact analysis should fail fast without contact encoder layers."""
        with pytest.raises(ValueError, match="num_layers"):
            HiCContactAnalysisConfig(num_layers=0)


class TestHiCContactAnalysis:
    """Tests for HiCContactAnalysis operator."""

    @pytest.fixture
    def sample_data(self):
        """Provide sample Hi-C contact data."""
        key = jax.random.key(0)
        n_bins = 100

        # Contact matrix (n_bins, n_bins) - symmetric
        key, subkey = jax.random.split(key)
        raw_contacts = jax.random.exponential(subkey, (n_bins, n_bins))
        contact_matrix = (raw_contacts + raw_contacts.T) / 2

        # Bin genomic features (n_bins, n_features)
        key, subkey = jax.random.split(key)
        bin_features = jax.random.normal(subkey, (n_bins, 16))

        return {
            "contact_matrix": contact_matrix,
            "bin_features": bin_features,
        }

    @pytest.fixture
    def small_config(self):
        """Provide small config for faster tests."""
        return HiCContactAnalysisConfig(
            n_bins=100,
            hidden_dim=64,
            num_layers=2,
            bin_features=16,
        )

    def test_initialization(self, rngs, small_config):
        """Test operator initialization."""
        op = HiCContactAnalysis(small_config, rngs=rngs)
        assert op is not None
        assert isinstance(op.contact_encoder.backbone, MLP)
        assert isinstance(op.feature_encoder.backbone, MLP)
        assert len(op.contact_encoder.backbone.layers) == small_config.num_layers
        assert len(op.feature_encoder.backbone.layers) == 2

    def test_output_contains_compartments(self, rngs, small_config, sample_data):
        """Test that output contains compartment assignments."""
        op = HiCContactAnalysis(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "compartment_scores" in transformed
        assert transformed["compartment_scores"].shape[0] == 100  # n_bins

    def test_output_contains_tad_boundaries(self, rngs, small_config, sample_data):
        """Test that output contains TAD boundary scores."""
        op = HiCContactAnalysis(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "tad_boundary_scores" in transformed
        assert transformed["tad_boundary_scores"].shape == (100,)  # n_bins

    def test_boundary_scores_valid(self, rngs, small_config, sample_data):
        """Test that boundary scores are valid (0-1)."""
        op = HiCContactAnalysis(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        scores = transformed["tad_boundary_scores"]
        assert jnp.all(scores >= 0)
        assert jnp.all(scores <= 1)

    def test_output_contains_bin_embeddings(self, rngs, small_config, sample_data):
        """Test that output contains bin embeddings."""
        op = HiCContactAnalysis(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "bin_embeddings" in transformed
        assert transformed["bin_embeddings"].shape == (100, 64)  # hidden_dim

    def test_output_contains_predicted_contacts(self, rngs, small_config, sample_data):
        """Test that output contains predicted contacts."""
        op = HiCContactAnalysis(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "predicted_contacts" in transformed
        assert transformed["predicted_contacts"].shape == (100, 100)

    def test_output_finite(self, rngs, small_config, sample_data):
        """Test that outputs are finite."""
        op = HiCContactAnalysis(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert jnp.isfinite(transformed["compartment_scores"]).all()
        assert jnp.isfinite(transformed["tad_boundary_scores"]).all()
        assert jnp.isfinite(transformed["bin_embeddings"]).all()


class TestGradientFlow:
    """Tests for gradient flow through Hi-C contact analysis."""

    @pytest.fixture
    def small_config(self):
        return HiCContactAnalysisConfig(
            n_bins=50,
            hidden_dim=32,
            num_layers=1,
            bin_features=8,
        )

    def test_gradient_flows_through_analysis(self, rngs, small_config):
        """Test that gradients flow through Hi-C analysis."""
        op = HiCContactAnalysis(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_bins = 50

        key, subkey = jax.random.split(key)
        raw_contacts = jax.random.exponential(subkey, (n_bins, n_bins))
        contact_matrix = (raw_contacts + raw_contacts.T) / 2

        key, subkey = jax.random.split(key)
        bin_features = jax.random.normal(subkey, (n_bins, 8))

        def loss_fn(contacts):
            data = {
                "contact_matrix": contacts,
                "bin_features": bin_features,
            }
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["compartment_scores"].sum()

        grad = jax.grad(loss_fn)(contact_matrix)
        assert grad is not None
        assert grad.shape == contact_matrix.shape
        assert jnp.isfinite(grad).all()

    def test_model_is_learnable(self, rngs, small_config):
        """Test that model parameters are learnable."""
        op = HiCContactAnalysis(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_bins = 50

        key, subkey = jax.random.split(key)
        raw_contacts = jax.random.exponential(subkey, (n_bins, n_bins))
        contact_matrix = (raw_contacts + raw_contacts.T) / 2

        key, subkey = jax.random.split(key)
        bin_features = jax.random.normal(subkey, (n_bins, 8))

        data = {
            "contact_matrix": contact_matrix,
            "bin_features": bin_features,
        }
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["predicted_contacts"] ** 2)

        loss, grads = loss_fn(op)

        assert loss is not None
        assert hasattr(grads, "contact_encoder")
        assert hasattr(grads, "feature_encoder")
        assert jnp.any(grads.contact_encoder.backbone.layers[0].kernel[...] != 0.0)
        assert jnp.any(grads.feature_encoder.backbone.layers[0].kernel[...] != 0.0)


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def small_config(self):
        return HiCContactAnalysisConfig(
            n_bins=50,
            hidden_dim=32,
            num_layers=1,
            bin_features=8,
        )

    def test_apply_is_jit_compatible(self, rngs, small_config):
        """Test that apply method works with JIT."""
        op = HiCContactAnalysis(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_bins = 50

        key, subkey = jax.random.split(key)
        raw_contacts = jax.random.exponential(subkey, (n_bins, n_bins))
        contact_matrix = (raw_contacts + raw_contacts.T) / 2

        key, subkey = jax.random.split(key)
        bin_features = jax.random.normal(subkey, (n_bins, 8))

        data = {
            "contact_matrix": contact_matrix,
            "bin_features": bin_features,
        }
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["compartment_scores"]).all()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_matrix(self, rngs):
        """Test with small contact matrix."""
        config = HiCContactAnalysisConfig(
            n_bins=20,
            hidden_dim=32,
            num_layers=1,
            bin_features=4,
        )
        op = HiCContactAnalysis(config, rngs=rngs)

        key = jax.random.key(0)
        n_bins = 20

        key, subkey = jax.random.split(key)
        raw_contacts = jax.random.exponential(subkey, (n_bins, n_bins))
        contact_matrix = (raw_contacts + raw_contacts.T) / 2

        key, subkey = jax.random.split(key)
        bin_features = jax.random.normal(subkey, (n_bins, 4))

        data = {
            "contact_matrix": contact_matrix,
            "bin_features": bin_features,
        }
        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["compartment_scores"]).all()

    def test_sparse_contacts(self, rngs):
        """Test with sparse contact matrix."""
        config = HiCContactAnalysisConfig(
            n_bins=50,
            hidden_dim=32,
            num_layers=1,
            bin_features=4,
        )
        op = HiCContactAnalysis(config, rngs=rngs)

        key = jax.random.key(0)
        n_bins = 50

        # Create sparse contacts (mostly zeros)
        key, subkey = jax.random.split(key)
        raw_contacts = jax.random.exponential(subkey, (n_bins, n_bins)) * 0.1
        mask = jax.random.bernoulli(subkey, 0.1, (n_bins, n_bins))
        contact_matrix = raw_contacts * mask
        contact_matrix = (contact_matrix + contact_matrix.T) / 2

        key, subkey = jax.random.split(key)
        bin_features = jax.random.normal(subkey, (n_bins, 4))

        data = {
            "contact_matrix": contact_matrix,
            "bin_features": bin_features,
        }
        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["compartment_scores"]).all()
