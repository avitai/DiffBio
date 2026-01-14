"""Tests for diffbio.operators.multiomics.spatial_deconvolution module.

These tests define the expected behavior of the SpatialDeconvolution
operator for differentiable cell type deconvolution of spatial transcriptomics.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.multiomics.spatial_deconvolution import (
    SpatialDeconvolution,
    SpatialDeconvolutionConfig,
)


class TestSpatialDeconvolutionConfig:
    """Tests for SpatialDeconvolutionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpatialDeconvolutionConfig()
        assert config.n_genes == 2000
        assert config.n_cell_types == 10
        assert config.hidden_dim == 128
        assert config.stochastic is False

    def test_custom_genes(self):
        """Test custom number of genes."""
        config = SpatialDeconvolutionConfig(n_genes=5000)
        assert config.n_genes == 5000

    def test_custom_cell_types(self):
        """Test custom number of cell types."""
        config = SpatialDeconvolutionConfig(n_cell_types=20)
        assert config.n_cell_types == 20

    def test_custom_hidden_dim(self):
        """Test custom hidden dimension."""
        config = SpatialDeconvolutionConfig(hidden_dim=256)
        assert config.hidden_dim == 256


class TestSpatialDeconvolution:
    """Tests for SpatialDeconvolution operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNGs for operator initialization."""
        return nnx.Rngs(42)

    @pytest.fixture
    def sample_data(self):
        """Provide sample spatial transcriptomics data."""
        key = jax.random.key(0)
        n_spots = 100
        n_genes = 500
        n_cell_types = 5

        # Spot expression (n_spots, n_genes)
        key, subkey = jax.random.split(key)
        spot_expression = jax.nn.softplus(
            jax.random.normal(subkey, (n_spots, n_genes))
        )

        # Reference single-cell profiles (n_cell_types, n_genes)
        key, subkey = jax.random.split(key)
        reference_profiles = jax.nn.softplus(
            jax.random.normal(subkey, (n_cell_types, n_genes))
        )

        # Spot coordinates (n_spots, 2)
        key, subkey = jax.random.split(key)
        coordinates = jax.random.uniform(subkey, (n_spots, 2))

        return {
            "spot_expression": spot_expression,
            "reference_profiles": reference_profiles,
            "coordinates": coordinates,
        }

    @pytest.fixture
    def small_config(self):
        """Provide small config for faster tests."""
        return SpatialDeconvolutionConfig(
            n_genes=500,
            n_cell_types=5,
            hidden_dim=64,
            num_layers=2,
        )

    def test_initialization(self, rngs, small_config):
        """Test operator initialization."""
        op = SpatialDeconvolution(small_config, rngs=rngs)
        assert op is not None

    def test_output_contains_cell_proportions(self, rngs, small_config, sample_data):
        """Test that output contains cell type proportions."""
        op = SpatialDeconvolution(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "cell_proportions" in transformed
        assert transformed["cell_proportions"].shape == (100, 5)  # n_spots x n_cell_types

    def test_proportions_sum_to_one(self, rngs, small_config, sample_data):
        """Test that cell proportions sum to 1 per spot."""
        op = SpatialDeconvolution(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        prop_sum = jnp.sum(transformed["cell_proportions"], axis=-1)
        assert jnp.allclose(prop_sum, 1.0, atol=1e-5)

    def test_proportions_non_negative(self, rngs, small_config, sample_data):
        """Test that cell proportions are non-negative."""
        op = SpatialDeconvolution(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert jnp.all(transformed["cell_proportions"] >= 0)

    def test_output_contains_reconstructed(self, rngs, small_config, sample_data):
        """Test that output contains reconstructed expression."""
        op = SpatialDeconvolution(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "reconstructed_expression" in transformed
        assert transformed["reconstructed_expression"].shape == (100, 500)

    def test_output_contains_spatial_features(self, rngs, small_config, sample_data):
        """Test that output contains spatial features."""
        op = SpatialDeconvolution(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "spatial_embeddings" in transformed

    def test_output_finite(self, rngs, small_config, sample_data):
        """Test that outputs are finite."""
        op = SpatialDeconvolution(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert jnp.isfinite(transformed["cell_proportions"]).all()
        assert jnp.isfinite(transformed["reconstructed_expression"]).all()


class TestGradientFlow:
    """Tests for gradient flow through spatial deconvolution."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def small_config(self):
        return SpatialDeconvolutionConfig(
            n_genes=100,
            n_cell_types=3,
            hidden_dim=32,
            num_layers=1,
        )

    def test_gradient_flows_through_deconvolution(self, rngs, small_config):
        """Test that gradients flow through deconvolution."""
        op = SpatialDeconvolution(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_spots = 20

        key, subkey = jax.random.split(key)
        spot_expr = jax.nn.softplus(jax.random.normal(subkey, (n_spots, 100)))

        key, subkey = jax.random.split(key)
        ref_profiles = jax.nn.softplus(jax.random.normal(subkey, (3, 100)))

        key, subkey = jax.random.split(key)
        coords = jax.random.uniform(subkey, (n_spots, 2))

        def loss_fn(spot_input):
            data = {
                "spot_expression": spot_input,
                "reference_profiles": ref_profiles,
                "coordinates": coords,
            }
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["cell_proportions"].sum()

        grad = jax.grad(loss_fn)(spot_expr)
        assert grad is not None
        assert grad.shape == spot_expr.shape
        assert jnp.isfinite(grad).all()

    def test_model_is_learnable(self, rngs, small_config):
        """Test that model parameters are learnable."""
        op = SpatialDeconvolution(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_spots = 20

        key, subkey = jax.random.split(key)
        spot_expr = jax.nn.softplus(jax.random.normal(subkey, (n_spots, 100)))

        key, subkey = jax.random.split(key)
        ref_profiles = jax.nn.softplus(jax.random.normal(subkey, (3, 100)))

        key, subkey = jax.random.split(key)
        coords = jax.random.uniform(subkey, (n_spots, 2))

        data = {
            "spot_expression": spot_expr,
            "reference_profiles": ref_profiles,
            "coordinates": coords,
        }
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["cell_proportions"].sum()

        loss, grads = loss_fn(op)

        # Check encoder has gradients
        assert hasattr(grads, "spot_encoder")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def small_config(self):
        return SpatialDeconvolutionConfig(
            n_genes=100,
            n_cell_types=3,
            hidden_dim=32,
            num_layers=1,
        )

    def test_apply_is_jit_compatible(self, rngs, small_config):
        """Test that apply method works with JIT."""
        op = SpatialDeconvolution(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_spots = 20

        key, subkey = jax.random.split(key)
        spot_expr = jax.nn.softplus(jax.random.normal(subkey, (n_spots, 100)))

        key, subkey = jax.random.split(key)
        ref_profiles = jax.nn.softplus(jax.random.normal(subkey, (3, 100)))

        key, subkey = jax.random.split(key)
        coords = jax.random.uniform(subkey, (n_spots, 2))

        data = {
            "spot_expression": spot_expr,
            "reference_profiles": ref_profiles,
            "coordinates": coords,
        }
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["cell_proportions"]).all()


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_few_spots(self, rngs):
        """Test with few spots."""
        config = SpatialDeconvolutionConfig(
            n_genes=50,
            n_cell_types=3,
            hidden_dim=32,
            num_layers=1,
        )
        op = SpatialDeconvolution(config, rngs=rngs)

        key = jax.random.key(0)
        n_spots = 5

        key, subkey = jax.random.split(key)
        spot_expr = jax.nn.softplus(jax.random.normal(subkey, (n_spots, 50)))

        key, subkey = jax.random.split(key)
        ref_profiles = jax.nn.softplus(jax.random.normal(subkey, (3, 50)))

        key, subkey = jax.random.split(key)
        coords = jax.random.uniform(subkey, (n_spots, 2))

        data = {
            "spot_expression": spot_expr,
            "reference_profiles": ref_profiles,
            "coordinates": coords,
        }
        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["cell_proportions"]).all()

    def test_many_cell_types(self, rngs):
        """Test with many cell types."""
        config = SpatialDeconvolutionConfig(
            n_genes=50,
            n_cell_types=20,
            hidden_dim=32,
            num_layers=1,
        )
        op = SpatialDeconvolution(config, rngs=rngs)

        key = jax.random.key(0)
        n_spots = 30

        key, subkey = jax.random.split(key)
        spot_expr = jax.nn.softplus(jax.random.normal(subkey, (n_spots, 50)))

        key, subkey = jax.random.split(key)
        ref_profiles = jax.nn.softplus(jax.random.normal(subkey, (20, 50)))

        key, subkey = jax.random.split(key)
        coords = jax.random.uniform(subkey, (n_spots, 2))

        data = {
            "spot_expression": spot_expr,
            "reference_profiles": ref_profiles,
            "coordinates": coords,
        }
        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["cell_proportions"].shape == (30, 20)
