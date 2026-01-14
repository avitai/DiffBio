"""Tests for diffbio.operators.singlecell.velocity module.

These tests define the expected behavior of the DifferentiableVelocity
operator for RNA velocity estimation via Neural ODEs.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.velocity import (
    DifferentiableVelocity,
    VelocityConfig,
)


class TestVelocityConfig:
    """Tests for VelocityConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VelocityConfig()
        assert config.n_genes == 2000
        assert config.hidden_dim == 64
        assert config.dt == 0.1
        assert config.n_steps == 10
        assert config.stochastic is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = VelocityConfig(
            n_genes=5000,
            hidden_dim=128,
            dt=0.05,
            n_steps=20,
        )
        assert config.n_genes == 5000
        assert config.hidden_dim == 128
        assert config.dt == 0.05
        assert config.n_steps == 20


class TestDifferentiableVelocity:
    """Tests for DifferentiableVelocity operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNGs for operator initialization."""
        return nnx.Rngs(42)

    @pytest.fixture
    def small_config(self):
        """Provide small config for faster tests."""
        return VelocityConfig(
            n_genes=50,
            hidden_dim=32,
            dt=0.1,
            n_steps=5,
        )

    @pytest.fixture
    def sample_data(self):
        """Provide sample spliced/unspliced count data."""
        key = jax.random.key(0)
        n_cells = 30
        n_genes = 50

        # Spliced counts (mature mRNA)
        key, subkey = jax.random.split(key)
        spliced = jax.nn.softplus(jax.random.normal(subkey, (n_cells, n_genes)))

        # Unspliced counts (nascent mRNA)
        key, subkey = jax.random.split(key)
        unspliced = jax.nn.softplus(jax.random.normal(subkey, (n_cells, n_genes)))

        return {
            "spliced": spliced,
            "unspliced": unspliced,
        }

    def test_initialization(self, rngs, small_config):
        """Test operator initialization."""
        op = DifferentiableVelocity(small_config, rngs=rngs)
        assert op is not None

    def test_output_contains_velocity(self, rngs, small_config, sample_data):
        """Test that output contains velocity estimates."""
        op = DifferentiableVelocity(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "velocity" in transformed
        assert transformed["velocity"].shape == (30, 50)

    def test_output_contains_kinetics(self, rngs, small_config, sample_data):
        """Test that output contains kinetics parameters."""
        op = DifferentiableVelocity(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        # Per-gene kinetics parameters
        assert "alpha" in transformed  # transcription rate
        assert "beta" in transformed   # splicing rate
        assert "gamma" in transformed  # degradation rate

    def test_output_contains_latent_time(self, rngs, small_config, sample_data):
        """Test that output contains latent time estimates."""
        op = DifferentiableVelocity(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "latent_time" in transformed
        # One time estimate per cell
        assert transformed["latent_time"].shape == (30,)

    def test_kinetics_positive(self, rngs, small_config, sample_data):
        """Test that kinetics parameters are positive."""
        op = DifferentiableVelocity(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert jnp.all(transformed["alpha"] > 0)
        assert jnp.all(transformed["beta"] > 0)
        assert jnp.all(transformed["gamma"] > 0)

    def test_output_finite(self, rngs, small_config, sample_data):
        """Test that outputs are finite."""
        op = DifferentiableVelocity(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert jnp.isfinite(transformed["velocity"]).all()
        assert jnp.isfinite(transformed["latent_time"]).all()
        assert jnp.isfinite(transformed["alpha"]).all()


class TestGradientFlow:
    """Tests for gradient flow through velocity estimation."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def small_config(self):
        return VelocityConfig(
            n_genes=30,
            hidden_dim=16,
            dt=0.1,
            n_steps=3,
        )

    def test_gradient_flows_through_velocity(self, rngs, small_config):
        """Test that gradients flow through velocity estimation."""
        op = DifferentiableVelocity(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_cells = 15
        n_genes = 30

        key, subkey = jax.random.split(key)
        spliced = jax.nn.softplus(jax.random.normal(subkey, (n_cells, n_genes)))

        key, subkey = jax.random.split(key)
        unspliced = jax.nn.softplus(jax.random.normal(subkey, (n_cells, n_genes)))

        def loss_fn(s):
            data = {"spliced": s, "unspliced": unspliced}
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["velocity"].sum()

        grad = jax.grad(loss_fn)(spliced)
        assert grad is not None
        assert grad.shape == spliced.shape
        assert jnp.isfinite(grad).all()

    def test_model_is_learnable(self, rngs, small_config):
        """Test that model parameters are learnable."""
        op = DifferentiableVelocity(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_cells = 15
        n_genes = 30

        key, subkey = jax.random.split(key)
        spliced = jax.nn.softplus(jax.random.normal(subkey, (n_cells, n_genes)))

        key, subkey = jax.random.split(key)
        unspliced = jax.nn.softplus(jax.random.normal(subkey, (n_cells, n_genes)))

        data = {"spliced": spliced, "unspliced": unspliced}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["velocity"].sum()

        loss, grads = loss_fn(op)

        assert hasattr(grads, "time_encoder")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def small_config(self):
        return VelocityConfig(
            n_genes=30,
            hidden_dim=16,
            dt=0.1,
            n_steps=3,
        )

    def test_apply_is_jit_compatible(self, rngs, small_config):
        """Test that apply method works with JIT."""
        op = DifferentiableVelocity(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_cells = 15
        n_genes = 30

        key, subkey = jax.random.split(key)
        spliced = jax.nn.softplus(jax.random.normal(subkey, (n_cells, n_genes)))

        key, subkey = jax.random.split(key)
        unspliced = jax.nn.softplus(jax.random.normal(subkey, (n_cells, n_genes)))

        data = {"spliced": spliced, "unspliced": unspliced}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["velocity"]).all()


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_few_cells(self, rngs):
        """Test with few cells."""
        config = VelocityConfig(
            n_genes=20,
            hidden_dim=8,
            dt=0.1,
            n_steps=3,
        )
        op = DifferentiableVelocity(config, rngs=rngs)

        key = jax.random.key(0)
        spliced = jax.nn.softplus(jax.random.normal(key, (5, 20)))
        unspliced = jax.nn.softplus(jax.random.normal(key, (5, 20)))

        data = {"spliced": spliced, "unspliced": unspliced}
        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["velocity"]).all()

    def test_different_dt(self, rngs):
        """Test with different time step sizes."""
        for dt in [0.01, 0.1, 0.5]:
            config = VelocityConfig(
                n_genes=20,
                hidden_dim=8,
                dt=dt,
                n_steps=5,
            )
            op = DifferentiableVelocity(config, rngs=rngs)

            key = jax.random.key(0)
            spliced = jax.nn.softplus(jax.random.normal(key, (10, 20)))
            unspliced = jax.nn.softplus(jax.random.normal(key, (10, 20)))

            data = {"spliced": spliced, "unspliced": unspliced}
            transformed, _, _ = op.apply(data, {}, None, None)
            assert jnp.isfinite(transformed["velocity"]).all()
