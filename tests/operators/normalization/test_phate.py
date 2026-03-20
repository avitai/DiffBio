"""Tests for differentiable PHATE dimensionality reduction operator.

Following TDD principles, these tests define the expected behavior
of the DifferentiablePHATE operator for embedding high-dimensional data
via potential of heat-diffusion for affinity-based trajectory embedding.
"""

import jax
import jax.numpy as jnp
import pytest

from diffbio.operators.normalization import PHATEConfig


class TestPHATEConfig:
    """Tests for PHATEConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from diffbio.operators.normalization import PHATEConfig

        config = PHATEConfig(stream_name=None)

        assert config.n_components == 2
        assert config.n_neighbors == 5
        assert config.decay == 40.0
        assert config.diffusion_t == 10
        assert config.gamma == 1.0
        assert config.input_features == 64
        assert config.hidden_dim == 32
        assert config.stochastic is False

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        from diffbio.operators.normalization import PHATEConfig

        config = PHATEConfig(
            n_components=3,
            n_neighbors=10,
            decay=20.0,
            diffusion_t=5,
            gamma=0.0,
            input_features=100,
            hidden_dim=64,
            stream_name=None,
        )

        assert config.n_components == 3
        assert config.n_neighbors == 10
        assert config.decay == 20.0
        assert config.diffusion_t == 5
        assert config.gamma == 0.0
        assert config.input_features == 100
        assert config.hidden_dim == 64


class TestDifferentiablePHATE:
    """Tests for DifferentiablePHATE operator."""

    @pytest.fixture
    def config(self) -> "PHATEConfig":
        """Provide default config."""
        from diffbio.operators.normalization import PHATEConfig

        return PHATEConfig(
            n_components=2,
            n_neighbors=5,
            decay=40.0,
            diffusion_t=10,
            gamma=1.0,
            input_features=10,
            hidden_dim=32,
            stream_name=None,
        )

    @pytest.fixture
    def phate(self, config, rngs):
        """Create PHATE instance."""
        from diffbio.operators.normalization import DifferentiablePHATE

        return DifferentiablePHATE(config, rngs=rngs)

    def test_output_keys(self, phate) -> None:
        """Test that apply returns the expected output keys."""
        features = jax.random.normal(jax.random.key(0), (30, 10))
        data = {"features": features}
        result, state, metadata = phate.apply(data, {}, None)

        assert "embedding" in result
        assert "potential_distances" in result
        assert "diffusion_operator" in result
        assert "features" in result

    def test_output_shapes(self, phate, config) -> None:
        """Test output shapes match expectations."""
        n_samples = 30
        features = jax.random.normal(jax.random.key(0), (n_samples, 10))
        data = {"features": features}
        result, _, _ = phate.apply(data, {}, None)

        assert result["embedding"].shape == (n_samples, config.n_components)
        assert result["potential_distances"].shape == (n_samples, n_samples)
        assert result["diffusion_operator"].shape == (n_samples, n_samples)

    def test_embedding_finite(self, phate) -> None:
        """Test that embedding values are all finite (no NaN or Inf)."""
        features = jax.random.normal(jax.random.key(0), (30, 10))
        data = {"features": features}
        result, _, _ = phate.apply(data, {}, None)

        assert jnp.all(jnp.isfinite(result["embedding"]))

    def test_potential_distances_symmetric(self, phate) -> None:
        """Test that potential distances are approximately symmetric."""
        features = jax.random.normal(jax.random.key(0), (30, 10))
        data = {"features": features}
        result, _, _ = phate.apply(data, {}, None)

        pot_dist = result["potential_distances"]
        assert jnp.allclose(pot_dist, pot_dist.T, atol=1e-5)

    def test_preserves_original_data(self, phate) -> None:
        """Test that original data keys are preserved in output."""
        features = jax.random.normal(jax.random.key(0), (30, 10))
        extra = jnp.array([1.0, 2.0, 3.0])
        data = {"features": features, "extra": extra}
        result, _, _ = phate.apply(data, {}, None)

        assert "extra" in result
        assert jnp.allclose(result["extra"], extra)


class TestDiffusionOperator:
    """Tests for properties of the diffusion operator."""

    @pytest.fixture
    def config(self) -> "PHATEConfig":
        """Provide config for diffusion tests."""
        from diffbio.operators.normalization import PHATEConfig

        return PHATEConfig(
            n_components=2,
            n_neighbors=5,
            decay=40.0,
            diffusion_t=5,
            input_features=10,
            hidden_dim=32,
            stream_name=None,
        )

    @pytest.fixture
    def phate(self, config, rngs):
        """Create PHATE instance."""
        from diffbio.operators.normalization import DifferentiablePHATE

        return DifferentiablePHATE(config, rngs=rngs)

    def test_row_stochastic(self, phate) -> None:
        """Test that the diffusion operator is row-stochastic (rows sum to 1)."""
        features = jax.random.normal(jax.random.key(0), (30, 10))
        data = {"features": features}
        result, _, _ = phate.apply(data, {}, None)

        diff_op = result["diffusion_operator"]
        row_sums = jnp.sum(diff_op, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-4)

    def test_converges_with_higher_t(self, rngs) -> None:
        """Test that diffusion operator converges (rows become similar) at high t."""
        from diffbio.operators.normalization import DifferentiablePHATE, PHATEConfig

        features = jax.random.normal(jax.random.key(0), (30, 10))

        config_low = PHATEConfig(
            n_components=2,
            n_neighbors=5,
            diffusion_t=1,
            input_features=10,
            stream_name=None,
        )
        config_high = PHATEConfig(
            n_components=2,
            n_neighbors=5,
            diffusion_t=50,
            input_features=10,
            stream_name=None,
        )

        phate_low = DifferentiablePHATE(config_low, rngs=rngs)
        phate_high = DifferentiablePHATE(config_high, rngs=rngs)

        data = {"features": features}
        result_low, _, _ = phate_low.apply(data, {}, None)
        result_high, _, _ = phate_high.apply(data, {}, None)

        # At high t, rows of diffusion operator should be more similar
        # (converging toward stationary distribution)
        diff_op_low = result_low["diffusion_operator"]
        diff_op_high = result_high["diffusion_operator"]

        # Measure row variance: lower variance = more convergence
        row_var_low = jnp.var(diff_op_low, axis=0).mean()
        row_var_high = jnp.var(diff_op_high, axis=0).mean()

        assert row_var_high < row_var_low


class TestGradientFlow:
    """Tests for gradient flow through the PHATE operator."""

    @pytest.fixture
    def config(self) -> "PHATEConfig":
        """Provide config for gradient tests."""
        from diffbio.operators.normalization import PHATEConfig

        return PHATEConfig(
            n_components=2,
            n_neighbors=5,
            diffusion_t=3,
            input_features=10,
            hidden_dim=32,
            stream_name=None,
        )

    def test_grads_through_embedding(self, config, rngs) -> None:
        """Test that gradients flow through the embedding output."""
        from diffbio.operators.normalization import DifferentiablePHATE

        phate = DifferentiablePHATE(config, rngs=rngs)

        def loss_fn(features: jax.Array) -> jax.Array:
            data = {"features": features}
            result, _, _ = phate.apply(data, {}, None)
            return result["embedding"].sum()

        features = jax.random.normal(jax.random.key(0), (30, 10))
        grad = jax.grad(loss_fn)(features)

        assert grad.shape == features.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_grads_through_potential_distance(self, config, rngs) -> None:
        """Test that gradients flow through the potential distance output."""
        from diffbio.operators.normalization import DifferentiablePHATE

        phate = DifferentiablePHATE(config, rngs=rngs)

        def loss_fn(features: jax.Array) -> jax.Array:
            data = {"features": features}
            result, _, _ = phate.apply(data, {}, None)
            return result["potential_distances"].sum()

        features = jax.random.normal(jax.random.key(0), (30, 10))
        grad = jax.grad(loss_fn)(features)

        assert grad.shape == features.shape
        assert jnp.all(jnp.isfinite(grad))


class TestJITCompatibility:
    """Tests for JIT compilation compatibility."""

    @pytest.fixture
    def config(self) -> "PHATEConfig":
        """Provide config for JIT tests."""
        from diffbio.operators.normalization import PHATEConfig

        return PHATEConfig(
            n_components=2,
            n_neighbors=5,
            diffusion_t=3,
            input_features=10,
            hidden_dim=32,
            stream_name=None,
        )

    def test_jit_apply(self, config, rngs) -> None:
        """Test JIT compilation of the apply method."""
        from diffbio.operators.normalization import DifferentiablePHATE

        phate = DifferentiablePHATE(config, rngs=rngs)

        @jax.jit
        def jit_apply(features: jax.Array) -> jax.Array:
            data = {"features": features}
            result, _, _ = phate.apply(data, {}, None)
            return result["embedding"]

        features = jax.random.normal(jax.random.key(0), (30, 10))
        result = jit_apply(features)

        assert result.shape == (30, 2)
        assert jnp.all(jnp.isfinite(result))

    def test_jit_gradient(self, config, rngs) -> None:
        """Test JIT compilation of gradient computation."""
        from diffbio.operators.normalization import DifferentiablePHATE

        phate = DifferentiablePHATE(config, rngs=rngs)

        @jax.jit
        def loss_and_grad(features: jax.Array) -> tuple[jax.Array, jax.Array]:
            def loss_fn(f: jax.Array) -> jax.Array:
                data = {"features": f}
                result, _, _ = phate.apply(data, {}, None)
                return result["embedding"].sum()

            return jax.value_and_grad(loss_fn)(features)

        features = jax.random.normal(jax.random.key(0), (30, 10))
        loss, grad = loss_and_grad(features)

        assert jnp.isfinite(loss)
        assert jnp.all(jnp.isfinite(grad))


class TestEdgeCases:
    """Tests for edge cases and special configurations."""

    def test_small_graph(self, rngs) -> None:
        """Test with a very small number of samples (5)."""
        from diffbio.operators.normalization import DifferentiablePHATE, PHATEConfig

        config = PHATEConfig(
            n_components=2,
            n_neighbors=3,
            diffusion_t=3,
            input_features=10,
            hidden_dim=32,
            stream_name=None,
        )
        phate = DifferentiablePHATE(config, rngs=rngs)

        features = jax.random.normal(jax.random.key(0), (5, 10))
        data = {"features": features}
        result, _, _ = phate.apply(data, {}, None)

        assert result["embedding"].shape == (5, 2)
        assert jnp.all(jnp.isfinite(result["embedding"]))

    def test_gamma_zero_sqrt_potential(self, rngs) -> None:
        """Test gamma=0 which uses sqrt potential instead of log potential."""
        from diffbio.operators.normalization import DifferentiablePHATE, PHATEConfig

        config = PHATEConfig(
            n_components=2,
            n_neighbors=5,
            diffusion_t=5,
            gamma=0.0,
            input_features=10,
            hidden_dim=32,
            stream_name=None,
        )
        phate = DifferentiablePHATE(config, rngs=rngs)

        features = jax.random.normal(jax.random.key(0), (30, 10))
        data = {"features": features}
        result, _, _ = phate.apply(data, {}, None)

        assert result["embedding"].shape == (30, 2)
        assert jnp.all(jnp.isfinite(result["embedding"]))
        assert jnp.all(jnp.isfinite(result["potential_distances"]))
