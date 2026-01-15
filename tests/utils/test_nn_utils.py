"""Tests for neural network utilities module."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.utils.nn_utils import (
    build_mlp_layers,
    ensure_rngs,
    get_rng_key,
    init_learnable_param,
    safe_divide,
    safe_log,
    sigmoid_blend,
    soft_threshold,
    temperature_scaled_softmax,
)


class TestInitLearnableParam:
    """Tests for init_learnable_param function."""

    def test_creates_nnx_param(self):
        """Should return an nnx.Param instance."""
        param = init_learnable_param(1.0)
        assert isinstance(param, nnx.Param)

    def test_correct_value(self):
        """Should contain the correct value."""
        param = init_learnable_param(42.5)
        assert float(param.value) == pytest.approx(42.5)

    def test_negative_value(self):
        """Should handle negative values."""
        param = init_learnable_param(-10.0)
        assert float(param.value) == pytest.approx(-10.0)

    def test_zero_value(self):
        """Should handle zero value."""
        param = init_learnable_param(0.0)
        assert float(param.value) == pytest.approx(0.0)


class TestEnsureRngs:
    """Tests for ensure_rngs function."""

    def test_returns_provided_rngs(self):
        """Should return the same rngs if provided."""
        rngs = nnx.Rngs(42)
        result = ensure_rngs(rngs)
        assert result is rngs

    def test_creates_rngs_if_none(self):
        """Should create new rngs if None provided."""
        result = ensure_rngs(None)
        assert isinstance(result, nnx.Rngs)

    def test_uses_provided_seed(self):
        """Should use the provided seed for new rngs."""
        result1 = ensure_rngs(None, seed=42)
        result2 = ensure_rngs(None, seed=42)
        # Both should produce the same key
        key1 = result1.params()
        key2 = result2.params()
        assert jnp.array_equal(key1, key2)


class TestGetRngKey:
    """Tests for get_rng_key function."""

    def test_returns_key_from_rngs(self):
        """Should return key from rngs when available."""
        rngs = nnx.Rngs(42)
        key = get_rng_key(rngs, "params")
        assert key.shape == ()  # JAX key has shape ()

    def test_fallback_when_none(self):
        """Should return fallback key when rngs is None."""
        key = get_rng_key(None, "params", fallback_seed=123)
        expected = jax.random.key(123)
        assert jnp.array_equal(key, expected)

    def test_fallback_when_stream_missing(self):
        """Should return fallback when stream name not in rngs."""
        rngs = nnx.Rngs(42)
        # Request a stream name that doesn't exist
        key = get_rng_key(rngs, "nonexistent_stream", fallback_seed=99)
        expected = jax.random.key(99)
        assert jnp.array_equal(key, expected)


class TestBuildMlpLayers:
    """Tests for build_mlp_layers function."""

    def test_creates_correct_number_of_layers(self, rngs):
        """Should create the specified number of layers."""
        layers, dropouts, out_dim = build_mlp_layers(
            in_features=10,
            hidden_dim=32,
            num_layers=3,
            rngs=rngs,
        )
        assert len(layers) == 3
        assert dropouts is None

    def test_output_dimension_correct(self, rngs):
        """Should return correct output dimension."""
        _, _, out_dim = build_mlp_layers(
            in_features=10,
            hidden_dim=64,
            num_layers=2,
            rngs=rngs,
        )
        assert out_dim == 64

    def test_creates_dropout_layers(self, rngs):
        """Should create dropout layers when requested."""
        layers, dropouts, _ = build_mlp_layers(
            in_features=10,
            hidden_dim=32,
            num_layers=2,
            rngs=rngs,
            with_dropout=True,
            dropout_rate=0.1,
        )
        assert dropouts is not None
        assert len(dropouts) == 2

    def test_no_dropout_layers_by_default(self, rngs):
        """Should not create dropout layers by default."""
        _, dropouts, _ = build_mlp_layers(
            in_features=10,
            hidden_dim=32,
            num_layers=2,
            rngs=rngs,
        )
        assert dropouts is None

    def test_layers_are_nnx_linear(self, rngs):
        """All layers should be nnx.Linear instances."""
        layers, _, _ = build_mlp_layers(
            in_features=10,
            hidden_dim=32,
            num_layers=2,
            rngs=rngs,
        )
        for layer in layers:
            assert isinstance(layer, nnx.Linear)

    def test_zero_layers(self, rngs):
        """Should handle zero layers."""
        layers, _, out_dim = build_mlp_layers(
            in_features=10,
            hidden_dim=32,
            num_layers=0,
            rngs=rngs,
        )
        assert len(layers) == 0
        assert out_dim == 10  # Output dim equals input dim


class TestSoftThreshold:
    """Tests for soft_threshold function."""

    def test_above_threshold_high_weight(self):
        """Values above threshold should have weight close to 1."""
        values = jnp.array([30.0, 40.0, 50.0])
        result = soft_threshold(values, threshold=20.0, temperature=1.0)
        assert jnp.all(result > 0.99)

    def test_below_threshold_low_weight(self):
        """Values below threshold should have weight close to 0."""
        values = jnp.array([0.0, 5.0, 10.0])
        result = soft_threshold(values, threshold=20.0, temperature=1.0)
        assert jnp.all(result < 0.01)

    def test_at_threshold_half_weight(self):
        """Values at threshold should have weight close to 0.5."""
        values = jnp.array([20.0])
        result = soft_threshold(values, threshold=20.0, temperature=1.0)
        assert jnp.abs(result[0] - 0.5) < 0.01

    def test_temperature_sharpness(self):
        """Lower temperature should make sharper transition."""
        values = jnp.array([19.0, 21.0])
        low_temp = soft_threshold(values, threshold=20.0, temperature=0.1)
        high_temp = soft_threshold(values, threshold=20.0, temperature=10.0)
        # Low temp should have more extreme values
        assert low_temp[0] < high_temp[0]
        assert low_temp[1] > high_temp[1]


class TestTemperatureScaledSoftmax:
    """Tests for temperature_scaled_softmax function."""

    def test_output_sums_to_one(self):
        """Output should sum to 1 along the specified axis."""
        logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = temperature_scaled_softmax(logits, temperature=1.0)
        sums = jnp.sum(result, axis=-1)
        assert jnp.allclose(sums, 1.0)

    def test_high_temperature_uniform(self):
        """High temperature should produce more uniform distribution."""
        logits = jnp.array([0.0, 1.0, 10.0])
        high_temp = temperature_scaled_softmax(logits, temperature=100.0)
        low_temp = temperature_scaled_softmax(logits, temperature=0.1)
        # High temp should have smaller max value
        assert high_temp.max() < low_temp.max()

    def test_temperature_one_equals_softmax(self):
        """Temperature=1 should equal standard softmax."""
        logits = jnp.array([1.0, 2.0, 3.0])
        result = temperature_scaled_softmax(logits, temperature=1.0)
        expected = jax.nn.softmax(logits)
        assert jnp.allclose(result, expected)


class TestSigmoidBlend:
    """Tests for sigmoid_blend function."""

    def test_high_weight_selects_value_a(self):
        """High blend weight should select value_a."""
        result = sigmoid_blend(
            value_a=jnp.array(10.0),
            value_b=jnp.array(0.0),
            blend_weight=jnp.array(10.0),  # High positive
        )
        assert result > 9.0

    def test_low_weight_selects_value_b(self):
        """Low blend weight should select value_b."""
        result = sigmoid_blend(
            value_a=jnp.array(10.0),
            value_b=jnp.array(0.0),
            blend_weight=jnp.array(-10.0),  # High negative
        )
        assert result < 1.0

    def test_zero_weight_blends_equally(self):
        """Zero blend weight should blend equally."""
        result = sigmoid_blend(
            value_a=jnp.array(10.0),
            value_b=jnp.array(0.0),
            blend_weight=jnp.array(0.0),
        )
        assert jnp.abs(result - 5.0) < 0.1


class TestSafeDivide:
    """Tests for safe_divide function."""

    def test_normal_division(self):
        """Should perform normal division when denominator is non-zero."""
        result = safe_divide(jnp.array(10.0), jnp.array(2.0))
        assert jnp.abs(result - 5.0) < 1e-6

    def test_zero_denominator(self):
        """Should not produce inf when denominator is zero."""
        result = safe_divide(jnp.array(10.0), jnp.array(0.0))
        assert jnp.isfinite(result)

    def test_custom_epsilon(self):
        """Should use custom epsilon."""
        result = safe_divide(jnp.array(1.0), jnp.array(0.0), epsilon=1.0)
        assert jnp.abs(result - 1.0) < 0.1


class TestSafeLog:
    """Tests for safe_log function."""

    def test_normal_log(self):
        """Should compute normal log for positive values."""
        result = safe_log(jnp.array(jnp.e))
        assert jnp.abs(result - 1.0) < 1e-6

    def test_zero_input(self):
        """Should not produce -inf for zero input."""
        result = safe_log(jnp.array(0.0))
        assert jnp.isfinite(result)

    def test_custom_epsilon(self):
        """Should use custom epsilon."""
        result = safe_log(jnp.array(0.0), epsilon=1.0)
        # log(0 + 1) = log(1) = 0
        assert jnp.abs(result) < 0.1


class TestGradientFlow:
    """Tests for gradient flow through utility functions."""

    def test_soft_threshold_gradient(self):
        """soft_threshold should have valid gradients."""

        def loss_fn(values):
            return soft_threshold(values, threshold=20.0).sum()

        values = jnp.array([15.0, 20.0, 25.0])
        grad = jax.grad(loss_fn)(values)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.all(grad > 0)  # Sigmoid derivative is always positive

    def test_temperature_scaled_softmax_gradient(self):
        """temperature_scaled_softmax should have valid gradients."""

        def loss_fn(logits):
            return temperature_scaled_softmax(logits, temperature=1.0).sum()

        logits = jnp.array([1.0, 2.0, 3.0])
        grad = jax.grad(loss_fn)(logits)
        assert jnp.all(jnp.isfinite(grad))

    def test_sigmoid_blend_gradient(self):
        """sigmoid_blend should have valid gradients."""

        def loss_fn(weight):
            return sigmoid_blend(
                value_a=jnp.array(10.0),
                value_b=jnp.array(0.0),
                blend_weight=weight,
            )

        weight = jnp.array(0.0)
        grad = jax.grad(loss_fn)(weight)
        assert jnp.isfinite(grad)
