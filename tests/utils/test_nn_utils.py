"""Tests for neural network utilities module."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.utils.nn_utils import (
    ensure_rngs,
    extract_windows_1d,
    get_rng_key,
    init_learnable_param,
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
        assert float(param[...]) == pytest.approx(42.5)

    def test_negative_value(self):
        """Should handle negative values."""
        param = init_learnable_param(-10.0)
        assert float(param[...]) == pytest.approx(-10.0)

    def test_zero_value(self):
        """Should handle zero value."""
        param = init_learnable_param(0.0)
        assert float(param[...]) == pytest.approx(0.0)


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


class TestExtractWindows1d:
    """Tests for extract_windows_1d function."""

    def test_basic_window_extraction(self):
        """Should extract windows of correct shape."""
        signal = jnp.ones((100, 4))
        windows = extract_windows_1d(signal, window_size=11)
        assert windows.shape == (100, 11, 4)

    def test_window_values_from_signal(self):
        """Windows should contain values from the signal."""
        # Create a signal with distinct values at each position
        signal = jnp.arange(10)[:, None] * jnp.ones((1, 4))  # (10, 4)
        windows = extract_windows_1d(signal, window_size=3)

        # Middle window (position 5) should have values from positions 4, 5, 6
        assert windows.shape == (10, 3, 4)
        # Center value of window at position 5 should be 5
        assert float(windows[5, 1, 0]) == 5.0

    def test_edge_padding(self):
        """Should pad edges correctly."""
        signal = jnp.arange(5)[:, None] * jnp.ones((1, 2))  # (5, 2)
        windows = extract_windows_1d(signal, window_size=3, pad_mode="edge")

        # First window should have padded first element
        assert windows.shape == (5, 3, 2)
        # First window's first element should be same as first signal value (edge padding)
        assert float(windows[0, 0, 0]) == 0.0

    def test_different_window_sizes(self):
        """Should work with different window sizes."""
        signal = jnp.ones((50, 6))

        for window_size in [5, 11, 21]:
            windows = extract_windows_1d(signal, window_size=window_size)
            assert windows.shape == (50, window_size, 6)

    def test_gradient_flow(self):
        """Gradients should flow through window extraction."""

        def loss_fn(signal):
            windows = extract_windows_1d(signal, window_size=5)
            return jnp.sum(windows)

        signal = jnp.ones((20, 4))
        grad = jax.grad(loss_fn)(signal)
        assert grad.shape == signal.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_compatible(self):
        """Should be JIT compatible."""
        jit_extract = jax.jit(lambda s: extract_windows_1d(s, window_size=7))
        signal = jnp.ones((30, 4))
        windows = jit_extract(signal)
        assert windows.shape == (30, 7, 4)


class TestEdgeCases:
    """Edge case tests for neural network utilities."""

    def test_extract_windows_single_position(self):
        """Test window extraction with single position signal."""
        signal = jnp.ones((1, 4))  # Single position
        windows = extract_windows_1d(signal, window_size=3, pad_mode="edge")
        assert windows.shape == (1, 3, 4)
        # All values should be 1 (original value padded)
        assert jnp.allclose(windows, 1.0)

    def test_extract_windows_large_window(self):
        """Test window extraction where window > signal length."""
        signal = jnp.arange(3)[:, None] * jnp.ones((1, 2))  # (3, 2)
        windows = extract_windows_1d(signal, window_size=7, pad_mode="edge")
        assert windows.shape == (3, 7, 2)
        # Should be padded with edge values
        assert jnp.all(jnp.isfinite(windows))

    def test_init_learnable_param_large_value(self):
        """Test init learnable param with large value."""
        param = init_learnable_param(1e10)
        assert float(param[...]) == pytest.approx(1e10)

    def test_ensure_rngs_different_seeds(self):
        """Test ensure rngs produces different values for different seeds."""
        rngs1 = ensure_rngs(None, seed=1)
        rngs2 = ensure_rngs(None, seed=2)
        key1 = rngs1.params()
        key2 = rngs2.params()
        # Keys should be different
        assert not jnp.array_equal(key1, key2)
