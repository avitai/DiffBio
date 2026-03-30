"""Tests for soft_ops types and utility functions.

Tests written first per TDD. These define the expected behavior
of SoftBool, SoftIndex type aliases and the 8 utility helpers.
"""

import jax
import jax.numpy as jnp
import pytest


class TestSoftTypes:
    """Test SoftBool and SoftIndex type aliases exist and are usable."""

    def test_soft_bool_is_importable(self) -> None:
        from diffbio.core.soft_ops._types import SoftBool

        assert SoftBool is not None

    def test_soft_index_is_importable(self) -> None:
        from diffbio.core.soft_ops._types import SoftIndex

        assert SoftIndex is not None


class TestValidateSoftness:
    """Test softness validation helper."""

    def test_positive_softness_passes(self) -> None:
        from diffbio.core.soft_ops._utils import validate_softness

        validate_softness(0.1)
        validate_softness(1.0)
        validate_softness(100.0)

    def test_zero_softness_raises(self) -> None:
        from diffbio.core.soft_ops._utils import validate_softness

        with pytest.raises(ValueError, match="positive"):
            validate_softness(0.0)

    def test_negative_softness_raises(self) -> None:
        from diffbio.core.soft_ops._utils import validate_softness

        with pytest.raises(ValueError, match="positive"):
            validate_softness(-1.0)

    def test_traced_softness_skips_validation(self) -> None:
        """Validation is skipped inside jit-traced contexts."""
        from diffbio.core.soft_ops._utils import validate_softness

        @jax.jit
        def fn(s: float) -> float:
            validate_softness(s)
            return s

        # Should not raise even with negative value during tracing
        result = fn(jnp.array(-1.0))
        assert result == -1.0


class TestEnsureFloat:
    """Test integer-to-float casting helper."""

    def test_float_input_unchanged(self) -> None:
        from diffbio.core.soft_ops._utils import ensure_float

        x = jnp.array([1.0, 2.0, 3.0])
        result = ensure_float(x)
        assert jnp.issubdtype(result.dtype, jnp.floating)
        assert jnp.array_equal(result, x)

    def test_int_input_cast_to_float(self) -> None:
        from diffbio.core.soft_ops._utils import ensure_float

        x = jnp.array([1, 2, 3])
        result = ensure_float(x)
        assert jnp.issubdtype(result.dtype, jnp.floating)

    def test_python_scalar_converted(self) -> None:
        from diffbio.core.soft_ops._utils import ensure_float

        result = ensure_float(5)
        assert jnp.issubdtype(result.dtype, jnp.floating)


class TestStandardizeAndSquash:
    """Test standardization + sigmoid squashing."""

    def test_output_in_zero_one(self) -> None:
        from diffbio.core.soft_ops._utils import standardize_and_squash

        x = jax.random.normal(jax.random.key(0), (10,))
        result = standardize_and_squash(x, axis=-1)
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)

    def test_returns_mean_std_when_requested(self) -> None:
        from diffbio.core.soft_ops._utils import standardize_and_squash

        x = jax.random.normal(jax.random.key(0), (10,))
        result, mean, std = standardize_and_squash(
            x,
            axis=-1,
            return_mean_std=True,
        )
        assert result.shape == x.shape
        assert mean.shape == (1,)
        assert std.shape == (1,)

    def test_higher_temperature_flatter_output(self) -> None:
        from diffbio.core.soft_ops._utils import standardize_and_squash

        x = jax.random.normal(jax.random.key(0), (10,))
        low_temp = standardize_and_squash(x, axis=-1, temperature=0.1)
        high_temp = standardize_and_squash(x, axis=-1, temperature=10.0)
        # Higher temperature -> values closer to 0.5 (flatter)
        assert float(jnp.std(high_temp)) < float(jnp.std(low_temp))


class TestUnsquashAndDestandardize:
    """Test inverse of standardize_and_squash."""

    def test_roundtrip(self) -> None:
        from diffbio.core.soft_ops._utils import (
            standardize_and_squash,
            unsquash_and_destandardize,
        )

        x = jax.random.normal(jax.random.key(0), (10,))
        squashed, mean, std = standardize_and_squash(
            x,
            axis=-1,
            return_mean_std=True,
        )
        recovered = unsquash_and_destandardize(squashed, mean, std)
        assert jnp.allclose(recovered, x, atol=1e-4)


class TestQuantileInterpolationParams:
    """Test quantile interpolation parameter computation."""

    def test_linear_method(self) -> None:
        from diffbio.core.soft_ops._utils import quantile_interpolation_params

        k, a, take_next = quantile_interpolation_params(
            jnp.array(0.5),
            n=5,
            method="linear",
        )
        assert int(k) == 2
        assert float(a) == 0.0
        assert take_next is True

    def test_lower_method(self) -> None:
        from diffbio.core.soft_ops._utils import quantile_interpolation_params

        k, a, take_next = quantile_interpolation_params(
            jnp.array(0.3),
            n=5,
            method="lower",
        )
        assert int(k) == 1
        assert float(a) == 0.0
        assert take_next is False

    def test_higher_method(self) -> None:
        from diffbio.core.soft_ops._utils import quantile_interpolation_params

        k, a, take_next = quantile_interpolation_params(
            jnp.array(0.3),
            n=5,
            method="higher",
        )
        assert int(k) == 2
        assert float(a) == 0.0
        assert take_next is False

    def test_unknown_method_raises(self) -> None:
        from diffbio.core.soft_ops._utils import quantile_interpolation_params

        with pytest.raises(ValueError, match="Unknown"):
            quantile_interpolation_params(
                jnp.array(0.5),
                n=5,
                method="bogus",  # type: ignore[arg-type]
            )


class TestMapInChunks:
    """Test chunked map with gradient checkpointing."""

    def test_identity_map(self) -> None:
        from diffbio.core.soft_ops._utils import map_in_chunks

        x = jnp.arange(10, dtype=jnp.float32).reshape(10, 1)
        result = map_in_chunks(lambda c: c * 2, x, chunk_size=3)
        expected = x * 2
        assert jnp.allclose(result, expected)

    def test_chunk_size_larger_than_input(self) -> None:
        from diffbio.core.soft_ops._utils import map_in_chunks

        x = jnp.arange(5, dtype=jnp.float32).reshape(5, 1)
        result = map_in_chunks(lambda c: c + 1, x, chunk_size=100)
        assert jnp.allclose(result, x + 1)

    def test_handles_remainder(self) -> None:
        from diffbio.core.soft_ops._utils import map_in_chunks

        # 7 elements, chunk_size=3 -> 2 full chunks + 1 partial
        x = jnp.arange(7, dtype=jnp.float32).reshape(7, 1)
        result = map_in_chunks(lambda c: c**2, x, chunk_size=3)
        expected = x**2
        assert jnp.allclose(result, expected)

    def test_differentiable(self) -> None:
        from diffbio.core.soft_ops._utils import map_in_chunks

        x = jnp.arange(6, dtype=jnp.float32).reshape(6, 1)
        grad = jax.grad(lambda x: jnp.sum(map_in_chunks(lambda c: c**2, x, chunk_size=2)))(x)
        assert jnp.allclose(grad, 2 * x)


class TestReduceInChunks:
    """Test chunked reduction with gradient checkpointing."""

    def test_sum_reduction(self) -> None:
        from diffbio.core.soft_ops._utils import reduce_in_chunks

        x = jnp.arange(10, dtype=jnp.float32).reshape(10, 1)
        result = reduce_in_chunks(
            lambda c: jnp.sum(c, axis=0),
            x,
            chunk_size=3,
        )
        expected = jnp.sum(x, axis=0)
        assert jnp.allclose(result, expected)

    def test_chunk_size_larger_than_input(self) -> None:
        from diffbio.core.soft_ops._utils import reduce_in_chunks

        x = jnp.arange(5, dtype=jnp.float32).reshape(5, 1)
        result = reduce_in_chunks(
            lambda c: jnp.sum(c, axis=0),
            x,
            chunk_size=100,
        )
        assert jnp.allclose(result, jnp.sum(x, axis=0))

    def test_differentiable(self) -> None:
        from diffbio.core.soft_ops._utils import reduce_in_chunks

        x = jnp.ones((6, 1), dtype=jnp.float32)
        grad = jax.grad(
            lambda x: jnp.sum(reduce_in_chunks(lambda c: jnp.sum(c**2, axis=0), x, chunk_size=2))
        )(x)
        assert jnp.allclose(grad, 2 * x)


class TestCanonicalizeAxis:
    """Test axis normalization helper."""

    def test_positive_axis(self) -> None:
        from diffbio.core.soft_ops._utils import canonicalize_axis

        assert canonicalize_axis(0, 3) == 0
        assert canonicalize_axis(2, 3) == 2

    def test_negative_axis(self) -> None:
        from diffbio.core.soft_ops._utils import canonicalize_axis

        assert canonicalize_axis(-1, 3) == 2
        assert canonicalize_axis(-3, 3) == 0

    def test_out_of_bounds_raises(self) -> None:
        from diffbio.core.soft_ops._utils import canonicalize_axis

        with pytest.raises(ValueError, match="out of bounds"):
            canonicalize_axis(3, 3)

    def test_none_axis_raises(self) -> None:
        from diffbio.core.soft_ops._utils import canonicalize_axis

        with pytest.raises(ValueError, match="must be specified"):
            canonicalize_axis(None, 3)
