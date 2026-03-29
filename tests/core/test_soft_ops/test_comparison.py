"""Tests for soft comparison operators."""

import jax
import jax.numpy as jnp
import pytest

from tests.core.test_soft_ops.conftest import assert_finite_grads, assert_softbool

MODES = ["smooth", "c0", "c1", "c2"]


class TestGreater:
    """Soft x > y."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.comparison import greater

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 2.0, 1.0])
        result = greater(x, y, mode="hard")
        expected = jnp.array([0.0, 0.0, 1.0])
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize("mode", MODES)
    def test_output_is_softbool(self, mode: str) -> None:
        from diffbio.core.soft_ops.comparison import greater

        x = jax.random.normal(jax.random.key(0), (10,))
        y = jax.random.normal(jax.random.key(1), (10,))
        result = greater(x, y, softness=0.1, mode=mode)
        assert_softbool(result)

    @pytest.mark.parametrize("mode", MODES)
    def test_differentiable(self, mode: str) -> None:
        from diffbio.core.soft_ops.comparison import greater

        x = jnp.array([1.0, 3.0])
        y = jnp.array([2.0, 2.0])
        assert_finite_grads(
            lambda x: greater(x, y, softness=0.1, mode=mode),
            (x,),
        )

    def test_clear_greater_approaches_one(self) -> None:
        from diffbio.core.soft_ops.comparison import greater

        result = greater(jnp.array(10.0), jnp.array(0.0), softness=0.1)
        assert float(result) > 0.99

    def test_clear_less_approaches_zero(self) -> None:
        from diffbio.core.soft_ops.comparison import greater

        result = greater(jnp.array(0.0), jnp.array(10.0), softness=0.1)
        assert float(result) < 0.01


class TestGreaterEqual:
    """Soft x >= y."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.comparison import greater_equal

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 2.0, 1.0])
        result = greater_equal(x, y, mode="hard")
        expected = jnp.array([0.0, 1.0, 1.0])
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize("mode", MODES)
    def test_output_is_softbool(self, mode: str) -> None:
        from diffbio.core.soft_ops.comparison import greater_equal

        x = jax.random.normal(jax.random.key(0), (10,))
        y = jax.random.normal(jax.random.key(1), (10,))
        assert_softbool(greater_equal(x, y, softness=0.1, mode=mode))


class TestLess:
    """Soft x < y."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.comparison import less

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 2.0, 1.0])
        result = less(x, y, mode="hard")
        expected = jnp.array([1.0, 0.0, 0.0])
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize("mode", MODES)
    def test_complement_of_greater_equal(self, mode: str) -> None:
        from diffbio.core.soft_ops.comparison import greater_equal, less

        x = jax.random.normal(jax.random.key(0), (5,))
        y = jax.random.normal(jax.random.key(1), (5,))
        r_less = less(x, y, softness=0.1, mode=mode)
        r_ge = greater_equal(x, y, softness=0.1, mode=mode)
        assert jnp.allclose(r_less + r_ge, 1.0, atol=0.02)


class TestLessEqual:
    """Soft x <= y."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.comparison import less_equal

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 2.0, 1.0])
        result = less_equal(x, y, mode="hard")
        expected = jnp.array([1.0, 1.0, 0.0])
        assert jnp.allclose(result, expected)


class TestEqual:
    """Soft x == y."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.comparison import equal

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 5.0, 3.0])
        result = equal(x, y, mode="hard")
        expected = jnp.array([1.0, 0.0, 1.0])
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize("mode", MODES)
    def test_equal_values_high_probability(self, mode: str) -> None:
        from diffbio.core.soft_ops.comparison import equal

        x = jnp.array([5.0, 5.0])
        y = jnp.array([5.0, 5.0])
        result = equal(x, y, softness=0.1, mode=mode)
        assert jnp.all(result > 0.5)


class TestNotEqual:
    """Soft x != y."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.comparison import not_equal

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 5.0, 3.0])
        result = not_equal(x, y, mode="hard")
        expected = jnp.array([0.0, 1.0, 0.0])
        assert jnp.allclose(result, expected)


class TestIsclose:
    """Soft approximate equality."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.comparison import isclose

        x = jnp.array([1.0, 1.0001, 2.0])
        y = jnp.array([1.0, 1.0, 1.0])
        result = isclose(x, y, mode="hard", atol=1e-3)
        expected = jnp.isclose(x, y, atol=1e-3).astype(jnp.float32)
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize("mode", MODES)
    def test_close_values_high_probability(self, mode: str) -> None:
        from diffbio.core.soft_ops.comparison import isclose

        x = jnp.array([1.0, 1.0])
        y = jnp.array([1.0, 1.0])
        result = isclose(x, y, softness=0.1, mode=mode)
        assert jnp.all(result > 0.5)
