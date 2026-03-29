"""Tests for autograd-safe math operations.

These functions provide NaN-free alternatives to standard JAX math
operations by using the double-where trick to avoid undefined gradients
at domain boundaries (e.g., sqrt at 0, arcsin at +/-1, log at 0).
"""

import jax
import jax.numpy as jnp

from tests.core.test_soft_ops.conftest import assert_finite_grads


class TestSqrt:
    """Safe square root: returns 0 for x <= 0, no NaN gradients."""

    def test_positive_values_match_jnp(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import sqrt

        x = jnp.array([1.0, 4.0, 9.0, 16.0])
        assert jnp.allclose(sqrt(x), jnp.sqrt(x))

    def test_zero_returns_zero(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import sqrt

        assert float(sqrt(jnp.array(0.0))) == 0.0

    def test_negative_returns_zero(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import sqrt

        x = jnp.array([-1.0, -5.0])
        assert jnp.allclose(sqrt(x), jnp.zeros(2))

    def test_gradient_at_zero_is_finite(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import sqrt

        grad = jax.grad(lambda x: jnp.sum(sqrt(x)))(jnp.array(0.0))
        assert jnp.isfinite(grad)

    def test_gradient_at_positive_is_finite(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import sqrt

        assert_finite_grads(lambda x: sqrt(x), (jnp.array([1.0, 4.0]),))

    def test_jit_compatible(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import sqrt

        result = jax.jit(sqrt)(jnp.array([4.0, 9.0]))
        assert jnp.allclose(result, jnp.array([2.0, 3.0]))


class TestArcsin:
    """Safe arcsin: returns +/-pi/2 at +/-1, no NaN gradients at boundary."""

    def test_interior_values_match_jnp(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import arcsin

        x = jnp.array([-0.5, 0.0, 0.5])
        assert jnp.allclose(arcsin(x), jnp.arcsin(x), atol=1e-6)

    def test_boundary_plus_one(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import arcsin

        result = arcsin(jnp.array(1.0))
        assert jnp.allclose(result, jnp.pi / 2, atol=1e-6)

    def test_boundary_minus_one(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import arcsin

        result = arcsin(jnp.array(-1.0))
        assert jnp.allclose(result, -jnp.pi / 2, atol=1e-6)

    def test_gradient_at_boundary_is_finite(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import arcsin

        for val in [1.0, -1.0]:
            grad = jax.grad(lambda x: jnp.sum(arcsin(x)))(jnp.array(val))
            assert jnp.isfinite(grad), f"NaN gradient at x={val}"


class TestArccos:
    """Safe arccos: returns 0 at 1, pi at -1, no NaN gradients."""

    def test_interior_values_match_jnp(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import arccos

        x = jnp.array([-0.5, 0.0, 0.5])
        assert jnp.allclose(arccos(x), jnp.arccos(x), atol=1e-6)

    def test_boundary_plus_one(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import arccos

        assert jnp.allclose(arccos(jnp.array(1.0)), 0.0, atol=1e-6)

    def test_boundary_minus_one(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import arccos

        assert jnp.allclose(arccos(jnp.array(-1.0)), jnp.pi, atol=1e-6)

    def test_gradient_at_boundary_is_finite(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import arccos

        for val in [1.0, -1.0]:
            grad = jax.grad(lambda x: jnp.sum(arccos(x)))(jnp.array(val))
            assert jnp.isfinite(grad), f"NaN gradient at x={val}"


class TestDiv:
    """Safe division: returns 0 when denominator is 0."""

    def test_normal_division(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import div

        x = jnp.array([6.0, 10.0])
        y = jnp.array([2.0, 5.0])
        assert jnp.allclose(div(x, y), jnp.array([3.0, 2.0]))

    def test_zero_denominator_returns_zero(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import div

        x = jnp.array([1.0, 5.0])
        y = jnp.array([0.0, 0.0])
        assert jnp.allclose(div(x, y), jnp.zeros(2))

    def test_gradient_at_zero_denominator_is_finite(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import div

        x = jnp.array(1.0)
        y = jnp.array(0.0)
        grad_x, grad_y = jax.grad(
            lambda x, y: jnp.sum(div(x, y)), argnums=(0, 1),
        )(x, y)
        assert jnp.isfinite(grad_x)
        assert jnp.isfinite(grad_y)


class TestLog:
    """Safe log: returns 0 for x <= 0, no NaN gradients."""

    def test_positive_values_match_jnp(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import log

        x = jnp.array([1.0, jnp.e, 10.0])
        assert jnp.allclose(log(x), jnp.log(x), atol=1e-6)

    def test_zero_returns_zero(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import log

        assert float(log(jnp.array(0.0))) == 0.0

    def test_negative_returns_zero(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import log

        assert float(log(jnp.array(-5.0))) == 0.0

    def test_gradient_at_zero_is_finite(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import log

        grad = jax.grad(lambda x: jnp.sum(log(x)))(jnp.array(0.0))
        assert jnp.isfinite(grad)


class TestNorm:
    """Safe L2 norm: no NaN gradients at zero vector."""

    def test_nonzero_vector(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import norm

        x = jnp.array([3.0, 4.0])
        assert jnp.allclose(norm(x), 5.0, atol=1e-6)

    def test_zero_vector_returns_zero(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import norm

        x = jnp.zeros(3)
        assert float(norm(x)) == 0.0

    def test_gradient_at_zero_is_finite(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import norm

        x = jnp.zeros(3)
        grad = jax.grad(lambda x: norm(x))(x)
        assert jnp.all(jnp.isfinite(grad))

    def test_axis_argument(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import norm

        x = jnp.array([[3.0, 4.0], [0.0, 0.0]])
        result = norm(x, axis=1)
        assert result.shape == (2,)
        assert jnp.allclose(result[0], 5.0, atol=1e-6)
        assert float(result[1]) == 0.0

    def test_keepdims(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import norm

        x = jnp.array([[3.0, 4.0]])
        result = norm(x, axis=1, keepdims=True)
        assert result.shape == (1, 1)

    def test_vmap_compatible(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import norm

        x = jax.random.normal(jax.random.key(0), (5, 3))
        result = jax.vmap(norm)(x)
        assert result.shape == (5,)
        assert jnp.all(jnp.isfinite(result))
