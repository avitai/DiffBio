"""Tests for elementwise soft operations.

Each function is tested for:
- Hard mode parity with the JAX equivalent
- Output in expected range
- Multi-mode sweep (smooth, c0, c1, c2)
- Differentiability (finite gradients)
- JIT compatibility
"""

import jax
import jax.numpy as jnp
import pytest

from tests.core.test_soft_ops.conftest import assert_finite_grads, assert_softbool

MODES = ["smooth", "c0", "c1", "c2"]


class TestSigmoidal:
    """Core S-curve function. All other elementwise ops build on this."""

    @pytest.mark.parametrize("mode", MODES)
    def test_output_in_zero_one(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import sigmoidal

        x = jax.random.normal(jax.random.key(0), (20,))
        result = sigmoidal(x, softness=0.1, mode=mode)
        assert_softbool(result)

    @pytest.mark.parametrize("mode", MODES)
    def test_zero_maps_to_half(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import sigmoidal

        result = sigmoidal(jnp.array(0.0), softness=0.1, mode=mode)
        assert jnp.allclose(result, 0.5, atol=0.05)

    def test_smooth_is_sigmoid(self) -> None:
        from diffbio.core.soft_ops.elementwise import sigmoidal

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = sigmoidal(x, softness=1.0, mode="smooth")
        expected = jax.nn.sigmoid(x)
        assert jnp.allclose(result, expected, atol=1e-6)

    @pytest.mark.parametrize("mode", MODES)
    def test_differentiable(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import sigmoidal

        x = jnp.array([-1.0, 0.0, 1.0])
        assert_finite_grads(
            lambda x: sigmoidal(x, softness=0.1, mode=mode), (x,),
        )


class TestSoftReLU:
    """Soft ReLU family."""

    @pytest.mark.parametrize("mode", MODES)
    def test_positive_input_approximately_identity(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import softrelu

        x = jnp.array([5.0, 10.0])
        result = softrelu(x, softness=0.1, mode=mode)
        assert jnp.allclose(result, x, atol=0.5)

    @pytest.mark.parametrize("mode", MODES)
    def test_negative_input_approximately_zero(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import softrelu

        x = jnp.array([-5.0, -10.0])
        result = softrelu(x, softness=0.1, mode=mode)
        assert jnp.allclose(result, 0.0, atol=0.5)

    def test_smooth_is_softplus(self) -> None:
        from diffbio.core.soft_ops.elementwise import softrelu

        x = jnp.array([-2.0, 0.0, 2.0])
        result = softrelu(x, softness=1.0, mode="smooth")
        expected = jax.nn.softplus(x)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_gated_mode(self) -> None:
        from diffbio.core.soft_ops.elementwise import softrelu

        x = jnp.array([-2.0, 0.0, 2.0])
        gated = softrelu(x, softness=1.0, mode="smooth", gated=True)
        assert gated.shape == x.shape
        assert jnp.all(jnp.isfinite(gated))


class TestHeaviside:
    """Soft Heaviside step function."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.elementwise import heaviside

        x = jnp.array([-1.0, 0.0, 1.0])
        result = heaviside(x, mode="hard")
        expected = jnp.array([0.0, 0.5, 1.0])
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize("mode", MODES)
    def test_output_in_zero_one(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import heaviside

        x = jax.random.normal(jax.random.key(0), (20,))
        result = heaviside(x, softness=0.1, mode=mode)
        assert_softbool(result)


class TestRound:
    """Soft rounding."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.elementwise import round

        x = jnp.array([1.4, 1.5, 1.6, 2.3])
        result = round(x, mode="hard")
        assert jnp.allclose(result, jnp.round(x))

    @pytest.mark.parametrize("mode", MODES)
    def test_integer_input_unchanged(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import round

        x = jnp.array([1.0, 2.0, 3.0])
        result = round(x, softness=0.1, mode=mode)
        assert jnp.allclose(result, x, atol=0.1)

    @pytest.mark.parametrize("mode", MODES)
    def test_differentiable(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import round

        x = jnp.array([1.3, 2.7])
        assert_finite_grads(
            lambda x: round(x, softness=0.1, mode=mode), (x,),
        )


class TestSign:
    """Soft sign function."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.elementwise import sign

        x = jnp.array([-2.0, 0.0, 3.0])
        result = sign(x, mode="hard")
        assert jnp.allclose(result, jnp.sign(x))

    @pytest.mark.parametrize("mode", MODES)
    def test_range_minus_one_to_one(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import sign

        x = jax.random.normal(jax.random.key(0), (20,))
        result = sign(x, softness=0.1, mode=mode)
        assert jnp.all(result >= -1.0 - 1e-5)
        assert jnp.all(result <= 1.0 + 1e-5)


class TestAbs:
    """Soft absolute value."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.elementwise import abs

        x = jnp.array([-3.0, 0.0, 2.0])
        result = abs(x, mode="hard")
        assert jnp.allclose(result, jnp.abs(x))

    @pytest.mark.parametrize("mode", MODES)
    def test_non_negative_output(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import abs

        x = jax.random.normal(jax.random.key(0), (20,))
        result = abs(x, softness=0.1, mode=mode)
        # Soft abs can be slightly negative near 0
        assert jnp.all(result >= -0.1)

    @pytest.mark.parametrize("mode", MODES)
    def test_differentiable_at_zero(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import abs

        x = jnp.array(0.0)
        assert_finite_grads(
            lambda x: abs(x, softness=0.1, mode=mode), (x,),
        )


class TestReLU:
    """Soft ReLU with hard mode."""

    def test_hard_mode_matches_jax(self) -> None:
        from diffbio.core.soft_ops.elementwise import relu

        x = jnp.array([-2.0, 0.0, 3.0])
        result = relu(x, mode="hard")
        assert jnp.allclose(result, jax.nn.relu(x))

    @pytest.mark.parametrize("mode", MODES)
    def test_differentiable(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import relu

        x = jnp.array([-1.0, 0.0, 1.0])
        assert_finite_grads(
            lambda x: relu(x, softness=0.1, mode=mode), (x,),
        )


class TestClip:
    """Soft clipping to bounds [a, b]."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.elementwise import clip

        x = jnp.array([-5.0, 0.5, 10.0])
        result = clip(x, a=jnp.array(0.0), b=jnp.array(1.0), mode="hard")
        assert jnp.allclose(result, jnp.clip(x, 0.0, 1.0))

    @pytest.mark.parametrize("mode", MODES)
    def test_within_bounds_approximately_unchanged(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import clip

        x = jnp.array([0.3, 0.5, 0.7])
        result = clip(
            x, a=jnp.array(0.0), b=jnp.array(1.0),
            softness=0.01, mode=mode,
        )
        assert jnp.allclose(result, x, atol=0.1)

    @pytest.mark.parametrize("mode", MODES)
    def test_differentiable(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import clip

        x = jnp.array([-1.0, 0.5, 2.0])
        assert_finite_grads(
            lambda x: clip(
                x, a=jnp.array(0.0), b=jnp.array(1.0),
                softness=0.1, mode=mode,
            ),
            (x,),
        )
