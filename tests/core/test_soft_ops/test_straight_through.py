"""Tests for straight-through estimator decorators and _st variants.

Straight-through estimators use the hard (non-differentiable) function
for the forward pass but compute gradients through the soft version
in the backward pass.
"""

import jax
import jax.numpy as jnp
import pytest


class TestStDecorator:
    """Test the st() decorator."""

    def test_forward_is_hard(self) -> None:
        from diffbio.core.soft_ops.elementwise import relu
        from diffbio.core.soft_ops.straight_through import st

        relu_st = st(relu)
        x = jnp.array([-1.0, 0.0, 1.0])
        result = relu_st(x, softness=0.1)
        # Forward should be hard relu
        expected = jax.nn.relu(x)
        assert jnp.allclose(result, expected)

    def test_backward_is_soft(self) -> None:
        from diffbio.core.soft_ops.elementwise import relu
        from diffbio.core.soft_ops.straight_through import st

        relu_st = st(relu)
        x = jnp.array([-0.1])
        # Hard relu gradient at -0.1 would be 0
        # Soft relu gradient at -0.1 should be non-zero
        grad = jax.grad(lambda x: jnp.sum(relu_st(x, softness=0.5)))(x)
        assert float(grad[0]) != 0.0  # soft gradient, not zero

    def test_with_explicit_mode(self) -> None:
        from diffbio.core.soft_ops.elementwise import sign
        from diffbio.core.soft_ops.straight_through import st

        sign_st = st(sign)
        x = jnp.array([-2.0, 0.5, 3.0])
        result = sign_st(x, softness=0.1, mode="c1")
        # Forward: hard sign
        expected = jnp.sign(x)
        assert jnp.allclose(result, expected)


class TestGradReplace:
    """Test the grad_replace decorator."""

    def test_forward_backward_split(self) -> None:
        from diffbio.core.soft_ops.straight_through import grad_replace

        @grad_replace
        def my_fn(x, forward=True):
            if forward:
                return jnp.round(x)  # hard
            return x  # identity (soft)

        x = jnp.array([1.7])
        result = my_fn(x)
        assert jnp.allclose(result, 2.0)  # forward = round

        grad = jax.grad(lambda x: jnp.sum(my_fn(x)))(x)
        assert jnp.allclose(grad, 1.0)  # backward = identity


class TestStVariants:
    """Test that all 27 _st variants exist and are callable."""

    ST_NAMES = [
        "abs_st",
        "argmax_st",
        "argmedian_st",
        "argmin_st",
        "argpercentile_st",
        "argquantile_st",
        "argsort_st",
        "clip_st",
        "equal_st",
        "greater_equal_st",
        "greater_st",
        "heaviside_st",
        "isclose_st",
        "less_equal_st",
        "less_st",
        "max_st",
        "median_st",
        "min_st",
        "not_equal_st",
        "percentile_st",
        "quantile_st",
        "rank_st",
        "relu_st",
        "round_st",
        "sign_st",
        "sort_st",
        "top_k_st",
    ]

    @pytest.mark.parametrize("name", ST_NAMES)
    def test_st_variant_importable(self, name: str) -> None:
        import diffbio.core.soft_ops.straight_through as st_mod

        fn = getattr(st_mod, name)
        assert callable(fn)

    def test_relu_st_hard_forward_soft_backward(self) -> None:
        from diffbio.core.soft_ops.straight_through import relu_st

        x = jnp.array([-0.5, 0.0, 1.0])
        result = relu_st(x, softness=0.1)
        # Forward: hard relu
        expected = jax.nn.relu(x)
        assert jnp.allclose(result, expected)
        # Backward: finite (soft) gradient even at -0.5
        grad = jax.grad(lambda x: jnp.sum(relu_st(x, softness=0.5)))(x)
        assert jnp.all(jnp.isfinite(grad))

    def test_sort_st_hard_forward(self) -> None:
        from diffbio.core.soft_ops.straight_through import sort_st

        x = jnp.array([3.0, 1.0, 2.0])
        result = sort_st(x, axis=0, softness=0.1)
        expected = jnp.sort(x)
        assert jnp.allclose(result, expected)

    def test_greater_st_returns_hard_bool(self) -> None:
        from diffbio.core.soft_ops.straight_through import greater_st

        x = jnp.array([1.0, 3.0])
        y = jnp.array([2.0, 2.0])
        result = greater_st(x, y, softness=0.1)
        expected = jnp.array([0.0, 1.0])
        assert jnp.allclose(result, expected)

    def test_count_is_27(self) -> None:
        assert len(self.ST_NAMES) == 27
