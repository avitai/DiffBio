"""Cross-cutting JIT and differentiability tests for all soft_ops.

Ensures every public function is:
1. Compatible with jax.jit (compiles and runs without error)
2. Compatible with jax.grad (produces finite gradients)
3. Compatible with jax.vmap (batch dimensions work correctly)
"""

import jax
import jax.numpy as jnp
import pytest

from tests.core.test_soft_ops.conftest import assert_finite_grads


# ── Autograd-safe ──────────────────────────────────────────────────────


class TestAutograduSafeJIT:
    """JIT compatibility for autograd-safe functions."""

    @pytest.mark.parametrize("fn_name", ["sqrt", "arcsin", "arccos", "log"])
    def test_unary_jit(self, fn_name: str) -> None:
        import diffbio.core.soft_ops.autograd_safe as ag

        fn = getattr(ag, fn_name)
        x = jnp.array([0.5])
        result = jax.jit(fn)(x)
        assert jnp.all(jnp.isfinite(result))

    def test_div_jit(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import div

        result = jax.jit(div)(jnp.array([6.0]), jnp.array([3.0]))
        assert jnp.allclose(result, 2.0)

    def test_norm_jit(self) -> None:
        from diffbio.core.soft_ops.autograd_safe import norm

        result = jax.jit(norm)(jnp.array([3.0, 4.0]))
        assert jnp.allclose(result, 5.0, atol=1e-5)

    @pytest.mark.parametrize("fn_name", ["sqrt", "arcsin", "arccos", "log"])
    def test_unary_grad(self, fn_name: str) -> None:
        import diffbio.core.soft_ops.autograd_safe as ag

        fn = getattr(ag, fn_name)
        x = jnp.array([0.5])
        assert_finite_grads(fn, (x,))


# ── Elementwise ────────────────────────────────────────────────────────


class TestElementwiseJIT:
    """JIT compatibility for elementwise functions."""

    @pytest.mark.parametrize(
        "fn_name", ["sigmoidal", "softrelu", "heaviside", "sign", "relu"],
    )
    @pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
    def test_jit(self, fn_name: str, mode: str) -> None:
        from diffbio.core.soft_ops import elementwise

        fn = getattr(elementwise, fn_name)
        x = jnp.array([-1.0, 0.0, 1.0])

        @jax.jit
        def f(x):
            return fn(x, softness=0.1, mode=mode)

        result = f(x)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
    def test_abs_jit(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import abs

        @jax.jit
        def f(x):
            return abs(x, softness=0.1, mode=mode)

        assert jnp.all(jnp.isfinite(f(jnp.array([-1.0, 0.0, 1.0]))))

    @pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
    def test_round_jit(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import round

        @jax.jit
        def f(x):
            return round(x, softness=0.1, mode=mode)

        assert jnp.all(jnp.isfinite(f(jnp.array([1.3, 2.7]))))

    @pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
    def test_clip_jit(self, mode: str) -> None:
        from diffbio.core.soft_ops.elementwise import clip

        @jax.jit
        def f(x):
            return clip(x, jnp.array(0.0), jnp.array(1.0), softness=0.1, mode=mode)

        assert jnp.all(jnp.isfinite(f(jnp.array([-1.0, 0.5, 2.0]))))

    @pytest.mark.parametrize(
        "fn_name", ["sigmoidal", "softrelu", "heaviside", "sign", "relu"],
    )
    def test_grad(self, fn_name: str) -> None:
        from diffbio.core.soft_ops import elementwise

        fn = getattr(elementwise, fn_name)
        x = jnp.array([-1.0, 0.0, 1.0])
        assert_finite_grads(lambda x: fn(x, softness=0.1), (x,))


# ── Comparison ─────────────────────────────────────────────────────────


class TestComparisonJIT:
    """JIT compatibility for comparison functions."""

    @pytest.mark.parametrize(
        "fn_name",
        ["greater", "greater_equal", "less", "less_equal", "equal", "not_equal"],
    )
    @pytest.mark.parametrize("mode", ["smooth", "c0"])
    def test_jit(self, fn_name: str, mode: str) -> None:
        from diffbio.core.soft_ops import comparison

        fn = getattr(comparison, fn_name)
        x = jnp.array([1.0, 3.0])
        y = jnp.array([2.0, 2.0])

        @jax.jit
        def f(x, y):
            return fn(x, y, softness=0.1, mode=mode)

        result = f(x, y)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.parametrize(
        "fn_name",
        ["greater", "greater_equal", "less", "less_equal"],
    )
    def test_grad(self, fn_name: str) -> None:
        from diffbio.core.soft_ops import comparison

        fn = getattr(comparison, fn_name)
        x = jnp.array([1.0, 3.0])
        y = jnp.array([2.0, 2.0])
        assert_finite_grads(
            lambda x: fn(x, y, softness=0.1), (x,),
        )


# ── Selection ──────────────────────────────────────────────────────────


class TestSelectionJIT:
    """JIT compatibility for selection functions."""

    def test_where_jit(self) -> None:
        from diffbio.core.soft_ops.selection import where

        @jax.jit
        def f(c, x, y):
            return where(c, x, y)

        c = jnp.array([0.8, 0.2])
        x = jnp.array([10.0, 20.0])
        y = jnp.array([1.0, 2.0])
        assert jnp.all(jnp.isfinite(f(c, x, y)))

    def test_take_along_axis_jit(self) -> None:
        from diffbio.core.soft_ops.selection import take_along_axis

        @jax.jit
        def f(x, idx):
            return take_along_axis(x, idx, axis=0)

        x = jnp.array([10.0, 20.0, 30.0])
        idx = jnp.array([[0.0, 1.0, 0.0]])
        assert jnp.all(jnp.isfinite(f(x, idx)))

    def test_where_grad(self) -> None:
        from diffbio.core.soft_ops.selection import where

        c = jnp.array([0.5])
        x = jnp.array([10.0])
        y = jnp.array([0.0])
        assert_finite_grads(lambda c: where(c, x, y), (c,))


# ── Sorting ────────────────────────────────────────────────────────────


class TestSortingJIT:
    """JIT compatibility for sorting functions."""

    @pytest.mark.parametrize("method", ["softsort", "neuralsort"])
    def test_argmax_jit(self, method: str) -> None:
        from diffbio.core.soft_ops.sorting import argmax

        @jax.jit
        def f(x):
            return argmax(x, axis=0, softness=0.1, method=method)

        result = f(jnp.array([1.0, 3.0, 2.0]))
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.parametrize("method", ["softsort", "neuralsort"])
    def test_sort_jit(self, method: str) -> None:
        from diffbio.core.soft_ops.sorting import sort

        @jax.jit
        def f(x):
            return sort(x, axis=0, softness=0.1, method=method)

        result = f(jnp.array([3.0, 1.0, 2.0]))
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.parametrize("method", ["softsort", "neuralsort"])
    def test_argmax_grad(self, method: str) -> None:
        from diffbio.core.soft_ops.sorting import argmax

        x = jnp.array([1.0, 3.0, 2.0])
        assert_finite_grads(
            lambda x: argmax(x, axis=0, softness=0.1, method=method),
            (x,),
        )

    @pytest.mark.parametrize("method", ["softsort", "neuralsort"])
    def test_sort_grad(self, method: str) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jnp.array([3.0, 1.0, 2.0])
        assert_finite_grads(
            lambda x: sort(x, axis=0, softness=0.1, method=method),
            (x,),
        )

    @pytest.mark.parametrize("method", ["softsort", "neuralsort"])
    def test_rank_jit(self, method: str) -> None:
        from diffbio.core.soft_ops.sorting import rank

        @jax.jit
        def f(x):
            return rank(x, axis=0, softness=0.1, method=method)

        result = f(jnp.array([3.0, 1.0, 2.0]))
        assert jnp.all(jnp.isfinite(result))


# ── Quantile ───────────────────────────────────────────────────────────


class TestQuantileJIT:
    """JIT compatibility for quantile functions."""

    def test_quantile_jit(self) -> None:
        from diffbio.core.soft_ops.quantile import quantile

        @jax.jit
        def f(x):
            return quantile(x, q=jnp.array(0.5), axis=0, softness=0.1)

        result = f(jnp.array([1.0, 3.0, 2.0, 4.0, 5.0]))
        assert jnp.all(jnp.isfinite(result))

    def test_median_jit(self) -> None:
        from diffbio.core.soft_ops.quantile import median

        @jax.jit
        def f(x):
            return median(x, axis=0, softness=0.1)

        result = f(jnp.array([5.0, 1.0, 3.0]))
        assert jnp.all(jnp.isfinite(result))

    def test_quantile_grad(self) -> None:
        from diffbio.core.soft_ops.quantile import quantile

        x = jnp.array([1.0, 3.0, 2.0, 4.0, 5.0])
        assert_finite_grads(
            lambda x: quantile(x, q=jnp.array(0.5), axis=0, softness=0.1),
            (x,),
        )


# ── Logical ────────────────────────────────────────────────────────────


class TestLogicalJIT:
    """JIT compatibility for logical functions."""

    def test_all_jit(self) -> None:
        from diffbio.core.soft_ops.logical import all

        @jax.jit
        def f(x):
            return all(x, axis=0)

        result = f(jnp.array([0.9, 0.8, 0.7]))
        assert jnp.isfinite(result)

    def test_logical_and_jit(self) -> None:
        from diffbio.core.soft_ops.logical import logical_and

        @jax.jit
        def f(x, y):
            return logical_and(x, y)

        result = f(jnp.array([0.9]), jnp.array([0.8]))
        assert jnp.all(jnp.isfinite(result))

    def test_all_grad(self) -> None:
        from diffbio.core.soft_ops.logical import all

        x = jnp.array([0.8, 0.9])
        assert_finite_grads(lambda x: all(x, axis=0), (x,))


# ── vmap ───────────────────────────────────────────────────────────────


class TestVmap:
    """vmap compatibility for key functions."""

    def test_sigmoidal_vmap(self) -> None:
        from diffbio.core.soft_ops.elementwise import sigmoidal

        x = jax.random.normal(jax.random.key(0), (5, 10))
        result = jax.vmap(lambda xi: sigmoidal(xi, softness=0.1))(x)
        assert result.shape == (5, 10)

    def test_greater_vmap(self) -> None:
        from diffbio.core.soft_ops.comparison import greater

        x = jax.random.normal(jax.random.key(0), (5, 3))
        y = jax.random.normal(jax.random.key(1), (5, 3))
        result = jax.vmap(
            lambda xi, yi: greater(xi, yi, softness=0.1),
        )(x, y)
        assert result.shape == (5, 3)

    def test_argmax_vmap(self) -> None:
        from diffbio.core.soft_ops.sorting import argmax

        x = jax.random.normal(jax.random.key(0), (4, 5))
        result = jax.vmap(
            lambda xi: argmax(xi, axis=0, softness=0.1),
        )(x)
        assert result.shape == (4, 5)

    def test_sort_vmap(self) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jax.random.normal(jax.random.key(0), (4, 5))
        result = jax.vmap(
            lambda xi: sort(xi, axis=0, softness=0.1),
        )(x)
        assert result.shape == (4, 5)
