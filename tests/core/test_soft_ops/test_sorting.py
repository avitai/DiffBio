"""Tests for soft sorting, argmax/argmin, and rank operators."""

import jax
import jax.numpy as jnp
import pytest

from diffbio.core.soft_ops.sorting import ArgMethod, RankMethod, SortMethod
from tests.core.test_soft_ops.conftest import assert_finite_grads, assert_simplex

CORE_ARG_METHODS: list[ArgMethod] = ["softsort", "neuralsort", "sorting_network"]
CORE_SORT_METHODS: list[SortMethod] = ["softsort", "neuralsort", "sorting_network"]
CORE_RANK_METHODS: list[RankMethod] = ["softsort", "neuralsort"]


class TestArgmax:
    """Soft argmax returning SoftIndex (probability distribution)."""

    def test_hard_mode_returns_one_hot(self) -> None:
        from diffbio.core.soft_ops.sorting import argmax

        x = jnp.array([1.0, 3.0, 2.0])
        result = argmax(x, axis=0, mode="hard")
        expected = jnp.array([0.0, 1.0, 0.0])
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize("method", CORE_ARG_METHODS)
    def test_output_is_simplex(self, method: ArgMethod) -> None:
        from diffbio.core.soft_ops.sorting import argmax

        x = jax.random.normal(jax.random.key(0), (5,))
        result = argmax(x, axis=0, softness=0.1, mode="smooth", method=method)
        assert_simplex(result, axis=-1, atol=1e-3)

    @pytest.mark.parametrize("method", CORE_ARG_METHODS)
    def test_low_softness_concentrates_on_max(self, method: ArgMethod) -> None:
        from diffbio.core.soft_ops.sorting import argmax

        x = jnp.array([1.0, 5.0, 2.0])
        result = argmax(x, axis=0, softness=0.01, mode="smooth", method=method)
        assert float(result[1]) > 0.9

    def test_keepdims(self) -> None:
        from diffbio.core.soft_ops.sorting import argmax

        x = jnp.array([[1.0, 3.0, 2.0], [5.0, 1.0, 4.0]])
        result = argmax(x, axis=1, keepdims=True, mode="smooth")
        # Shape should be (2, 1, 3) -- keepdims adds singleton
        assert result.shape == (2, 1, 3)

    @pytest.mark.parametrize("method", CORE_ARG_METHODS)
    def test_differentiable(self, method: ArgMethod) -> None:
        from diffbio.core.soft_ops.sorting import argmax

        x = jnp.array([1.0, 3.0, 2.0])
        assert_finite_grads(
            lambda x: argmax(x, axis=0, softness=0.1, method=method),
            (x,),
        )


class TestMax:
    """Soft max returning scalar value."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.sorting import max

        x = jnp.array([1.0, 5.0, 3.0])
        assert jnp.allclose(max(x, axis=0, mode="hard"), 5.0)

    @pytest.mark.parametrize("method", CORE_SORT_METHODS)
    def test_soft_max_approaches_hard(self, method: SortMethod) -> None:
        from diffbio.core.soft_ops.sorting import max

        x = jnp.array([1.0, 5.0, 3.0])
        result = max(x, axis=0, softness=0.01, mode="smooth", method=method)
        assert jnp.allclose(result, 5.0, atol=0.5)

    @pytest.mark.parametrize("method", CORE_SORT_METHODS)
    def test_differentiable(self, method: SortMethod) -> None:
        from diffbio.core.soft_ops.sorting import max

        x = jnp.array([1.0, 5.0, 3.0])
        assert_finite_grads(
            lambda x: max(x, axis=0, softness=0.1, method=method),
            (x,),
        )


class TestArgmin:
    """Soft argmin: argmax(-x)."""

    def test_hard_mode(self) -> None:
        from diffbio.core.soft_ops.sorting import argmin

        x = jnp.array([3.0, 1.0, 2.0])
        result = argmin(x, axis=0, mode="hard")
        expected = jnp.array([0.0, 1.0, 0.0])
        assert jnp.allclose(result, expected)


class TestMin:
    """Soft min: -max(-x)."""

    def test_hard_mode(self) -> None:
        from diffbio.core.soft_ops.sorting import min

        x = jnp.array([3.0, 1.0, 2.0])
        assert jnp.allclose(min(x, axis=0, mode="hard"), 1.0)


class TestArgsort:
    """Soft argsort returning permutation matrix (SoftIndex)."""

    def test_hard_mode_returns_one_hot_perm(self) -> None:
        from diffbio.core.soft_ops.sorting import argsort

        x = jnp.array([3.0, 1.0, 2.0])
        result = argsort(x, axis=0, mode="hard")
        # result[sorted_pos, :] should be one-hot pointing to original element
        assert result.shape == (3, 3)
        # Row sums should be 1
        assert jnp.allclose(jnp.sum(result, axis=-1), 1.0)

    @pytest.mark.parametrize("method", CORE_ARG_METHODS)
    def test_rows_are_simplex(self, method: ArgMethod) -> None:
        from diffbio.core.soft_ops.sorting import argsort

        x = jnp.array([3.0, 1.0, 2.0])
        result = argsort(
            x,
            axis=0,
            softness=0.1,
            mode="smooth",
            method=method,
        )
        for i in range(3):
            assert_simplex(result[i], axis=-1, atol=0.05)

    @pytest.mark.parametrize("method", CORE_ARG_METHODS)
    def test_differentiable(self, method: ArgMethod) -> None:
        from diffbio.core.soft_ops.sorting import argsort

        x = jnp.array([3.0, 1.0, 2.0])
        assert_finite_grads(
            lambda x: argsort(x, axis=0, softness=0.1, method=method),
            (x,),
        )


class TestSort:
    """Soft sort returning sorted values."""

    def test_hard_mode_matches_jnp(self) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jnp.array([3.0, 1.0, 2.0])
        result = sort(x, axis=0, mode="hard")
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))

    def test_hard_descending(self) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jnp.array([3.0, 1.0, 2.0])
        result = sort(x, axis=0, descending=True, mode="hard")
        assert jnp.allclose(result, jnp.array([3.0, 2.0, 1.0]))

    @pytest.mark.parametrize("method", CORE_SORT_METHODS)
    def test_soft_sort_approaches_hard(self, method: SortMethod) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jnp.array([5.0, 1.0, 3.0])
        result = sort(
            x,
            axis=0,
            softness=0.01,
            mode="smooth",
            method=method,
        )
        expected = jnp.array([1.0, 3.0, 5.0])
        assert jnp.allclose(result, expected, atol=1.0)

    @pytest.mark.parametrize("method", CORE_SORT_METHODS)
    def test_differentiable(self, method: SortMethod) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jnp.array([3.0, 1.0, 2.0])
        assert_finite_grads(
            lambda x: sort(x, axis=0, softness=0.1, method=method),
            (x,),
        )

    def test_batch_sort(self) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jax.random.normal(jax.random.key(0), (3, 5))
        result = sort(x, axis=1, softness=0.1, mode="smooth")
        assert result.shape == (3, 5)


class TestRank:
    """Soft rank: fractional ranking via soft argsort."""

    @pytest.mark.parametrize("method", CORE_RANK_METHODS)
    def test_basic_ranking(self, method: RankMethod) -> None:
        from diffbio.core.soft_ops.sorting import rank

        x = jnp.array([30.0, 10.0, 20.0])
        result = rank(
            x,
            axis=0,
            softness=0.01,
            mode="smooth",
            method=method,
        )
        # Expected ranks: 3, 1, 2 (ascending)
        assert jnp.allclose(result, jnp.array([3.0, 1.0, 2.0]), atol=0.5)

    @pytest.mark.parametrize("method", CORE_RANK_METHODS)
    def test_differentiable(self, method: RankMethod) -> None:
        from diffbio.core.soft_ops.sorting import rank

        x = jnp.array([3.0, 1.0, 2.0])
        assert_finite_grads(
            lambda x: rank(x, axis=0, softness=0.1, method=method),
            (x,),
        )


class TestTopK:
    """Soft top-k selection."""

    def test_hard_mode(self) -> None:
        from diffbio.core.soft_ops.sorting import top_k

        x = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        values, indices = top_k(x, k=2, axis=0, mode="hard")
        assert values.shape == (2,)
        # Top 2 should be 5 and 4
        assert jnp.allclose(jnp.sort(values), jnp.array([4.0, 5.0]))

    def test_soft_mode_returns_k_values(self) -> None:
        from diffbio.core.soft_ops.sorting import top_k

        x = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        values, indices = top_k(x, k=2, axis=0, softness=0.1, mode="smooth")
        assert values.shape == (2,)
        assert indices.shape == (2, 5)

    def test_differentiable(self) -> None:
        from diffbio.core.soft_ops.sorting import top_k

        x = jnp.array([1.0, 5.0, 3.0, 2.0])

        def fn(x):
            values, _ = top_k(x, k=2, axis=0, softness=0.1)
            return jnp.sum(values)

        assert_finite_grads(fn, (x,))

    def test_soft_mode_with_negative_axis(self) -> None:
        """top_k with axis=-1 produces matching (k, n) soft indices.

        Regression test: the rank axis in soft_index shifts when the
        probability-distribution axis is appended by argsort. Prior to the
        fix this path raised ``ValueError: Size of label 'n' ... does not
        match previous terms`` from the take_along_axis einsum.
        """
        from diffbio.core.soft_ops.sorting import top_k

        x = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        values, indices = top_k(x, k=2, axis=-1, softness=0.1, mode="smooth")
        assert values.shape == (2,)
        assert indices.shape == (2, 5)

    def test_soft_mode_with_negative_axis_batched(self) -> None:
        """top_k with axis=-1 on a 2D batch produces (batch, k, n) indices."""
        from diffbio.core.soft_ops.sorting import top_k

        x = jnp.array([[1.0, 5.0, 3.0, 2.0, 4.0], [4.0, 1.0, 2.0, 5.0, 3.0]])
        values, indices = top_k(x, k=2, axis=-1, softness=0.1, mode="smooth")
        assert values.shape == (2, 2)
        assert indices.shape == (2, 2, 5)
