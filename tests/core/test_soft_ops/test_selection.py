"""Tests for soft selection/indexing operators."""

import jax
import jax.numpy as jnp

from tests.core.test_soft_ops.conftest import assert_finite_grads


class TestWhere:
    """Soft where: x * condition + y * (1 - condition)."""

    def test_hard_condition(self) -> None:
        from diffbio.core.soft_ops.selection import where

        cond = jnp.array([1.0, 0.0, 1.0])
        x = jnp.array([10.0, 20.0, 30.0])
        y = jnp.array([1.0, 2.0, 3.0])
        result = where(cond, x, y)
        expected = jnp.array([10.0, 2.0, 30.0])
        assert jnp.allclose(result, expected)

    def test_soft_condition_interpolates(self) -> None:
        from diffbio.core.soft_ops.selection import where

        cond = jnp.array([0.5])
        x = jnp.array([10.0])
        y = jnp.array([0.0])
        result = where(cond, x, y)
        assert jnp.allclose(result, 5.0)

    def test_differentiable_through_condition(self) -> None:
        from diffbio.core.soft_ops.selection import where

        x = jnp.array([10.0])
        y = jnp.array([0.0])
        assert_finite_grads(
            lambda c: where(c, x, y),
            (jnp.array([0.5]),),
        )


class TestTakeAlongAxis:
    """Soft take_along_axis via weighted dot product."""

    def test_one_hot_index_matches_hard(self) -> None:
        from diffbio.core.soft_ops.selection import take_along_axis

        x = jnp.array([10.0, 20.0, 30.0])
        # SoftIndex: select element 1 (one-hot)
        soft_idx = jnp.array([[0.0, 1.0, 0.0]])  # (1, 3)
        result = take_along_axis(x, soft_idx, axis=0)
        assert jnp.allclose(result, jnp.array([20.0]))

    def test_soft_index_interpolates(self) -> None:
        from diffbio.core.soft_ops.selection import take_along_axis

        x = jnp.array([0.0, 10.0, 20.0])
        soft_idx = jnp.array([[0.5, 0.5, 0.0]])  # (1, 3)
        result = take_along_axis(x, soft_idx, axis=0)
        assert jnp.allclose(result, jnp.array([5.0]))

    def test_batch_dimensions(self) -> None:
        from diffbio.core.soft_ops.selection import take_along_axis

        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        # Select 1 element per row: shape (2, 1, 3)
        soft_idx = jax.nn.one_hot(jnp.array([[0], [2]]), 3)  # (2, 1, 3)
        result = take_along_axis(x, soft_idx, axis=1)
        assert result.shape == (2, 1)
        assert jnp.allclose(result, jnp.array([[1.0], [6.0]]))

    def test_differentiable(self) -> None:
        from diffbio.core.soft_ops.selection import take_along_axis

        x = jnp.array([1.0, 2.0, 3.0])
        soft_idx = jnp.array([[0.3, 0.5, 0.2]])
        assert_finite_grads(
            lambda x: take_along_axis(x, soft_idx, axis=0),
            (x,),
        )


class TestTake:
    """Soft take via weighted dot product."""

    def test_one_hot_matches_hard(self) -> None:
        from diffbio.core.soft_ops.selection import take

        x = jnp.array([10.0, 20.0, 30.0])
        soft_idx = jax.nn.one_hot(jnp.array([2, 0]), 3)  # (2, 3)
        result = take(x, soft_idx, axis=0)
        assert jnp.allclose(result, jnp.array([30.0, 10.0]))


class TestChoose:
    """Soft choose among multiple arrays."""

    def test_one_hot_selects_correct_choice(self) -> None:
        from diffbio.core.soft_ops.selection import choose

        choices = jnp.array([[1.0], [2.0], [3.0]])  # (3, 1)
        soft_idx = jnp.array([[0.0, 0.0, 1.0]])  # (1, 3)
        result = choose(soft_idx, choices)
        assert jnp.allclose(result, jnp.array([3.0]))

    def test_weighted_mix(self) -> None:
        from diffbio.core.soft_ops.selection import choose

        choices = jnp.array([[0.0], [10.0]])  # (2, 1)
        soft_idx = jnp.array([[0.3, 0.7]])  # (1, 2)
        result = choose(soft_idx, choices)
        assert jnp.allclose(result, jnp.array([7.0]))


class TestDynamicIndexInDim:
    """Soft dynamic indexing along a dimension."""

    def test_one_hot_selects_element(self) -> None:
        from diffbio.core.soft_ops.selection import dynamic_index_in_dim

        x = jnp.array([10.0, 20.0, 30.0])
        soft_idx = jnp.array([0.0, 1.0, 0.0])  # select index 1
        result = dynamic_index_in_dim(x, soft_idx, axis=0, keepdims=True)
        assert result.shape == (1,)
        assert jnp.allclose(result, jnp.array([20.0]))

    def test_keepdims_false(self) -> None:
        from diffbio.core.soft_ops.selection import dynamic_index_in_dim

        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        soft_idx = jnp.array([0.0, 0.0, 1.0])  # select last row
        result = dynamic_index_in_dim(x, soft_idx, axis=0, keepdims=False)
        assert result.shape == (2,)
        assert jnp.allclose(result, jnp.array([5.0, 6.0]))


class TestDynamicSliceInDim:
    """Soft dynamic slicing along a dimension."""

    def test_hard_start_index(self) -> None:
        from diffbio.core.soft_ops.selection import dynamic_slice_in_dim

        x = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        soft_start = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])  # start at 1
        result = dynamic_slice_in_dim(x, soft_start, slice_size=3, axis=0)
        assert result.shape == (3,)
        assert jnp.allclose(result, jnp.array([20.0, 30.0, 40.0]), atol=1e-5)


class TestDynamicSlice:
    """Soft dynamic slicing across multiple dimensions."""

    def test_2d_slice(self) -> None:
        from diffbio.core.soft_ops.selection import dynamic_slice

        x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        start_0 = jnp.array([0.0, 1.0, 0.0])  # start at row 1
        start_1 = jnp.array([0.0, 1.0, 0.0, 0.0])  # start at col 1
        result = dynamic_slice(x, [start_0, start_1], [2, 2])
        assert result.shape == (2, 2)
        expected = x[1:3, 1:3]
        assert jnp.allclose(result, expected, atol=1e-5)
