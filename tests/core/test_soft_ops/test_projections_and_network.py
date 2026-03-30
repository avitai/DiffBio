"""Tests for simplex projection and bitonic sorting network.

These are internal modules that provide the mathematical backbone
for soft sorting, argmax, and quantile operations.
"""

import jax
import jax.numpy as jnp
import pytest

from diffbio.core.soft_ops._projections_simplex import SimplexMode
from tests.core.test_soft_ops.conftest import assert_finite_grads, assert_simplex

SIMPLEX_MODES: list[SimplexMode] = ["smooth", "c0", "c1", "c2"]


class TestProjSimplex:
    """Test simplex projection across all modes."""

    @pytest.mark.parametrize("mode", SIMPLEX_MODES)
    def test_output_is_simplex(self, mode: SimplexMode) -> None:
        """Projection output must be non-negative and sum to 1."""
        from diffbio.core.soft_ops._projections_simplex import proj_simplex

        x = jax.random.normal(jax.random.key(0), (5,))
        result = proj_simplex(x, axis=0, softness=0.1, mode=mode)
        assert_simplex(result, axis=0, atol=1e-4)

    @pytest.mark.parametrize("mode", SIMPLEX_MODES)
    def test_batch_dimensions(self, mode: SimplexMode) -> None:
        """Projection works with batch dimensions."""
        from diffbio.core.soft_ops._projections_simplex import proj_simplex

        x = jax.random.normal(jax.random.key(1), (3, 5))
        result = proj_simplex(x, axis=1, softness=0.1, mode=mode)
        assert result.shape == (3, 5)
        for i in range(3):
            assert_simplex(result[i], axis=0, atol=1e-4)

    def test_smooth_mode_is_softmax(self) -> None:
        """Smooth mode should produce softmax output."""
        from diffbio.core.soft_ops._projections_simplex import proj_simplex

        x = jax.random.normal(jax.random.key(2), (5,))
        softness = 0.5
        result = proj_simplex(x, axis=0, softness=softness, mode="smooth")
        expected = jax.nn.softmax(x / softness, axis=0)
        assert jnp.allclose(result, expected, atol=1e-6)

    @pytest.mark.parametrize("mode", SIMPLEX_MODES)
    def test_differentiable(self, mode: SimplexMode) -> None:
        """Gradient through projection must be finite."""
        from diffbio.core.soft_ops._projections_simplex import proj_simplex

        x = jax.random.normal(jax.random.key(3), (5,))
        assert_finite_grads(
            lambda x: proj_simplex(x, axis=0, softness=0.1, mode=mode),
            (x,),
        )

    def test_invalid_mode_raises(self) -> None:
        from diffbio.core.soft_ops._projections_simplex import proj_simplex

        x = jnp.ones(3)
        with pytest.raises(ValueError, match="Invalid mode"):
            proj_simplex(x, axis=0, softness=0.1, mode="bogus")  # type: ignore[arg-type]

    def test_low_softness_approaches_hard_argmax(self) -> None:
        """With very low softness, smooth mode should approach one-hot."""
        from diffbio.core.soft_ops._projections_simplex import proj_simplex

        x = jnp.array([1.0, 3.0, 2.0])
        result = proj_simplex(x, axis=0, softness=0.001, mode="smooth")
        assert float(result[1]) > 0.99  # argmax at index 1


class TestSortViaSortingNetwork:
    """Test bitonic sorting network for soft sorting."""

    def test_sort_ascending(self) -> None:
        from diffbio.core.soft_ops._sorting_network import sort_via_sorting_network

        x = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        result = sort_via_sorting_network(
            x,
            softness=0.01,
            mode="smooth",
            descending=False,
        )
        expected = jnp.sort(x)
        assert jnp.allclose(result, expected, atol=0.1)

    def test_sort_descending(self) -> None:
        from diffbio.core.soft_ops._sorting_network import sort_via_sorting_network

        x = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        result = sort_via_sorting_network(
            x,
            softness=0.01,
            mode="smooth",
            descending=True,
        )
        expected = jnp.sort(x)[::-1]
        assert jnp.allclose(result, expected, atol=0.1)

    def test_non_power_of_two(self) -> None:
        """Handles input length that isn't a power of 2."""
        from diffbio.core.soft_ops._sorting_network import sort_via_sorting_network

        x = jnp.array([5.0, 2.0, 8.0, 1.0, 3.0])  # length 5
        result = sort_via_sorting_network(
            x,
            softness=0.01,
            mode="smooth",
            descending=False,
        )
        assert result.shape == (5,)
        expected = jnp.sort(x)
        assert jnp.allclose(result, expected, atol=0.2)

    def test_batch_sort(self) -> None:
        from diffbio.core.soft_ops._sorting_network import sort_via_sorting_network

        x = jax.random.normal(jax.random.key(0), (3, 4))
        result = sort_via_sorting_network(
            x,
            softness=0.01,
            mode="smooth",
            descending=False,
        )
        assert result.shape == (3, 4)

    def test_differentiable(self) -> None:
        from diffbio.core.soft_ops._sorting_network import sort_via_sorting_network

        x = jnp.array([3.0, 1.0, 4.0, 2.0])
        assert_finite_grads(
            lambda x: sort_via_sorting_network(
                x,
                softness=0.1,
                mode="smooth",
                descending=False,
            ),
            (x,),
        )


class TestArgsortViaSortingNetwork:
    """Test bitonic sorting network for soft argsort (permutation matrix)."""

    def test_output_is_permutation_matrix(self) -> None:
        from diffbio.core.soft_ops._sorting_network import (
            argsort_via_sorting_network,
        )

        x = jnp.array([3.0, 1.0, 4.0, 2.0])
        P = argsort_via_sorting_network(
            x,
            softness=0.01,
            mode="smooth",
            descending=False,
        )
        assert P.shape == (4, 4)
        # Rows should sum to ~1 (doubly stochastic)
        row_sums = jnp.sum(P, axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=0.1)

    def test_differentiable(self) -> None:
        from diffbio.core.soft_ops._sorting_network import (
            argsort_via_sorting_network,
        )

        x = jnp.array([3.0, 1.0, 4.0, 2.0])
        assert_finite_grads(
            lambda x: argsort_via_sorting_network(
                x,
                softness=0.1,
                mode="smooth",
                descending=False,
            ),
            (x,),
        )
