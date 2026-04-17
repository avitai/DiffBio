"""Tests for advanced sorting methods (fast_soft_sort, smooth_sort, ot).

These methods require optional dependencies (optimistix, lineax, ott-jax).
Tests are skipped if dependencies are not installed.
"""

import warnings

import jax.numpy as jnp
import pytest

from diffbio.core.soft_ops.elementwise import SigmoidalMode
from tests.core.test_soft_ops.conftest import assert_finite_grads, assert_simplex

_SOFT_MODES: list[SigmoidalMode] = ["smooth", "c0", "c1", "c2"]

try:
    import optimistix  # noqa: F401

    HAS_OPTIMISTIX = True
except ImportError:
    HAS_OPTIMISTIX = False

requires_optimistix = pytest.mark.skipif(
    not HAS_OPTIMISTIX,
    reason="optimistix not installed",
)


class TestFastSoftSort:
    """Test fast_soft_sort method (permutahedron PAV projection)."""

    @requires_optimistix
    @pytest.mark.parametrize("mode", _SOFT_MODES)
    def test_sort_output_shape(self, mode: SigmoidalMode) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = sort(
            x,
            axis=0,
            softness=0.1,
            mode=mode,
            method="fast_soft_sort",
        )
        assert result.shape == x.shape

    @requires_optimistix
    def test_sort_ascending_approaches_hard(self) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jnp.array([5.0, 1.0, 3.0])
        result = sort(
            x,
            axis=0,
            softness=0.01,
            mode="c0",
            method="fast_soft_sort",
        )
        expected = jnp.sort(x)
        assert jnp.allclose(result, expected, atol=1.0)

    @requires_optimistix
    def test_sort_descending(self) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jnp.array([5.0, 1.0, 3.0])
        result = sort(
            x,
            axis=0,
            softness=0.01,
            mode="c0",
            method="fast_soft_sort",
            descending=True,
        )
        expected = jnp.sort(x)[::-1]
        assert jnp.allclose(result, expected, atol=1.0)

    @requires_optimistix
    def test_differentiable(self) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jnp.array([3.0, 1.0, 2.0])
        assert_finite_grads(
            lambda x: sort(
                x,
                axis=0,
                softness=0.1,
                mode="c0",
                method="fast_soft_sort",
            ),
            (x,),
        )


class TestSmoothSort:
    """Test smooth_sort method (ESP + LBFGS permutahedron)."""

    @requires_optimistix
    def test_sort_output_shape(self) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = sort(
            x,
            axis=0,
            softness=0.1,
            mode="smooth",
            method="smooth_sort",
        )
        assert result.shape == x.shape

    @requires_optimistix
    def test_only_smooth_mode(self) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jnp.array([3.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="smooth_sort only supports"):
            sort(x, axis=0, softness=0.1, mode="c0", method="smooth_sort")

    @requires_optimistix
    def test_differentiable(self) -> None:
        from diffbio.core.soft_ops.sorting import sort

        x = jnp.array([3.0, 1.0, 2.0])
        assert_finite_grads(
            lambda x: sort(
                x,
                axis=0,
                softness=0.5,
                mode="smooth",
                method="smooth_sort",
            ),
            (x,),
        )


class TestOTMethod:
    """Test optimal transport method for argmax and argsort."""

    @requires_optimistix
    def test_argmax_ot(self) -> None:
        from diffbio.core.soft_ops.sorting import argmax

        x = jnp.array([1.0, 5.0, 3.0])
        result = argmax(
            x,
            axis=0,
            softness=0.1,
            mode="smooth",
            method="ot",
            ot_kwargs={"use_entropic_ot_sinkhorn_on_entropic": False},
        )
        assert result.shape == (3,)
        assert_simplex(result, axis=-1, atol=0.1)

    @requires_optimistix
    def test_argsort_ot(self) -> None:
        from diffbio.core.soft_ops.sorting import argsort

        x = jnp.array([3.0, 1.0, 2.0])
        result = argsort(
            x,
            axis=0,
            softness=0.1,
            mode="smooth",
            method="ot",
            ot_kwargs={"use_entropic_ot_sinkhorn_on_entropic": False},
        )
        assert result.shape == (3, 3)
        for i in range(3):
            assert_simplex(result[i], axis=-1, atol=0.1)

    @requires_optimistix
    def test_argmax_ot_differentiable(self) -> None:
        from diffbio.core.soft_ops.sorting import argmax

        x = jnp.array([1.0, 5.0, 3.0])
        assert_finite_grads(
            lambda x: argmax(
                x,
                axis=0,
                softness=0.1,
                mode="smooth",
                method="ot",
                ot_kwargs={"use_entropic_ot_sinkhorn_on_entropic": False},
            ),
            (x,),
        )

    @requires_optimistix
    def test_ot_method_does_not_request_unavailable_float64(self) -> None:
        from diffbio.core.soft_ops.sorting import argmax

        x = jnp.array([1.0, 5.0, 3.0], dtype=jnp.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            result = argmax(
                x,
                axis=0,
                softness=0.1,
                mode="smooth",
                method="ot",
                ot_kwargs={"use_entropic_ot_sinkhorn_on_entropic": False},
            )

        assert result.shape == (3,)
        assert not any(
            "Explicitly requested dtype float64" in str(w.message) for w in caught
        )
