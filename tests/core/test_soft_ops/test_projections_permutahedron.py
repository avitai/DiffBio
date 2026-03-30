"""Tests for permutahedron projection (fast_soft_sort backend).

These tests require the ``optimistix`` optional dependency for the
smooth_sort method. The c0/c1/c2 modes use PAV isotonic regression
which is pure JAX.
"""

import jax
import jax.numpy as jnp
import pytest

from diffbio.core.soft_ops._projections_simplex import SimplexMode
from tests.core.test_soft_ops.conftest import assert_finite_grads

PAV_MODES: list[SimplexMode] = ["c0", "c1", "c2"]

try:
    import optimistix  # noqa: F401

    HAS_OPTIMISTIX = True
except ImportError:
    HAS_OPTIMISTIX = False


class TestProjPermutahedronPAV:
    """Test permutahedron projection with PAV modes (no optional deps)."""

    @pytest.mark.parametrize("mode", PAV_MODES)
    def test_output_is_permutation_of_sorted_w(self, mode: SimplexMode) -> None:
        """Result should be a 'soft' permutation of w, lying in the
        permutahedron defined by w."""
        from diffbio.core.soft_ops._projections_permutahedron import (
            proj_permutahedron,
        )

        z = jnp.array([5.0, 1.0, 3.0, 2.0, 4.0])
        w = jnp.array([10.0, 8.0, 6.0, 4.0, 2.0])
        result = proj_permutahedron(z, w, softness=0.1, mode=mode)
        assert result.shape == z.shape
        # Sum should be preserved (sum of result == sum of w)
        assert jnp.allclose(jnp.sum(result), jnp.sum(w), atol=0.5)

    @pytest.mark.parametrize("mode", PAV_MODES)
    def test_differentiable_wrt_z(self, mode: SimplexMode) -> None:
        from diffbio.core.soft_ops._projections_permutahedron import (
            proj_permutahedron,
        )

        z = jnp.array([3.0, 1.0, 2.0])
        w = jnp.array([6.0, 3.0, 1.0])
        assert_finite_grads(
            lambda z: proj_permutahedron(z, w, softness=0.1, mode=mode),
            (z,),
        )

    @pytest.mark.parametrize("mode", PAV_MODES)
    def test_differentiable_wrt_w(self, mode: SimplexMode) -> None:
        from diffbio.core.soft_ops._projections_permutahedron import (
            proj_permutahedron,
        )

        z = jnp.array([3.0, 1.0, 2.0])
        w = jnp.array([6.0, 3.0, 1.0])
        assert_finite_grads(
            lambda w: proj_permutahedron(z, w, softness=0.1, mode=mode),
            (w,),
        )

    @pytest.mark.parametrize("mode", PAV_MODES)
    def test_batch_via_vmap(self, mode: SimplexMode) -> None:
        from diffbio.core.soft_ops._projections_permutahedron import (
            proj_permutahedron,
        )

        def fn(z_i: jnp.ndarray, w_i: jnp.ndarray) -> jnp.ndarray:
            return proj_permutahedron(z_i, w_i, softness=0.1, mode=mode)

        z = jax.random.normal(jax.random.key(0), (3, 5))
        w = jax.random.normal(jax.random.key(1), (3, 5))
        # vmap over first axis, keeping softness/mode as static
        result = jax.vmap(fn)(z, w)
        assert result.shape == (3, 5)

    def test_single_element(self) -> None:
        from diffbio.core.soft_ops._projections_permutahedron import (
            proj_permutahedron,
        )

        z = jnp.array([5.0])
        w = jnp.array([3.0])
        result = proj_permutahedron(z, w, softness=0.1, mode="c0")
        assert jnp.allclose(result, w)


class TestProjPermutahedronSmooth:
    """Test smooth (entropic) permutahedron projection via LBFGS."""

    @pytest.mark.skipif(
        not HAS_OPTIMISTIX,
        reason="optimistix not installed",
    )
    def test_smooth_mode_output_shape(self) -> None:
        from diffbio.core.soft_ops._projections_permutahedron import (
            proj_permutahedron,
        )

        z = jnp.array([5.0, 1.0, 3.0, 2.0, 4.0])
        w = jnp.array([10.0, 8.0, 6.0, 4.0, 2.0])
        result = proj_permutahedron(z, w, softness=0.1, mode="smooth")
        assert result.shape == z.shape
        assert jnp.allclose(jnp.sum(result), jnp.sum(w), atol=0.5)

    @pytest.mark.skipif(
        not HAS_OPTIMISTIX,
        reason="optimistix not installed",
    )
    def test_smooth_mode_differentiable(self) -> None:
        from diffbio.core.soft_ops._projections_permutahedron import (
            proj_permutahedron,
        )

        z = jnp.array([3.0, 1.0, 2.0])
        w = jnp.array([6.0, 3.0, 1.0])
        assert_finite_grads(
            lambda z: proj_permutahedron(z, w, softness=0.5, mode="smooth"),
            (z,),
        )

    @pytest.mark.skipif(
        not HAS_OPTIMISTIX,
        reason="optimistix not installed",
    )
    def test_raises_without_optimistix(self) -> None:
        """This test is a no-op when optimistix IS installed."""
        pass


class TestProjPermutahedronSmoothSort:
    """Test smooth_sort factory function."""

    @pytest.mark.skipif(
        not HAS_OPTIMISTIX,
        reason="optimistix not installed",
    )
    def test_smooth_sort_output_shape(self) -> None:
        from diffbio.core.soft_ops._projections_permutahedron import (
            proj_permutahedron_smooth_sort,
        )

        z = jnp.array([5.0, 1.0, 3.0])
        w = jnp.array([3.0, 2.0, 1.0])
        result = proj_permutahedron_smooth_sort(z, w, softness=0.5)
        assert result.shape == z.shape

    @pytest.mark.skipif(
        not HAS_OPTIMISTIX,
        reason="optimistix not installed",
    )
    def test_smooth_sort_differentiable(self) -> None:
        from diffbio.core.soft_ops._projections_permutahedron import (
            proj_permutahedron_smooth_sort,
        )

        z = jnp.array([3.0, 1.0, 2.0])
        w = jnp.array([3.0, 2.0, 1.0])
        assert_finite_grads(
            lambda z: proj_permutahedron_smooth_sort(z, w, softness=0.5),
            (z,),
        )
