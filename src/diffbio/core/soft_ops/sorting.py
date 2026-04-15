"""Soft sorting, argmax/argmin, argsort, rank, and top-k operators.

Provides differentiable relaxations of discrete ordering operations
using multiple algorithmic approaches:

- **softsort**: Simplex projection (O(n log n)). Default for argmax/argmin.
- **neuralsort**: Pairwise comparison + simplex projection (O(n^2)).
  Default for argsort/sort.
- **sorting_network**: Bitonic sorting network (O(n log^2 n)).
- **ot**: Optimal transport projection (requires optional deps).
- **fast_soft_sort**: Permutahedron projection via PAV (requires optional deps).
- **smooth_sort**: Smooth permutahedron via ESP bounds (requires optional deps).

The ``ot``, ``fast_soft_sort``, and ``smooth_sort`` methods require
the ``soft-ops-advanced`` optional dependency group. They raise
``ImportError`` with a helpful message if called without installation.
"""

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from diffbio.core.soft_ops._projections_simplex import SimplexMode, proj_simplex
from diffbio.core.soft_ops._sorting_network import (
    argsort_via_sorting_network,
    sort_via_sorting_network,
)
from diffbio.core.soft_ops._types import SoftIndex
from diffbio.core.soft_ops._utils import (
    canonicalize_axis,
    ensure_float,
    map_in_chunks,
    normalize_axis_argument,
    reduce_in_chunks,
    standardize_and_squash,
    unsquash_and_destandardize,
)
from diffbio.core.soft_ops.elementwise import abs as soft_abs
from diffbio.core.soft_ops.selection import take_along_axis

# Optional-dependency imports (permutahedron + transport polytope).
# These are lazy-loaded at call time to avoid ImportError at import.
_ADVANCED_INSTALL_MSG = "Install with: uv pip install -e '.[soft-ops-advanced]'"


def _get_proj_permutahedron():
    """Lazy import of permutahedron projection."""
    from diffbio.core.soft_ops._projections_permutahedron import (
        proj_permutahedron,
    )

    return proj_permutahedron


def _get_proj_permutahedron_smooth_sort():
    """Lazy import of smooth sort permutahedron projection."""
    from diffbio.core.soft_ops._projections_permutahedron import (
        proj_permutahedron_smooth_sort,
    )

    return proj_permutahedron_smooth_sort


def _get_proj_transport_polytope():
    """Lazy import of transport polytope projection."""
    from diffbio.core.soft_ops._projections_transport import (
        proj_transport_polytope,
    )

    return proj_transport_polytope


Mode = Literal["hard", "smooth", "c0", "c1", "c2"]
ArgMethod = Literal["softsort", "neuralsort", "sorting_network", "ot"]
RankMethod = Literal["softsort", "neuralsort"]
SortMethod = Literal[
    "softsort",
    "neuralsort",
    "sorting_network",
    "ot",
    "fast_soft_sort",
    "smooth_sort",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _neuralsort_a_sum(
    x_last: Array,
    mode: SimplexMode,
    softness: float | Array,
) -> Array:
    """``a_sum[..., j] = sum_i soft_abs(x[..., i] - x[..., j])``."""
    n = x_last.shape[-1]
    x_flat = x_last.reshape(-1, n)

    def _single(x_row: Array) -> Array:
        """Compute pairwise absolute difference sums for one row."""

        def _chunk_fn(x_chunk_j: Array) -> Array:
            """Sum absolute differences against a chunk of columns."""
            return soft_abs(
                x_row[:, None] - x_chunk_j[None, :],
                mode=mode,
                softness=softness,
            ).sum(axis=0)

        return map_in_chunks(f=_chunk_fn, xs=x_row, chunk_size=128)

    return jax.vmap(_single)(x_flat).reshape(x_last.shape)


def _sorting_network_permutation(
    x_last: Array,
    softness: float | Array,
    mode: SimplexMode,
    *,
    descending: bool,
    standardized: bool,
) -> Array:
    """Return the differentiable permutation from the sorting-network backend."""
    return argsort_via_sorting_network(
        x_last,
        softness,
        mode,
        descending=descending,
        standardized=standardized,
    )


def _sorting_network_argmax_index(
    x_last: Array,
    softness: float | Array,
    mode: SimplexMode,
    *,
    standardize: bool,
) -> Array:
    """Return the soft argmax index from the sorting-network backend."""
    perm = _sorting_network_permutation(
        x_last,
        softness,
        mode,
        descending=True,
        standardized=standardize,
    )
    return perm[..., 0, :]


def _softsort_fused_sort(
    x_last: Array,
    batch_dims: list[int],
    softness: float | Array,
    mode: SimplexMode,
    descending: bool,
    standardize: bool,
    gated_grad: bool,
) -> Array:
    """Sorted values via SoftSort, O(n) memory."""
    n = x_last.shape[-1]
    x_std = standardize_and_squash(x_last, axis=-1) if standardize else x_last
    x_orig_flat = x_last.reshape(-1, n)
    x_std_flat = x_std.reshape(-1, n)

    def _single(x_orig_row: Array, x_std_row: Array) -> Array:
        """Compute soft-sorted values for a single row via SoftSort."""

        def _chunk_fn(anchors_chunk: Array) -> Array:
            """Project an anchor chunk onto the simplex and gather values."""
            diff = jnp.abs(anchors_chunk[:, None] - x_std_row[None, :])
            p_chunk = proj_simplex(-diff, axis=-1, softness=softness, mode=mode)
            if not gated_grad:
                p_chunk = jax.lax.stop_gradient(p_chunk)
            return jnp.einsum("cn,n->c", p_chunk, x_orig_row)

        anchors_row = jnp.sort(x_std_row, descending=descending)
        return map_in_chunks(f=_chunk_fn, xs=anchors_row, chunk_size=128)

    result = jax.vmap(_single)(x_orig_flat, x_std_flat)
    return result.reshape(*batch_dims, n)


def _neuralsort_fused_sort(
    x_last: Array,
    batch_dims: list[int],
    softness: float | Array,
    mode: SimplexMode,
    descending: bool,
    standardize: bool,
    gated_grad: bool,
) -> Array:
    """Sorted values via NeuralSort, O(n) memory."""
    n = x_last.shape[-1]
    x_std = standardize_and_squash(x_last, axis=-1) if standardize else x_last
    a_sum = _neuralsort_a_sum(x_last=x_std, mode=mode, softness=softness)

    i = jnp.arange(1, n + 1)
    if descending:
        i = i[::-1]
    coef = n + 1 - 2 * i
    coef = jnp.broadcast_to(coef, (*batch_dims, n))

    x_orig_flat = x_last.reshape(-1, n)
    x_std_flat = x_std.reshape(-1, n)
    a_sum_flat = a_sum.reshape(-1, n)
    coef_flat = coef.reshape(-1, n)

    def _single(
        x_orig_row: Array,
        x_std_row: Array,
        a_sum_row: Array,
        coef_row: Array,
    ) -> Array:
        """Compute soft-sorted values for a single row via NeuralSort."""

        def _chunk_fn(coef_chunk: Array) -> Array:
            """Project a coefficient chunk onto the simplex and gather values."""
            z_chunk = -(coef_chunk[:, None] * x_std_row[None, :] + a_sum_row[None, :])
            p_chunk = proj_simplex(z_chunk, axis=-1, softness=softness, mode=mode)
            if not gated_grad:
                p_chunk = jax.lax.stop_gradient(p_chunk)
            return jnp.einsum("cn,n->c", p_chunk, x_orig_row)

        return map_in_chunks(f=_chunk_fn, xs=coef_row, chunk_size=128)

    result = jax.vmap(_single)(
        x_orig_flat,
        x_std_flat,
        a_sum_flat,
        coef_flat,
    )
    return result.reshape(*batch_dims, n)


def _softsort_fused_rank(
    x_last: Array,
    batch_dims: list[int],
    softness: float | Array,
    mode: SimplexMode,
    descending: bool,
) -> Array:
    """Ranks via SoftSort, O(n) memory. x_last should be standardized."""
    n = x_last.shape[-1]
    nums = jnp.arange(1, n + 1, dtype=x_last.dtype)
    x_flat = x_last.reshape(-1, n)

    def _single(x_row: Array) -> Array:
        """Compute soft ranks for a single row via SoftSort."""

        def _chunk_fn(x_chunk: Array) -> Array:
            """Compute rank contributions for a chunk of elements."""
            diff = jnp.abs(x_chunk[:, None] - anchors_row[None, :])
            p_chunk = proj_simplex(-diff, axis=-1, softness=softness, mode=mode)
            return jnp.einsum("cn,n->c", p_chunk, nums)

        anchors_row = jnp.sort(x_row, descending=descending)
        return map_in_chunks(f=_chunk_fn, xs=x_row, chunk_size=128)

    result = jax.vmap(_single)(x_flat)
    return result.reshape(*batch_dims, n)


def _neuralsort_fused_rank(
    x_last: Array,
    batch_dims: list[int],
    softness: float | Array,
    mode: SimplexMode,
    descending: bool,
) -> Array:
    """Ranks via NeuralSort, O(n) memory. x_last should be standardized."""
    n = x_last.shape[-1]
    nums = jnp.arange(1, n + 1, dtype=x_last.dtype)
    row_sums = _neuralsort_a_sum(x_last=x_last, mode=mode, softness=softness)

    i = jnp.arange(1, n + 1)
    if descending:
        i = i[::-1]
    coef = n + 1 - 2 * i
    coef = jnp.broadcast_to(
        coef.reshape(*(1,) * len(batch_dims), n),
        (*batch_dims, n),
    )

    x_flat = x_last.reshape(-1, n)
    row_sums_flat = row_sums.reshape(-1, n)
    coef_flat = coef.reshape(-1, n)

    def _single(
        x_row: Array,
        row_sums_row: Array,
        coef_row: Array,
    ) -> Array:
        """Compute soft ranks for a single row via NeuralSort."""
        coef_and_nums = jnp.stack([coef_row, nums], axis=-1)

        def _chunk_fn(data_chunk: Array) -> Array:
            """Accumulate rank contributions from a chunk of coefficients."""
            coef_chunk = data_chunk[:, 0]
            nums_chunk = data_chunk[:, 1]
            z_chunk = -(coef_chunk[:, None] * x_row[None, :] + row_sums_row[None, :])
            p_chunk = proj_simplex(z_chunk, axis=-1, softness=softness, mode=mode)
            col_sum = p_chunk.sum(axis=0)
            weighted = (nums_chunk[:, None] * p_chunk).sum(axis=0)
            return jnp.stack([col_sum, weighted])

        result = reduce_in_chunks(f=_chunk_fn, xs=coef_and_nums, chunk_size=128)
        col_sums = result[0]
        weighted_sums = result[1]
        return weighted_sums / jnp.clip(col_sums, min=1e-10)

    result = jax.vmap(_single)(x_flat, row_sums_flat, coef_flat)
    return result.reshape(*batch_dims, n)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def argmax(
    x: Array,
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: ArgMethod = "softsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
) -> SoftIndex:
    """Soft argmax returning a SoftIndex (probability distribution).

    Args:
        x: Input array.
        axis: Axis along which to compute argmax. None flattens first.
        keepdims: If True, keep the reduced dimension as singleton.
        softness: Controls sharpness (> 0).
        mode: Smoothness mode.
        method: Algorithm: ``"softsort"``, ``"neuralsort"``,
            ``"sorting_network"``, or ``"ot"``.
        standardize: If True, standardize input for numerical stability.
        ot_kwargs: Extra kwargs for OT method.

    Returns:
        SoftIndex of shape ``(..., {1}, ..., [n])``.
    """
    if mode == "hard":
        indices = jnp.argmax(x, axis=axis, keepdims=keepdims)
        num_classes = jnp.size(x, axis=axis)
        return jax.nn.one_hot(indices, num_classes=num_classes, axis=-1)

    x = ensure_float(x)
    if axis is None:
        num_dims = x.ndim
        x = jnp.ravel(x)
        _axis = 0
    else:
        _axis = canonicalize_axis(axis, x.ndim)
        num_dims = None

    if standardize:
        x = standardize_and_squash(x, axis=_axis)

    x_last = jnp.moveaxis(x, _axis, -1)
    *batch_dims, n = x_last.shape

    if method == "softsort":
        soft_index = proj_simplex(
            x_last,
            axis=-1,
            softness=softness,
            mode=mode,
        )
    elif method == "neuralsort":
        a_sum = _neuralsort_a_sum(x_last, mode=mode, softness=softness)
        z = (n - 1) * x_last - a_sum
        soft_index = proj_simplex(z, axis=-1, softness=softness, mode=mode)
    elif method == "sorting_network":
        soft_index = _sorting_network_argmax_index(
            x_last,
            softness,
            mode,
            standardize=standardize,
        )
    elif method == "ot":
        _proj_tp = _get_proj_transport_polytope()
        anchors = jnp.array([0.0, 1.0], dtype=x.dtype)
        anchors = jnp.broadcast_to(anchors, (*batch_dims, 2))
        cost = (x_last[..., :, None] - anchors[..., None, :]) ** 2
        mu = jnp.ones((n,), dtype=x.dtype) / n
        nu = jnp.array([(n - 1) / n, 1 / n], dtype=x.dtype)
        if ot_kwargs is None:
            ot_kwargs = {}
        out = _proj_tp(
            cost=cost,
            mu=mu,
            nu=nu,
            softness=softness,
            mode=mode,
            **ot_kwargs,
        )
        soft_index = out[..., :, 1]
    else:
        msg = f"Invalid method: {method!r}"
        raise ValueError(msg)

    if keepdims:
        if num_dims is not None:
            soft_index = soft_index.reshape(*(1,) * num_dims, n)
        else:
            soft_index = jnp.expand_dims(soft_index, axis=_axis)
    return soft_index


def max(
    x: Array,
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: SortMethod = "softsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> Array:
    """Soft max via argmax + take_along_axis.

    For ``sorting_network`` method, uses sort + take first element.

    Args:
        x: Input array.
        axis: Axis along which to compute max.
        keepdims: If True, keep reduced dimension.
        softness: Controls sharpness (> 0).
        mode: Smoothness mode.
        method: Algorithm (see :func:`argmax` and :func:`sort`).
        standardize: If True, standardize input.
        ot_kwargs: Extra kwargs for OT method.
        gated_grad: If False, stop gradient through soft index.

    Returns:
        Soft maximum value(s).
    """
    if mode == "hard":
        return jnp.max(x, axis=axis, keepdims=keepdims)

    if axis is None:
        num_dims = x.ndim
        x = jnp.ravel(x)
        _axis = 0
    else:
        _axis = canonicalize_axis(axis, x.ndim)
        num_dims = None

    sort_methods: set[str] = {"sorting_network", "fast_soft_sort", "smooth_sort"}
    if method in sort_methods:
        soft_sorted = sort(
            x,
            axis=_axis,
            descending=True,
            softness=softness,
            standardize=standardize,
            mode=mode,
            method=method,
        )
        max_val = jnp.take(soft_sorted, indices=0, axis=_axis)
        if num_dims is not None and keepdims:
            max_val = max_val.reshape(*(1,) * num_dims)
        elif keepdims:
            max_val = jnp.expand_dims(max_val, axis=_axis)
    else:
        # method is one of ArgMethod: softsort, neuralsort, sorting_network, ot
        arg_method: ArgMethod = method  # type: ignore[assignment]
        soft_index = argmax(
            x,
            axis=_axis,
            keepdims=True,
            softness=softness,
            mode=mode,
            method=arg_method,
            standardize=standardize,
            ot_kwargs=ot_kwargs,
        )
        if not gated_grad:
            soft_index = jax.lax.stop_gradient(soft_index)
        max_val = take_along_axis(x, soft_index, axis=_axis)
        if num_dims is not None:
            max_val = max_val.reshape(*(1,) * num_dims)
        if not keepdims:
            max_val = jnp.squeeze(max_val, axis=axis)
    return max_val


def argmin(
    x: Array,
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: ArgMethod = "softsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
) -> SoftIndex:
    """Soft argmin: :func:`argmax` on ``-x``."""
    return argmax(
        -x,
        axis=axis,
        mode=mode,
        method=method,
        softness=softness,
        keepdims=keepdims,
        standardize=standardize,
        ot_kwargs=ot_kwargs,
    )


def min(
    x: Array,
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: SortMethod = "softsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> Array:
    """Soft min: ``-max(-x)``."""
    return -max(
        -x,
        axis=axis,
        softness=softness,
        mode=mode,
        method=method,
        keepdims=keepdims,
        standardize=standardize,
        ot_kwargs=ot_kwargs,
        gated_grad=gated_grad,
    )


def argsort(
    x: Array,
    axis: int | None = None,
    descending: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: ArgMethod = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
) -> SoftIndex:
    """Soft argsort returning a soft permutation matrix.

    Output shape is ``(..., n, ..., [n])`` where the last dimension
    is the probability distribution over original elements.

    Args:
        x: Input array.
        axis: Axis along which to argsort. None flattens first.
        descending: If True, sort descending.
        softness: Controls sharpness (> 0).
        mode: Smoothness mode.
        method: Algorithm.
        standardize: If True, standardize input.
        ot_kwargs: Extra kwargs for OT method.

    Returns:
        SoftIndex permutation matrix.
    """
    if mode == "hard":
        indices = jnp.argsort(x, axis=axis, descending=descending)
        num_classes = jnp.size(x, axis=axis)
        return jax.nn.one_hot(indices, num_classes=num_classes, axis=-1)

    x = ensure_float(x)
    x, axis = normalize_axis_argument(x, axis)

    if standardize:
        x = standardize_and_squash(x, axis=axis)

    x_last = jnp.moveaxis(x, axis, -1)
    *batch_dims, n = x_last.shape

    if method == "softsort":
        anchors = jnp.sort(x_last, axis=-1, descending=descending)
        diff = jnp.abs(anchors[..., :, None] - x_last[..., None, :])
        soft_index = proj_simplex(-diff, axis=-1, softness=softness, mode=mode)
    elif method == "neuralsort":
        a_sum = _neuralsort_a_sum(x_last, mode=mode, softness=softness)
        i = jnp.arange(1, n + 1)
        if descending:
            i = i[::-1]
        coef = n + 1 - 2 * i
        coef = jnp.broadcast_to(coef, (*batch_dims, n))
        z = -(coef[..., :, None] * x_last[..., None, :] + a_sum[..., None, :])
        soft_index = proj_simplex(z, axis=-1, softness=softness, mode=mode)
    elif method == "sorting_network":
        soft_index = _sorting_network_permutation(
            x_last,
            softness,
            mode,
            descending=descending,
            standardized=standardize,
        )
    elif method == "ot":
        _proj_tp = _get_proj_transport_polytope()
        anchors = jnp.linspace(0, n, n, dtype=x.dtype) / n
        if descending:
            anchors = anchors[::-1]
        anchors = jnp.broadcast_to(anchors, (*batch_dims, n))
        cost = (x_last[..., :, None] - anchors[..., None, :]) ** 2
        mu = jnp.ones((n,), dtype=x.dtype) / n
        nu = jnp.ones((n,), dtype=x.dtype) / n
        if ot_kwargs is None:
            ot_kwargs = {}
        out = _proj_tp(
            cost=cost,
            mu=mu,
            nu=nu,
            softness=softness,
            mode=mode,
            **ot_kwargs,
        )
        soft_index = jnp.swapaxes(out, -2, -1)
    else:
        msg = f"Invalid method: {method!r}"
        raise ValueError(msg)

    return jnp.moveaxis(soft_index, -2, axis)


def sort(
    x: Array,
    axis: int | None = None,
    descending: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: SortMethod = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> Array:
    """Soft sort returning sorted values.

    Args:
        x: Input array.
        axis: Axis along which to sort. None flattens first.
        descending: If True, sort descending.
        softness: Controls sharpness (> 0).
        mode: Smoothness mode.
        method: Algorithm.
        standardize: If True, standardize input.
        ot_kwargs: Extra kwargs for OT method.
        gated_grad: If False, stop gradient through soft index.

    Returns:
        Soft-sorted values.
    """
    if mode == "hard":
        return jnp.sort(x, axis=axis, descending=descending)

    x = ensure_float(x)
    x, axis = normalize_axis_argument(x, axis)

    if method == "sorting_network":
        if standardize:
            x, mean, std = standardize_and_squash(
                x,
                axis=axis,
                return_mean_std=True,
            )
        x_last = jnp.moveaxis(x, axis, -1)
        soft_values = sort_via_sorting_network(
            x_last,
            softness=softness,
            mode=mode,
            descending=descending,
            standardized=standardize,
        )
        soft_values = jnp.moveaxis(soft_values, -1, axis)
        if standardize:
            soft_values = unsquash_and_destandardize(
                y=soft_values,
                mean=mean,
                std=std,
            )
    elif method == "fast_soft_sort":
        _proj_perm = _get_proj_permutahedron()
        if standardize:
            x, mean, std = standardize_and_squash(
                x,
                axis=axis,
                return_mean_std=True,
            )
        x_last = jnp.moveaxis(x, axis, -1)
        *batch_dims, n = x_last.shape
        w = x_last
        anchors = jnp.arange(n, dtype=x.dtype) / jnp.maximum((n - 1), 1)
        anchors = jnp.broadcast_to(anchors, (*batch_dims, n))
        soft_values = _proj_perm(anchors, w, softness=softness, mode=mode)
        soft_values = jnp.moveaxis(soft_values, -1, axis)
        if descending:
            soft_values = jnp.flip(soft_values, axis=axis)
        if standardize:
            soft_values = unsquash_and_destandardize(
                y=soft_values,
                mean=mean,
                std=std,
            )
    elif method == "smooth_sort":
        _proj_perm_ss = _get_proj_permutahedron_smooth_sort()
        if mode != "smooth":
            msg = f"smooth_sort only supports mode='smooth', got mode={mode!r}"
            raise ValueError(msg)
        x_last = jnp.moveaxis(x, axis, -1)
        *batch_dims, n = x_last.shape
        w = x_last
        anchors = jnp.arange(n, dtype=x.dtype) / jnp.maximum((n - 1), 1)
        anchors = jnp.broadcast_to(anchors, (*batch_dims, n))
        soft_values = _proj_perm_ss(anchors, w, softness=softness)
        soft_values = jnp.moveaxis(soft_values, -1, axis)
        if descending:
            soft_values = jnp.flip(soft_values, axis=axis)
    elif method == "softsort":
        x_last = jnp.moveaxis(x, axis, -1)
        *batch_dims, n = x_last.shape
        soft_values = _softsort_fused_sort(
            x_last,
            batch_dims,
            softness,
            mode,
            descending,
            standardize,
            gated_grad,
        )
        soft_values = jnp.moveaxis(soft_values, -1, axis)
    elif method == "neuralsort":
        x_last = jnp.moveaxis(x, axis, -1)
        *batch_dims, n = x_last.shape
        soft_values = _neuralsort_fused_sort(
            x_last,
            batch_dims,
            softness,
            mode,
            descending,
            standardize,
            gated_grad,
        )
        soft_values = jnp.moveaxis(soft_values, -1, axis)
    else:
        # Fallback: argsort + take_along_axis (for ot method)
        arg_method: ArgMethod = method  # type: ignore[assignment]
        soft_index = argsort(
            x,
            axis=axis,
            descending=descending,
            softness=softness,
            mode=mode,
            method=arg_method,
            standardize=standardize,
            ot_kwargs=ot_kwargs,
        )
        if not gated_grad:
            soft_index = jax.lax.stop_gradient(soft_index)
        soft_values = take_along_axis(x, soft_index, axis=axis)
    return soft_values


def rank(
    x: Array,
    axis: int | None = None,
    descending: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: RankMethod = "softsort",
    standardize: bool = True,
) -> Array:
    """Soft fractional ranking.

    Returns continuous ranks in [1, n] where 1 is the smallest.

    Args:
        x: Input array.
        axis: Axis along which to rank.
        descending: If True, rank 1 = largest.
        softness: Controls sharpness (> 0).
        mode: Smoothness mode.
        method: ``"softsort"`` or ``"neuralsort"``.
        standardize: If True, standardize input.

    Returns:
        Continuous ranks.
    """
    if mode == "hard":
        indices = jnp.argsort(x, axis=axis, descending=descending)
        ranks = jnp.empty_like(indices, dtype=jnp.float32)
        n = jnp.size(x, axis=axis)
        nums = jnp.arange(1, n + 1)
        # Scatter ranks back to original positions
        if axis is None:
            ranks = ranks.ravel()
            ranks = ranks.at[indices.ravel()].set(nums.astype(jnp.float32))
            return ranks.reshape(x.shape)
        ranks = jnp.take_along_axis(
            jnp.broadcast_to(
                jnp.expand_dims(nums, tuple(range(x.ndim - 1))),
                x.shape,
            ).astype(jnp.float32),
            jnp.argsort(indices, axis=axis),
            axis=axis,
        )
        return ranks

    x = ensure_float(x)
    x, axis = normalize_axis_argument(x, axis)

    if standardize:
        x = standardize_and_squash(x, axis=axis)

    x_last = jnp.moveaxis(x, axis, -1)
    *batch_dims, n = x_last.shape

    if method == "softsort":
        result = _softsort_fused_rank(
            x_last,
            batch_dims,
            softness,
            mode,
            descending,
        )
    elif method == "neuralsort":
        result = _neuralsort_fused_rank(
            x_last,
            batch_dims,
            softness,
            mode,
            descending,
        )
    else:
        msg = f"Invalid method for rank: {method!r}"
        raise ValueError(msg)

    return jnp.moveaxis(result, -1, axis)


def top_k(
    x: Array,
    k: int,
    axis: int = -1,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: SortMethod = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> tuple[Array, SoftIndex | None]:
    """Soft top-k selection.

    Returns the k largest values and their soft indices.

    Args:
        x: Input array.
        k: Number of top elements.
        axis: Axis along which to select. Default -1 (last axis).
        softness: Controls sharpness (> 0).
        mode: Smoothness mode.
        method: Sorting algorithm. Default ``"neuralsort"``.
        standardize: If True, standardize input.
        ot_kwargs: Extra keyword arguments for OT-based methods.
        gated_grad: If False, stop gradient through soft index.

    Returns:
        Tuple of (values, soft_indices) where values has shape
        ``(..., k, ...)`` and soft_indices has shape
        ``(..., k, ..., [n])``. soft_indices may be None for
        methods that only return values (fast_soft_sort, sorting_network).
    """
    if mode == "hard":
        indices = jnp.argsort(x, axis=axis, descending=True)
        indices_k = jnp.take(indices, jnp.arange(k), axis=axis)
        values = jnp.take_along_axis(x, indices_k, axis=axis)
        soft_indices = jax.nn.one_hot(indices_k, x.shape[axis], axis=-1)
        return values, soft_indices

    # Methods that only return sorted values (no indices)
    if method in ("fast_soft_sort", "sorting_network"):
        sorted_vals = sort(
            x,
            axis=axis,
            descending=True,
            softness=softness,
            mode=mode,
            method=method,
            standardize=standardize,
            ot_kwargs=ot_kwargs,
        )
        values = jnp.take(sorted_vals, jnp.arange(k), axis=axis)
        return values, None

    # Methods that produce soft indices
    arg_method: ArgMethod = method  # type: ignore[assignment]
    soft_index = argsort(
        x,
        axis=axis,
        descending=True,
        softness=softness,
        mode=mode,
        method=arg_method,
        standardize=standardize,
        ot_kwargs=ot_kwargs,
    )
    if not gated_grad:
        soft_index = jax.lax.stop_gradient(soft_index)

    soft_index_k = jnp.take(soft_index, jnp.arange(k), axis=axis)
    values = take_along_axis(x, soft_index_k, axis=axis)
    return values, soft_index_k
