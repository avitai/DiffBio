"""Soft quantile, median, and percentile operators.

Provides differentiable relaxations of quantile-based statistics.
Quantiles are computed via :func:`~diffbio.core.soft_ops.sorting.argsort`
or :func:`~diffbio.core.soft_ops.sorting.sort`, with interpolation
following the same methods as ``jax.numpy.quantile``.
"""

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from diffbio.core.soft_ops._projections_simplex import proj_simplex
from diffbio.core.soft_ops._sorting_network import argsort_via_sorting_network
from diffbio.core.soft_ops._types import SoftIndex
from diffbio.core.soft_ops._utils import (
    canonicalize_axis,
    ensure_float,
    quantile_interpolation_params,
    standardize_and_squash,
)
from diffbio.core.soft_ops.selection import take_along_axis
from diffbio.core.soft_ops.sorting import (
    _neuralsort_a_sum,
)

Mode = Literal["hard", "smooth", "c0", "c1", "c2"]
ArgMethod = Literal["softsort", "neuralsort", "sorting_network"]


def argquantile(
    x: Array,
    q: Array,
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: ArgMethod = "neuralsort",
    quantile_method: Literal[
        "linear",
        "lower",
        "higher",
        "nearest",
        "midpoint",
    ] = "linear",
    standardize: bool = True,
) -> SoftIndex:
    """Soft argquantile returning SoftIndex.

    Args:
        x: Input array.
        q: Quantile(s) in [0, 1]. Scalar or 1-D array.
        axis: Axis along which to compute. None flattens.
        keepdims: If True, keep reduced dimension.
        softness: Controls sharpness (> 0).
        mode: Smoothness mode.
        method: Algorithm.
        quantile_method: Interpolation method.
        standardize: If True, standardize input.

    Returns:
        SoftIndex probability distribution over quantile position(s).
    """
    q_arr = jnp.asarray(q)
    if q_arr.ndim > 1:
        msg = f"q must be scalar or 1-D, got shape {q_arr.shape}"
        raise ValueError(msg)
    if q_arr.ndim == 1:

        def _single(qi: Array) -> SoftIndex:
            return argquantile(
                x,
                q=qi,
                axis=axis,
                keepdims=keepdims,
                softness=softness,
                mode=mode,
                method=method,
                quantile_method=quantile_method,
                standardize=standardize,
            )

        return jax.vmap(_single)(q_arr)

    orig_axis_is_none = axis is None
    if axis is None:
        num_dims = x.ndim
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = canonicalize_axis(axis, x.ndim)
        num_dims = None

    if mode != "hard":
        x = ensure_float(x)
    if standardize and mode != "hard":
        x = standardize_and_squash(x, axis=axis)

    x_last = jnp.moveaxis(x, axis, -1)
    *batch_dims, n = x_last.shape

    q_val = jnp.clip(q, 0.0, 1.0)
    k, a, take_next = quantile_interpolation_params(q_val, n, quantile_method)
    a_b = jnp.expand_dims(a, axis=-1)
    kp1 = jnp.minimum(k + 1, n - 1)

    if mode == "hard":
        indices = jnp.argsort(x_last, axis=-1, descending=False)
        if take_next:
            idx_pair = jnp.stack(
                [indices[..., k], indices[..., kp1]],
                axis=-1,
            )
            oh = jax.nn.one_hot(idx_pair, num_classes=n, axis=-1)
            soft_index = (1.0 - a_b) * oh[..., 0, :] + a_b * oh[..., 1, :]
        else:
            soft_index = jax.nn.one_hot(
                indices[..., k],
                num_classes=n,
                axis=-1,
            )
    elif method == "softsort":
        x_sorted = jnp.sort(x_last, axis=-1, descending=False)
        if take_next:
            anchors = jnp.stack(
                [x_sorted[..., k], x_sorted[..., kp1]],
                axis=-1,
            )
            abs_diff = jnp.abs(
                anchors[..., :, None] - x_last[..., None, :],
            )
            proj = proj_simplex(-abs_diff, axis=-1, softness=softness, mode=mode)
            soft_index = (1.0 - a_b) * proj[..., 0, :] + a_b * proj[..., 1, :]
        else:
            anchors = x_sorted[..., k, None]
            abs_diff = jnp.abs(
                anchors[..., :, None] - x_last[..., None, :],
            )
            soft_index = proj_simplex(
                -abs_diff,
                axis=-1,
                softness=softness,
                mode=mode,
            )[..., 0, :]
    elif method == "neuralsort":
        a_sum = _neuralsort_a_sum(x_last, mode=mode, softness=softness)
        if take_next:
            i = jnp.array([k + 1, k + 2])
            coef = n + 1 - 2 * i
            coef = jnp.broadcast_to(coef, (*batch_dims, 2))
            z = -(coef[..., :, None] * x_last[..., None, :] + a_sum[..., None, :])
            proj = proj_simplex(z, axis=-1, softness=softness, mode=mode)
            soft_index = (1.0 - a_b) * proj[..., 0, :] + a_b * proj[..., 1, :]
        else:
            coef = jnp.array([n + 1 - 2 * (k + 1)])
            coef = jnp.broadcast_to(coef, (*batch_dims, 1))
            z = -(coef[..., :, None] * x_last[..., None, :] + a_sum[..., None, :])
            soft_index = proj_simplex(
                z,
                axis=-1,
                softness=softness,
                mode=mode,
            )[..., 0, :]
    elif method == "sorting_network":
        perm = argsort_via_sorting_network(
            x_last,
            softness,
            mode,
            descending=False,
            standardized=standardize,
        )
        if take_next:
            soft_index = (1.0 - a_b) * perm[..., k, :] + a_b * perm[..., kp1, :]
        else:
            soft_index = perm[..., k, :]
    else:
        msg = f"Invalid method: {method!r}"
        raise ValueError(msg)

    if keepdims:
        if orig_axis_is_none:
            soft_index = soft_index.reshape(*(1,) * num_dims, n)
        else:
            soft_index = jnp.expand_dims(soft_index, axis=axis)

    return soft_index


def quantile(
    x: Array,
    q: Array,
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: str = "neuralsort",
    quantile_method: Literal[
        "linear",
        "lower",
        "higher",
        "nearest",
        "midpoint",
    ] = "linear",
    standardize: bool = True,
    gated_grad: bool = True,
) -> Array:
    """Soft quantile returning value.

    Implemented as :func:`argquantile` + :func:`take_along_axis`
    for most methods.

    Args:
        x: Input array.
        q: Quantile(s) in [0, 1].
        axis: Axis along which to compute.
        keepdims: If True, keep reduced dimension.
        softness: Controls sharpness (> 0).
        mode: Smoothness mode.
        method: Algorithm.
        quantile_method: Interpolation method.
        standardize: If True, standardize input.
        gated_grad: If False, stop gradient through soft index.

    Returns:
        Quantile value(s).
    """
    if mode == "hard":
        return jnp.quantile(x, q, axis=axis, keepdims=keepdims, method=quantile_method)

    soft_idx = argquantile(
        x,
        q,
        axis=axis,
        keepdims=True,
        softness=softness,
        mode=mode,
        method=method,
        quantile_method=quantile_method,
        standardize=standardize,
    )
    if not gated_grad:
        soft_idx = jax.lax.stop_gradient(soft_idx)

    _axis = 0 if axis is None else canonicalize_axis(axis, x.ndim)
    if axis is None:
        x = jnp.ravel(x)

    q_arr = jnp.asarray(q)
    if q_arr.ndim == 1:
        result = jax.vmap(lambda si: take_along_axis(x, si, axis=_axis))(soft_idx)
    else:
        result = take_along_axis(x, soft_idx, axis=_axis)

    if not keepdims:
        if q_arr.ndim == 0:
            result = jnp.squeeze(result, axis=_axis if axis is not None else 0)
    return result


def argmedian(
    x: Array,
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: ArgMethod = "neuralsort",
    standardize: bool = True,
) -> SoftIndex:
    """Soft argmedian: :func:`argquantile` with ``q=0.5``."""
    return argquantile(
        x,
        q=jnp.array(0.5),
        axis=axis,
        keepdims=keepdims,
        softness=softness,
        mode=mode,
        method=method,
        standardize=standardize,
    )


def median(
    x: Array,
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: str = "neuralsort",
    standardize: bool = True,
    gated_grad: bool = True,
) -> Array:
    """Soft median: :func:`quantile` with ``q=0.5``."""
    return quantile(
        x,
        q=jnp.array(0.5),
        axis=axis,
        keepdims=keepdims,
        softness=softness,
        mode=mode,
        method=method,
        standardize=standardize,
        gated_grad=gated_grad,
    )


def argpercentile(
    x: Array,
    p: Array,
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: ArgMethod = "neuralsort",
    standardize: bool = True,
) -> SoftIndex:
    """Soft argpercentile: :func:`argquantile` with ``q = p / 100``."""
    return argquantile(
        x,
        q=jnp.asarray(p) / 100.0,
        axis=axis,
        keepdims=keepdims,
        softness=softness,
        mode=mode,
        method=method,
        standardize=standardize,
    )


def percentile(
    x: Array,
    p: Array,
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    method: str = "neuralsort",
    standardize: bool = True,
    gated_grad: bool = True,
) -> Array:
    """Soft percentile: :func:`quantile` with ``q = p / 100``."""
    return quantile(
        x,
        q=jnp.asarray(p) / 100.0,
        axis=axis,
        keepdims=keepdims,
        softness=softness,
        mode=mode,
        method=method,
        standardize=standardize,
        gated_grad=gated_grad,
    )
