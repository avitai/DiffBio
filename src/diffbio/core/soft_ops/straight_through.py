"""Straight-through estimator decorators and _st variants.

Straight-through estimators use the hard (exact, non-differentiable)
function for the forward pass but route gradients through the soft
(differentiable) version during backpropagation.

This module provides:
- :func:`st`: Decorator that creates a straight-through version of any
  soft_ops function with a ``mode`` parameter.
- :func:`grad_replace`: Lower-level decorator for custom forward/backward
  split functions.
- 27 pre-built ``_st`` variants (e.g., :func:`relu_st`, :func:`sort_st`).
"""

import functools
import inspect
from collections.abc import Callable

import jax
from jax import tree_util as jtu

from diffbio.core.soft_ops import comparison, elementwise, quantile, sorting


def grad_replace(fn: Callable) -> Callable:
    """Decorator for custom forward/backward computation split.

    The decorated function is called twice: once with ``forward=True``
    (output used for forward pass) and once with ``forward=False``
    (output used for gradient computation).

    Args:
        fn: Function accepting a ``forward: bool`` keyword argument.

    Returns:
        Wrapped function using hard forward, soft backward.
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        fw_y = fn(*args, **kwargs, forward=True)
        bw_y = fn(*args, **kwargs, forward=False)
        fw_leaves, fw_treedef = jtu.tree_flatten(
            fw_y,
            is_leaf=lambda x: x is None,
        )
        bw_leaves, _ = jtu.tree_flatten(
            bw_y,
            is_leaf=lambda x: x is None,
        )
        out_leaves = [
            f if f is None or b is None else jax.lax.stop_gradient(f - b) + b
            for f, b in zip(fw_leaves, bw_leaves)
        ]
        return jtu.tree_unflatten(fw_treedef, out_leaves)

    return wrapped


def st(fn: Callable) -> Callable:
    """Decorator creating a straight-through estimator from a soft_ops function.

    The decorated function is called twice: once with ``mode="hard"``
    (forward pass) and once with the specified ``mode`` (backward pass).
    The trick ``stop_gradient(hard - soft) + soft`` ensures the forward
    output is hard but gradients flow through the soft version.

    Args:
        fn: Function with a ``mode`` parameter (e.g., any elementwise,
            comparison, or sorting function).

    Returns:
        Wrapped straight-through estimator function.
    """
    sig = inspect.signature(fn)
    mode_param = sig.parameters.get("mode")
    if mode_param is not None:
        mode_default = mode_param.default
        mode_idx = list(sig.parameters.keys()).index("mode")
    else:
        mode_default = "smooth"
        mode_idx = None

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if mode_idx is not None and len(args) > mode_idx:
            mode = args[mode_idx]
            args = args[:mode_idx] + args[mode_idx + 1 :]
        else:
            mode = kwargs.pop("mode", mode_default)
        fw_y = fn(*args, **kwargs, mode="hard")
        bw_y = fn(*args, **kwargs, mode=mode)
        fw_leaves, fw_treedef = jtu.tree_flatten(
            fw_y,
            is_leaf=lambda x: x is None,
        )
        bw_leaves, _ = jtu.tree_flatten(
            bw_y,
            is_leaf=lambda x: x is None,
        )
        out_leaves = [
            f if f is None or b is None else jax.lax.stop_gradient(f - b) + b
            for f, b in zip(fw_leaves, bw_leaves)
        ]
        return jtu.tree_unflatten(fw_treedef, out_leaves)

    return wrapped


# ---------------------------------------------------------------------------
# Cached ST wrapper to avoid repeated inspect.signature() calls
# ---------------------------------------------------------------------------

_st_cache: dict[Callable, Callable] = {}


def _cached_st(fn: Callable) -> Callable:
    """Return a cached st() wrapper."""
    if fn not in _st_cache:
        _st_cache[fn] = st(fn)
    return _st_cache[fn]


# ---------------------------------------------------------------------------
# 27 pre-built straight-through variants
# ---------------------------------------------------------------------------


def abs_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.elementwise.abs`."""
    return _cached_st(elementwise.abs)(*args, **kwargs)


def argmax_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.sorting.argmax`."""
    return _cached_st(sorting.argmax)(*args, **kwargs)


def argmedian_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.quantile.argmedian`."""
    return _cached_st(quantile.argmedian)(*args, **kwargs)


def argmin_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.sorting.argmin`."""
    return _cached_st(sorting.argmin)(*args, **kwargs)


def argpercentile_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.quantile.argpercentile`."""
    return _cached_st(quantile.argpercentile)(*args, **kwargs)


def argquantile_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.quantile.argquantile`."""
    return _cached_st(quantile.argquantile)(*args, **kwargs)


def argsort_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.sorting.argsort`."""
    return _cached_st(sorting.argsort)(*args, **kwargs)


def clip_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.elementwise.clip`."""
    return _cached_st(elementwise.clip)(*args, **kwargs)


def equal_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.comparison.equal`."""
    return _cached_st(comparison.equal)(*args, **kwargs)


def greater_equal_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.comparison.greater_equal`."""
    return _cached_st(comparison.greater_equal)(*args, **kwargs)


def greater_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.comparison.greater`."""
    return _cached_st(comparison.greater)(*args, **kwargs)


def heaviside_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.elementwise.heaviside`."""
    return _cached_st(elementwise.heaviside)(*args, **kwargs)


def isclose_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.comparison.isclose`."""
    return _cached_st(comparison.isclose)(*args, **kwargs)


def less_equal_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.comparison.less_equal`."""
    return _cached_st(comparison.less_equal)(*args, **kwargs)


def less_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.comparison.less`."""
    return _cached_st(comparison.less)(*args, **kwargs)


def max_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.sorting.max`."""
    return _cached_st(sorting.max)(*args, **kwargs)


def median_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.quantile.median`."""
    return _cached_st(quantile.median)(*args, **kwargs)


def min_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.sorting.min`."""
    return _cached_st(sorting.min)(*args, **kwargs)


def not_equal_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.comparison.not_equal`."""
    return _cached_st(comparison.not_equal)(*args, **kwargs)


def percentile_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.quantile.percentile`."""
    return _cached_st(quantile.percentile)(*args, **kwargs)


def quantile_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.quantile.quantile`."""
    return _cached_st(quantile.quantile)(*args, **kwargs)


def rank_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.sorting.rank`."""
    return _cached_st(sorting.rank)(*args, **kwargs)


def relu_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.elementwise.relu`."""
    return _cached_st(elementwise.relu)(*args, **kwargs)


def round_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.elementwise.round`."""
    return _cached_st(elementwise.round)(*args, **kwargs)


def sign_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.elementwise.sign`."""
    return _cached_st(elementwise.sign)(*args, **kwargs)


def sort_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.sorting.sort`."""
    return _cached_st(sorting.sort)(*args, **kwargs)


def top_k_st(*args, **kwargs):
    """Straight-through :func:`~diffbio.core.soft_ops.sorting.top_k`."""
    return _cached_st(sorting.top_k)(*args, **kwargs)
