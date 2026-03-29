"""Differentiable soft operations for DiffBio.

Provides smooth relaxations of discrete, piecewise-linear, and sharp
operations, enabling end-to-end gradient-based optimization through
operations that are normally non-differentiable.

Built on JAX and integrated with the DiffBio/Flax NNX ecosystem.

.. warning::
    Several names (``abs``, ``all``, ``any``, ``round``, ``max``, ``min``)
    shadow Python builtins. Use qualified imports::

        from diffbio.core import soft_ops
        soft_ops.max(x, softness=0.1)

    or alias on import::

        from diffbio.core.soft_ops import max as soft_max
"""

# --- Types ---
from diffbio.core.soft_ops._types import SoftBool, SoftIndex

# --- Autograd-safe math ---
from diffbio.core.soft_ops.autograd_safe import arccos, arcsin, div, log, norm, sqrt

# --- Elementwise ---
from diffbio.core.soft_ops.elementwise import (
    abs,
    clip,
    heaviside,
    relu,
    round,
    sign,
    sigmoidal,
    softrelu,
)

# --- Comparison ---
from diffbio.core.soft_ops.comparison import (
    equal,
    greater,
    greater_equal,
    isclose,
    less,
    less_equal,
    not_equal,
)

# --- Logical ---
from diffbio.core.soft_ops.logical import (
    all,
    any,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
)

# --- Selection ---
from diffbio.core.soft_ops.selection import (
    choose,
    dynamic_index_in_dim,
    dynamic_slice,
    dynamic_slice_in_dim,
    take,
    take_along_axis,
    where,
)

# --- Sorting / Ordering ---
from diffbio.core.soft_ops.sorting import (
    argmax,
    argmin,
    argsort,
    max,
    min,
    rank,
    sort,
    top_k,
)

# --- Quantile ---
from diffbio.core.soft_ops.quantile import (
    argmedian,
    argpercentile,
    argquantile,
    median,
    percentile,
    quantile,
)

# --- Straight-through estimators ---
from diffbio.core.soft_ops.straight_through import (
    abs_st,
    argmax_st,
    argmedian_st,
    argmin_st,
    argpercentile_st,
    argquantile_st,
    argsort_st,
    clip_st,
    equal_st,
    grad_replace,
    greater_equal_st,
    greater_st,
    heaviside_st,
    isclose_st,
    less_equal_st,
    less_st,
    max_st,
    median_st,
    min_st,
    not_equal_st,
    percentile_st,
    quantile_st,
    rank_st,
    relu_st,
    round_st,
    sign_st,
    sort_st,
    st,
    top_k_st,
)

__all__ = [
    # Types
    "SoftBool",
    "SoftIndex",
    # Autograd-safe
    "arccos",
    "arcsin",
    "div",
    "log",
    "norm",
    "sqrt",
    # Elementwise
    "abs",
    "clip",
    "heaviside",
    "relu",
    "round",
    "sign",
    "sigmoidal",
    "softrelu",
    # Comparison
    "equal",
    "greater",
    "greater_equal",
    "isclose",
    "less",
    "less_equal",
    "not_equal",
    # Logical
    "all",
    "any",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    # Selection
    "choose",
    "dynamic_index_in_dim",
    "dynamic_slice",
    "dynamic_slice_in_dim",
    "take",
    "take_along_axis",
    "where",
    # Sorting
    "argmax",
    "argmin",
    "argsort",
    "max",
    "min",
    "rank",
    "sort",
    "top_k",
    # Quantile
    "argmedian",
    "argpercentile",
    "argquantile",
    "median",
    "percentile",
    "quantile",
    # Straight-through decorators
    "grad_replace",
    "st",
    # Straight-through variants
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
