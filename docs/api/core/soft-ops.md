# Soft Operations

The `diffbio.core.soft_ops` module provides smooth, differentiable relaxations of discrete, piecewise-linear, and sharp operations. These relaxations enable end-to-end gradient-based optimization through operations that are normally non-differentiable -- comparisons, sorting, indexing, rounding, and logical gates all become continuous functions with well-defined gradients.

!!! note "Acknowledgment"
    The soft operations in this module are based on the algorithms and implementations from [SoftJAX](https://github.com/a-paulus/softjax) (Paulus et al., 2026; [arXiv:2603.08824](https://arxiv.org/abs/2603.08824)), adapted for the DiffBio/JAX/Flax NNX ecosystem.

All soft operations are JIT-compatible and support `jax.grad` and `jax.vmap`. Most accept a `softness` parameter controlling the width of the transition region (higher = smoother) and a `mode` parameter selecting the smoothness family:

| Mode | Description |
|------|-------------|
| `"hard"` | Exact (non-differentiable) version matching JAX |
| `"smooth"` | C-infinity smooth via logistic sigmoid |
| `"c0"` | Continuous (C0) via piecewise linear/quadratic |
| `"c1"` | Once differentiable (C1) via cubic Hermite polynomial |
| `"c2"` | Twice differentiable (C2) via quintic Hermite polynomial |

!!! warning "Name shadowing"
    Several names (`abs`, `all`, `any`, `round`, `max`, `min`) shadow Python builtins.
    Use qualified imports:

    ```python
    from diffbio.core import soft_ops
    soft_ops.max(x, softness=0.1)
    ```

    or alias on import:

    ```python
    from diffbio.core.soft_ops import max as soft_max
    ```

---

## Types

### SoftBool

::: diffbio.core.soft_ops._types.SoftBool
    options:
      show_root_heading: true
      show_source: false

### SoftIndex

::: diffbio.core.soft_ops._types.SoftIndex
    options:
      show_root_heading: true
      show_source: false

---

## Autograd-Safe Math

NaN-free alternatives to standard math functions. These use the double-where trick so that the forward pass computes correct values even at domain boundaries, while the backward pass produces finite (zero) gradients instead of NaN or Inf.

### sqrt

::: diffbio.core.soft_ops.autograd_safe.sqrt
    options:
      show_root_heading: true
      show_source: false

### arcsin

::: diffbio.core.soft_ops.autograd_safe.arcsin
    options:
      show_root_heading: true
      show_source: false

### arccos

::: diffbio.core.soft_ops.autograd_safe.arccos
    options:
      show_root_heading: true
      show_source: false

### div

::: diffbio.core.soft_ops.autograd_safe.div
    options:
      show_root_heading: true
      show_source: false

### log

::: diffbio.core.soft_ops.autograd_safe.log
    options:
      show_root_heading: true
      show_source: false

### norm

::: diffbio.core.soft_ops.autograd_safe.norm
    options:
      show_root_heading: true
      show_source: false

---

## Elementwise

Differentiable relaxations of elementwise non-smooth functions using sigmoidal smoothing with configurable smoothness modes.

### sigmoidal

::: diffbio.core.soft_ops.elementwise.sigmoidal
    options:
      show_root_heading: true
      show_source: false

### softrelu

::: diffbio.core.soft_ops.elementwise.softrelu
    options:
      show_root_heading: true
      show_source: false

### heaviside

::: diffbio.core.soft_ops.elementwise.heaviside
    options:
      show_root_heading: true
      show_source: false

### round

::: diffbio.core.soft_ops.elementwise.round
    options:
      show_root_heading: true
      show_source: false

### sign

::: diffbio.core.soft_ops.elementwise.sign
    options:
      show_root_heading: true
      show_source: false

### abs

::: diffbio.core.soft_ops.elementwise.abs
    options:
      show_root_heading: true
      show_source: false

### relu

::: diffbio.core.soft_ops.elementwise.relu
    options:
      show_root_heading: true
      show_source: false

### clip

::: diffbio.core.soft_ops.elementwise.clip
    options:
      show_root_heading: true
      show_source: false

---

## Comparison

Differentiable relaxations of elementwise comparison operations, returning `SoftBool` values in [0, 1]. Each function uses `sigmoidal` as the underlying smooth step function.

### greater

::: diffbio.core.soft_ops.comparison.greater
    options:
      show_root_heading: true
      show_source: false

### greater_equal

::: diffbio.core.soft_ops.comparison.greater_equal
    options:
      show_root_heading: true
      show_source: false

### less

::: diffbio.core.soft_ops.comparison.less
    options:
      show_root_heading: true
      show_source: false

### less_equal

::: diffbio.core.soft_ops.comparison.less_equal
    options:
      show_root_heading: true
      show_source: false

### equal

::: diffbio.core.soft_ops.comparison.equal
    options:
      show_root_heading: true
      show_source: false

### not_equal

::: diffbio.core.soft_ops.comparison.not_equal
    options:
      show_root_heading: true
      show_source: false

### isclose

::: diffbio.core.soft_ops.comparison.isclose
    options:
      show_root_heading: true
      show_source: false

---

## Logical

Differentiable fuzzy logic operations on `SoftBool` values. These operate purely on probability values in [0, 1] and do not take a `softness` parameter.

Fuzzy logic semantics:

- **NOT**: `1 - x`
- **AND** (product): `prod(x)` or geometric mean
- **OR**: `1 - AND(NOT(x))`
- **XOR**: `AND(x, NOT(y)) OR AND(NOT(x), y)`

### logical_not

::: diffbio.core.soft_ops.logical.logical_not
    options:
      show_root_heading: true
      show_source: false

### all

::: diffbio.core.soft_ops.logical.all
    options:
      show_root_heading: true
      show_source: false

### any

::: diffbio.core.soft_ops.logical.any
    options:
      show_root_heading: true
      show_source: false

### logical_and

::: diffbio.core.soft_ops.logical.logical_and
    options:
      show_root_heading: true
      show_source: false

### logical_or

::: diffbio.core.soft_ops.logical.logical_or
    options:
      show_root_heading: true
      show_source: false

### logical_xor

::: diffbio.core.soft_ops.logical.logical_xor
    options:
      show_root_heading: true
      show_source: false

---

## Selection

Differentiable relaxations of array selection and indexing operations. These use `SoftBool` conditions and `SoftIndex` probability distributions in place of discrete boolean masks and integer indices.

### where

::: diffbio.core.soft_ops.selection.where
    options:
      show_root_heading: true
      show_source: false

### take_along_axis

::: diffbio.core.soft_ops.selection.take_along_axis
    options:
      show_root_heading: true
      show_source: false

### take

::: diffbio.core.soft_ops.selection.take
    options:
      show_root_heading: true
      show_source: false

### choose

::: diffbio.core.soft_ops.selection.choose
    options:
      show_root_heading: true
      show_source: false

### dynamic_index_in_dim

::: diffbio.core.soft_ops.selection.dynamic_index_in_dim
    options:
      show_root_heading: true
      show_source: false

### dynamic_slice_in_dim

::: diffbio.core.soft_ops.selection.dynamic_slice_in_dim
    options:
      show_root_heading: true
      show_source: false

### dynamic_slice

::: diffbio.core.soft_ops.selection.dynamic_slice
    options:
      show_root_heading: true
      show_source: false

---

## Sorting

Differentiable relaxations of discrete ordering operations including argmax/argmin, argsort, sort, rank, and top-k. Multiple algorithmic backends are available:

| Method | Complexity | Default for |
|--------|-----------|-------------|
| `"softsort"` | O(n log n) | `argmax`, `argmin` |
| `"neuralsort"` | O(n^2) | `argsort`, `sort` |
| `"sorting_network"` | O(n log^2 n) | -- |
| `"ot"` | varies | -- |
| `"fast_soft_sort"` | O(n log n) | -- |
| `"smooth_sort"` | O(n log n) | -- |

The `ot`, `fast_soft_sort`, and `smooth_sort` methods require the `soft-ops-advanced` optional dependency group.

### argmax

::: diffbio.core.soft_ops.sorting.argmax
    options:
      show_root_heading: true
      show_source: false

### max

::: diffbio.core.soft_ops.sorting.max
    options:
      show_root_heading: true
      show_source: false

### argmin

::: diffbio.core.soft_ops.sorting.argmin
    options:
      show_root_heading: true
      show_source: false

### min

::: diffbio.core.soft_ops.sorting.min
    options:
      show_root_heading: true
      show_source: false

### argsort

::: diffbio.core.soft_ops.sorting.argsort
    options:
      show_root_heading: true
      show_source: false

### sort

::: diffbio.core.soft_ops.sorting.sort
    options:
      show_root_heading: true
      show_source: false

### rank

::: diffbio.core.soft_ops.sorting.rank
    options:
      show_root_heading: true
      show_source: false

### top_k

::: diffbio.core.soft_ops.sorting.top_k
    options:
      show_root_heading: true
      show_source: false

---

## Quantile

Differentiable relaxations of quantile-based statistics. Quantiles are computed via soft argsort or soft sort with interpolation following the same methods as `jax.numpy.quantile`.

### argquantile

::: diffbio.core.soft_ops.quantile.argquantile
    options:
      show_root_heading: true
      show_source: false

### quantile

::: diffbio.core.soft_ops.quantile.quantile
    options:
      show_root_heading: true
      show_source: false

### argmedian

::: diffbio.core.soft_ops.quantile.argmedian
    options:
      show_root_heading: true
      show_source: false

### median

::: diffbio.core.soft_ops.quantile.median
    options:
      show_root_heading: true
      show_source: false

### argpercentile

::: diffbio.core.soft_ops.quantile.argpercentile
    options:
      show_root_heading: true
      show_source: false

### percentile

::: diffbio.core.soft_ops.quantile.percentile
    options:
      show_root_heading: true
      show_source: false

---

## Straight-Through Estimators

Straight-through estimators use the hard (exact, non-differentiable) function for the forward pass but route gradients through the soft (differentiable) version during backpropagation. The trick `stop_gradient(hard - soft) + soft` ensures the forward output is exact while gradients flow through the smooth relaxation.

### st

::: diffbio.core.soft_ops.straight_through.st
    options:
      show_root_heading: true
      show_source: false

### grad_replace

::: diffbio.core.soft_ops.straight_through.grad_replace
    options:
      show_root_heading: true
      show_source: false

### Pre-built `_st` Variants

The module provides 27 pre-built straight-through variants, one for each soft operation that accepts a `mode` parameter. Each variant uses the hard function for the forward pass and the corresponding soft function for the backward pass.

| Elementwise | Comparison | Sorting | Quantile |
|-------------|------------|---------|----------|
| `abs_st` | `equal_st` | `argmax_st` | `argmedian_st` |
| `clip_st` | `greater_st` | `argmin_st` | `argpercentile_st` |
| `heaviside_st` | `greater_equal_st` | `argsort_st` | `argquantile_st` |
| `relu_st` | `isclose_st` | `max_st` | `median_st` |
| `round_st` | `less_st` | `min_st` | `percentile_st` |
| `sign_st` | `less_equal_st` | `rank_st` | `quantile_st` |
| | `not_equal_st` | `sort_st` | |
| | | `top_k_st` | |

Each `_st` variant accepts the same arguments as its base function. For example:

```python
from diffbio.core.soft_ops import relu_st, sort_st

# Forward uses jax.nn.relu; backward uses soft relu
y = relu_st(x, softness=0.1, mode="smooth")

# Forward uses jnp.sort; backward uses soft sort
y = sort_st(x, axis=-1, softness=0.1, mode="smooth")
```
