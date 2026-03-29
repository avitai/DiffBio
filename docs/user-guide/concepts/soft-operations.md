# Soft Operations

Soft operations are differentiable relaxations of discrete, piecewise-linear, and
sharp mathematical functions. They form the computational backbone of DiffBio,
enabling gradient-based optimization through operations -- comparisons, sorting,
selection, logical gates -- that are normally non-differentiable.

---

## Why Soft Operations Matter

Bioinformatics pipelines are built from discrete decisions: "is this quality score
above 20?", "which read has the highest alignment score?", "select the top 10
candidate variants." Each of these operations has zero or undefined gradients
almost everywhere, which blocks end-to-end training with gradient descent.

Soft operations replace each hard decision with a smooth approximation that:

1. **Returns the same result** in the limit as the smoothing decreases.
2. **Produces well-defined gradients** everywhere, allowing upstream parameters
   to learn from downstream losses.
3. **Preserves semantic meaning** -- a soft comparison still answers "is x greater
   than y?", but as a probability rather than a binary flag.

All soft operations live in the `diffbio.core.soft_ops` module and are designed
for use under `jax.jit`, `jax.grad`, and `jax.vmap`.

The algorithms are based on the work of
[Paulus et al. (2026)](https://arxiv.org/abs/2603.08824)
and the [SoftJAX](https://github.com/a-paulus/softjax) library,
adapted for DiffBio's bioinformatics operators and Flax NNX integration.

```python
from diffbio.core import soft_ops
```

---

## Smoothness Modes

Every soft operation accepts a `mode` parameter that controls the mathematical
class of the smoothing function. The five modes trade off computation cost against
the regularity of the resulting function:

| Mode | Continuity Class | Description | Implementation |
|---|---|---|---|
| `"hard"` | Discontinuous | Exact non-differentiable operation (matches JAX/NumPy) | `jnp.greater`, `jnp.sort`, etc. |
| `"smooth"` | C-infinity | Infinitely differentiable via logistic sigmoid | `jax.nn.sigmoid(x / softness)` |
| `"c0"` | C0 (continuous) | Piecewise-linear interpolation through transition region | Linear polynomial |
| `"c1"` | C1 (differentiable) | Once-differentiable via cubic Hermite polynomial | Cubic polynomial |
| `"c2"` | C2 (twice differentiable) | Twice-differentiable via quintic Hermite polynomial | Quintic polynomial |

### When to Use Each Mode

- **`"hard"`** -- Evaluation and inference. No gradients needed. Produces exact
  results identical to standard JAX functions.
- **`"smooth"`** (default) -- General training. The logistic sigmoid is
  C-infinity smooth, meaning all higher-order derivatives exist. This is the
  safest choice for gradient-based optimization and works well with second-order
  optimizers (L-BFGS, natural gradient).
- **`"c0"`** -- When you need continuous output but want the cheapest possible
  smooth approximation. Gradients exist almost everywhere but have discontinuous
  jumps at transition boundaries.
- **`"c1"`** -- When first-order optimizers (Adam, SGD) require smooth gradients
  without jumps. Slightly more expensive than `"c0"` but eliminates gradient
  discontinuities.
- **`"c2"`** -- When second-order methods or Hessian-vector products are needed
  and you want to avoid the global support of the logistic sigmoid. The quintic
  polynomial has compact support (gradients are exactly zero outside the
  transition region) while still being twice differentiable.

### The Sigmoidal Foundation

All soft operations are built on a single primitive, the `sigmoidal` function,
which maps the real line to (0, 1) through an S-curve:

```python
from diffbio.core.soft_ops import sigmoidal

# C-infinity smooth (logistic sigmoid)
y = sigmoidal(x, softness=0.1, mode="smooth")

# C1 differentiable (cubic Hermite)
y = sigmoidal(x, softness=0.1, mode="c1")
```

For `"smooth"` mode, this is the standard logistic sigmoid
$\sigma(x / s)$ where $s$ is the softness. For piecewise modes (`"c0"`,
`"c1"`, `"c2"`), the transition region spans $[-5s, 5s]$, matching the
effective range of the logistic sigmoid.

---

## The `softness` Parameter

The `softness` parameter (equivalent to `temperature` in DiffBio operators)
controls the width of the transition region between the two sides of a sharp
operation. It governs the fundamental trade-off between accuracy and
differentiability:

- **`softness` approaching 0**: The soft operation converges to the exact hard
  operation. Outputs are precise but gradients become very small (vanishing
  gradient problem).
- **`softness` approaching infinity**: The output becomes maximally smooth but
  loses discriminative power. Everything blurs toward the midpoint.
- **Practical range (0.01--1.0)**: Useful gradients flow while outputs remain
  close to exact values. The default `softness=0.1` is a reasonable starting
  point for most operations.

```python
import jax.numpy as jnp
from diffbio.core import soft_ops

x = jnp.array([0.5, 1.5, 2.5])
threshold = 1.0

# Tight approximation -- almost binary
soft_ops.greater(x, threshold, softness=0.01)
# Array([~0.0, ~1.0, ~1.0])

# Moderate smoothing -- good gradient flow
soft_ops.greater(x, threshold, softness=0.1)
# Array([~0.007, ~0.993, ~1.0])

# Heavy smoothing -- gradients everywhere but less decisive
soft_ops.greater(x, threshold, softness=1.0)
# Array([~0.38, ~0.62, ~0.82])
```

### Softness vs. Temperature

In DiffBio's operator system, the `temperature` config field maps directly to
`softness` in soft_ops. The `TemperatureOperator` base class stores this value
and exposes convenience methods (`soft_max`, `soft_argmax`) that pass it through
automatically. When writing custom operators, you typically set the temperature
once in the config and let the base class handle the plumbing.

---

## Key Types

Soft operations introduce two semantic type aliases that distinguish soft outputs
from regular arrays:

### SoftBool

A `SoftBool` is a JAX array with values in [0, 1], representing a probability
of truth. It is the return type of all comparison and elementwise threshold
operations.

```python
from diffbio.core.soft_ops import SoftBool

# A soft comparison returns SoftBool
mask: SoftBool = soft_ops.greater(quality_scores, 20.0, softness=0.1)
# Values near 0.0 mean "probably false", near 1.0 means "probably true"
```

SoftBool values compose naturally with fuzzy logic operations (`logical_and`,
`logical_or`, `logical_not`) and drive soft conditional branching via `where`.

### SoftIndex

A `SoftIndex` is a JAX array whose last dimension is a probability distribution
over indices (values sum to 1 along the last axis). It is the return type of
`argmax`, `argmin`, `argsort`, and related ordering operations.

```python
from diffbio.core.soft_ops import SoftIndex

# Soft argmax returns a probability distribution over positions
idx: SoftIndex = soft_ops.argmax(scores, axis=-1, softness=0.1)
# Shape: same as scores but last dim is a probability distribution
# idx[..., i] is the probability that position i holds the maximum
```

SoftIndex values drive soft indexing via `take_along_axis`, `take`, `choose`,
and `dynamic_slice_in_dim`. Where a hard index selects one element, a SoftIndex
computes a weighted combination.

---

## Categories of Operations

The `soft_ops` module organizes operations into six categories. Each function
accepts `softness` and `mode` parameters unless otherwise noted.

### Elementwise

Smooth relaxations of per-element non-smooth functions.

| Function | Hard Equivalent | Notes |
|---|---|---|
| `abs` | `jnp.abs` | Via `x * sign(x)` |
| `sign` | `jnp.sign` | Maps to [-1, 1] |
| `relu` | `jax.nn.relu` | Optional gated variant (SiLU-style) |
| `clip` | `jnp.clip` | Soft clamping to [a, b] |
| `round` | `jnp.round` | Weighted sum over neighboring integers |
| `heaviside` | `jnp.heaviside` | Step function, returns SoftBool |
| `sigmoidal` | -- | Foundation S-curve, not a relaxation of a specific JAX op |
| `softrelu` | -- | Antiderivative of sigmoidal; smooth ReLU family |

**Bioinformatics use case:** Soft `clip` enforces that predicted quality scores
stay within the valid Phred range [0, 60] while keeping the operation
differentiable.

### Comparison

Elementwise comparisons returning SoftBool values.

| Function | Hard Equivalent |
|---|---|
| `greater` | `jnp.greater` |
| `greater_equal` | `jnp.greater_equal` |
| `less` | `jnp.less` |
| `less_equal` | `jnp.less_equal` |
| `equal` | `jnp.equal` |
| `not_equal` | `jnp.not_equal` |
| `isclose` | `jnp.isclose` |

**Bioinformatics use case:** Soft `greater` filters sequencing reads by quality
score, producing a continuous mask instead of a binary one.

### Logical

Fuzzy logic on SoftBool inputs. These operate purely on probability values and
do not take a `softness` parameter.

| Function | Hard Equivalent | Semantics |
|---|---|---|
| `logical_not` | `~x` | `1 - x` |
| `logical_and` | `x & y` | Product `x * y` |
| `logical_or` | `x \| y` | `1 - (1-x)(1-y)` |
| `logical_xor` | `x ^ y` | `and(x, not y) or and(not x, y)` |
| `all` | `jnp.all` | Product reduction along axis |
| `any` | `jnp.any` | `1 - prod(1 - x)` along axis |

**Bioinformatics use case:** Combining multiple soft filters -- "quality above 20
AND mapping quality above 30" -- as `logical_and(q_mask, mq_mask)`.

### Selection

Soft indexing and conditional branching using SoftBool and SoftIndex.

| Function | Hard Equivalent | Notes |
|---|---|---|
| `where` | `jnp.where` | Interpolates: `x * cond + y * (1 - cond)` |
| `take_along_axis` | `jnp.take_along_axis` | Weighted dot product with SoftIndex |
| `take` | `jnp.take` | 2-D SoftIndex applied across batch dims |
| `choose` | `jnp.choose` | Select among multiple arrays |
| `dynamic_index_in_dim` | `lax.dynamic_index_in_dim` | Single soft index along one axis |
| `dynamic_slice_in_dim` | `lax.dynamic_slice_in_dim` | Soft windowed extraction |
| `dynamic_slice` | `lax.dynamic_slice` | Multi-axis soft slicing |

**Bioinformatics use case:** Soft `dynamic_slice_in_dim` extracts a window around
a candidate variant position, where the position itself is a soft index learned
during training.

### Sorting and Ordering

Differentiable sorting, ranking, argmax/argmin, and top-k selection. These are the
most algorithmically complex operations and support multiple solving methods (see
[Sorting Methods](#sorting-methods) below).

| Function | Hard Equivalent | Return Type |
|---|---|---|
| `sort` | `jnp.sort` | Sorted values |
| `argsort` | `jnp.argsort` | Soft permutation matrix (SoftIndex) |
| `argmax` | `jnp.argmax` | SoftIndex (probability over positions) |
| `argmin` | `jnp.argmin` | SoftIndex |
| `max` | `jnp.max` | Scalar or reduced array |
| `min` | `jnp.min` | Scalar or reduced array |
| `rank` | scipy.stats.rankdata | Continuous ranks in [1, n] |
| `top_k` | `jax.lax.top_k` | Tuple of (values, SoftIndex) |

**Bioinformatics use case:** Soft `rank` computes differentiable gene expression
rankings for differential expression analysis. Soft `top_k` selects the k most
significant variants while maintaining gradient flow.

### Quantile

Differentiable quantile, median, and percentile computation.

| Function | Hard Equivalent | Return Type |
|---|---|---|
| `quantile` | `jnp.quantile` | Quantile values |
| `median` | `jnp.median` | Median values |
| `percentile` | `jnp.percentile` | Percentile values |
| `argquantile` | -- | SoftIndex at quantile position |
| `argmedian` | -- | SoftIndex at median position |
| `argpercentile` | -- | SoftIndex at percentile position |

**Bioinformatics use case:** Soft `median` computes a differentiable median
expression level for normalizing single-cell count matrices.

### Autograd-Safe Math

NaN-free alternatives to standard math functions using the double-where trick.
These are not soft approximations but rather numerically safe versions that
produce finite gradients at domain boundaries (e.g., `sqrt(0)`, `log(0)`,
`arcsin(1)`).

| Function | Protects Against |
|---|---|
| `sqrt` | NaN gradient at x = 0 |
| `log` | NaN at x = 0 |
| `div` | NaN at denominator = 0 |
| `norm` | NaN gradient at zero vector |
| `arcsin` | NaN gradient at x = +/-1 |
| `arccos` | NaN gradient at x = +/-1 |

---

## Sorting Methods

Sorting and ordering operations support multiple algorithmic backends, selected
via the `method` parameter. The choice affects computational complexity, memory
usage, and gradient quality.

### Default Methods

These are available without additional dependencies:

| Method | Complexity | Default For | Description |
|---|---|---|---|
| `"softsort"` | O(n log n) | `argmax`, `argmin` | Sorts input, projects distances onto simplex. Fast and memory-efficient for argmax/argmin where only one row of the permutation matrix is needed. |
| `"neuralsort"` | O(n^2) | `argsort`, `sort`, `rank` | Computes pairwise absolute differences, then projects onto simplex. Produces higher-quality soft permutation matrices at quadratic cost. |
| `"sorting_network"` | O(n log^2 n) | -- | Bitonic sorting network with soft comparators. Deterministic structure, good for moderate n. |

### Advanced Methods (Optional)

These require the `soft-ops-advanced` optional dependency group:

```bash
uv pip install -e '.[soft-ops-advanced]'
```

| Method | Complexity | Description |
|---|---|---|
| `"fast_soft_sort"` | O(n log n) | Permutahedron projection via Pool Adjacent Violators (PAV). Produces sorted values directly without constructing a permutation matrix. |
| `"smooth_sort"` | O(n log n) | Smooth permutahedron projection using ESP bounds. Only supports `mode="smooth"`. |
| `"ot"` | O(n^2) | Optimal transport projection onto the Birkhoff polytope. Produces doubly-stochastic permutation matrices. |

### Choosing a Method

- For **argmax/argmin** on large arrays: use `"softsort"` (default). It only
  computes one row of the permutation matrix in O(n log n) time.
- For **full sort/argsort** when permutation quality matters: use `"neuralsort"`
  (default). The O(n^2) cost is acceptable for arrays up to a few thousand
  elements.
- For **sort on very large arrays** (n > 10,000): consider `"fast_soft_sort"` or
  `"sorting_network"` to avoid the quadratic cost.
- For **doubly-stochastic permutation matrices**: use `"ot"`. This is the only
  method that guarantees both row and column sums equal 1.

---

## Straight-Through Estimators

Sometimes you want the **exact** hard output during the forward pass but still
need gradients to flow backward. Straight-through estimators (STE) achieve this
by computing:

```
output = stop_gradient(hard - soft) + soft
```

In the forward pass, `output == hard` (the stop_gradient kills the soft term's
contribution). In the backward pass, gradients flow through `soft` only (the
stop_gradient kills the hard term's gradient).

### The `st` Decorator

The `st` decorator wraps any soft_ops function that has a `mode` parameter:

```python
from diffbio.core.soft_ops import st, sort

sort_st = st(sort)

# Forward pass uses jnp.sort (hard), backward uses soft sort
values = sort_st(x, axis=-1, softness=0.1, mode="smooth")
```

### Pre-Built STE Variants

The module provides 27 pre-built `_st` variants for convenience:

```python
from diffbio.core import soft_ops

# Hard forward, soft backward for each operation
soft_ops.sort_st(x, axis=-1, softness=0.1)
soft_ops.argmax_st(x, axis=-1, softness=0.1)
soft_ops.greater_st(x, y, softness=0.1)
soft_ops.relu_st(x, softness=0.1)
soft_ops.rank_st(x, axis=0, softness=0.1)
soft_ops.top_k_st(x, k=10, softness=0.1)
# ... and 21 more
```

### The `grad_replace` Decorator

For more control, `grad_replace` lets you define separate forward and backward
functions explicitly:

```python
from diffbio.core.soft_ops import grad_replace

@grad_replace
def custom_threshold(x, t, forward: bool = True):
    if forward:
        return (x > t).astype(float)  # Hard threshold
    return jax.nn.sigmoid((x - t) / 0.1)  # Soft gradient source
```

### When to Use STEs

- **Training with hard outputs**: When downstream logic requires binary masks or
  integer indices but you still need gradients for upstream parameters.
- **Avoiding soft approximation bias**: Soft sort introduces small ordering errors;
  STE sort is exact in the forward pass.
- **Mixed-precision pipelines**: Use hard operations for numerical stability in
  the forward pass while relying on soft gradients for learning.

---

## Usage Examples

### Soft Quality Filtering

Filter sequencing reads by quality score with a smooth threshold:

```python
import jax.numpy as jnp
from diffbio.core import soft_ops

quality_scores = jnp.array([15.0, 22.0, 18.0, 30.0, 12.0])
threshold = 20.0

# Soft mask: values near 1.0 for reads above threshold
mask = soft_ops.greater(quality_scores, threshold, softness=0.5)

# Apply mask to read weights (keeps gradient flow)
weighted_reads = reads * mask[..., None]
```

### Soft Gene Ranking

Rank genes by expression level with differentiable ranking:

```python
import jax.numpy as jnp
from diffbio.core import soft_ops

# Expression matrix: (num_genes, num_samples)
expression = jnp.array([[2.1, 0.5, 3.8, 1.2],
                         [4.0, 1.1, 0.3, 2.7]])

# Continuous ranks along the gene axis (axis=0)
# Rank 1 = lowest expression, rank n = highest
ranks = soft_ops.rank(expression, axis=0, softness=0.1)
```

### Soft Top-k Selection

Select the k highest-scoring candidate variants:

```python
import jax.numpy as jnp
from diffbio.core import soft_ops

variant_scores = jnp.array([0.2, 0.9, 0.1, 0.85, 0.3, 0.95])

# Select top 3 variants with differentiable selection
values, soft_indices = soft_ops.top_k(variant_scores, k=3, softness=0.1)
# values: soft-sorted top-3 scores (descending)
# soft_indices: (3, 6) SoftIndex matrix -- each row is a probability
#   distribution over the 6 original positions
```

### Conditional Logic

Implement soft conditional branching for differentiable data transformations:

```python
import jax.numpy as jnp
from diffbio.core import soft_ops

x = jnp.array([-1.0, 0.5, -0.3, 2.0, -0.1])

# Soft ReLU via where: keep positive values, zero out negatives
# Equivalent to soft_ops.relu(x) but demonstrates composability
condition = soft_ops.greater(x, 0.0, softness=0.1)
result = soft_ops.where(condition, x, jnp.zeros_like(x))
```

### Combining Soft Filters

Chain multiple soft conditions with fuzzy logic:

```python
import jax.numpy as jnp
from diffbio.core import soft_ops

quality = jnp.array([25.0, 15.0, 30.0, 10.0])
mapping_quality = jnp.array([40.0, 50.0, 20.0, 60.0])

# Soft masks for each criterion
q_pass = soft_ops.greater(quality, 20.0, softness=0.5)
mq_pass = soft_ops.greater(mapping_quality, 30.0, softness=0.5)

# Combined filter: both conditions must be satisfied
combined = soft_ops.logical_and(q_pass, mq_pass)
# Result: high probability only for reads passing both thresholds
```

### Differentiable Median Normalization

Normalize a count matrix by its soft median:

```python
import jax.numpy as jnp
from diffbio.core import soft_ops

# Single-cell count matrix: (num_cells, num_genes)
counts = jnp.array([[100.0, 200.0, 50.0],
                     [150.0, 80.0, 300.0]])

# Differentiable median per cell
cell_medians = soft_ops.median(counts, axis=-1, softness=0.1)

# Normalize each cell by its median
normalized = counts / cell_medians[..., None]
```

---

## Integration with Operators

DiffBio operators use soft_ops internally through the `TemperatureOperator` base
class. This base class stores the `softness`/`temperature` parameter (optionally
as a learnable `nnx.Param`) and provides convenience methods:

```python
from diffbio.core.base_operators import TemperatureOperator
from datarax.core.config import OperatorConfig
from dataclasses import dataclass

@dataclass(frozen=True)
class MyFilterConfig(OperatorConfig):
    """Configuration for a quality-aware filter."""
    temperature: float = 0.5
    learnable_temperature: bool = False
    quality_threshold: float = 20.0

class QualityFilter(TemperatureOperator):
    def __init__(self, config, *, rngs=None, name=None):
        super().__init__(config, rngs=rngs, name=name)
        self.threshold = config.quality_threshold

    def apply(self, data, state, metadata, random_params=None, stats=None):
        from diffbio.core import soft_ops

        quality = data["quality_scores"]

        # self._temperature comes from TemperatureOperator
        mask = soft_ops.greater(
            quality,
            self.threshold,
            softness=self._temperature,
        )
        filtered = soft_ops.where(mask, data["reads"], 0.0)
        return {**data, "filtered_reads": filtered}, state, metadata
```

The `TemperatureOperator` also provides `soft_max` (via logsumexp relaxation)
and `soft_argmax` methods that are used directly by core algorithms like
Smith-Waterman alignment, Viterbi decoding, and Nussinov RNA folding.

### Learnable Temperature

When `learnable_temperature=True` in the operator config, the temperature
becomes an `nnx.Param` that is updated by gradient descent alongside all
other model parameters. This allows the model to learn the optimal
accuracy/smoothness trade-off from data.

---

## Summary

Soft operations provide the differentiable foundation for DiffBio's entire
operator ecosystem. By replacing hard decisions with smooth approximations:

- **Comparisons** become probabilities (SoftBool)
- **Indices** become distributions (SoftIndex)
- **Sorting** becomes continuous reordering
- **Selection** becomes weighted averaging
- **Logic** becomes fuzzy logic

The `softness` parameter gives fine-grained control over the
accuracy/differentiability trade-off, and the five smoothness modes (`hard`,
`smooth`, `c0`, `c1`, `c2`) let you choose the right mathematical regularity
for your optimization method. Straight-through estimators bridge the gap when
you need exact forward-pass outputs with soft backward-pass gradients.
