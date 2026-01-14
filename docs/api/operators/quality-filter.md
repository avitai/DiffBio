# Quality Filter API

Differentiable quality filter for sequence preprocessing.

## DifferentiableQualityFilter

::: diffbio.operators.quality_filter.DifferentiableQualityFilter
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## QualityFilterConfig

::: diffbio.operators.quality_filter.QualityFilterConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Basic Quality Filtering

```python
import jax.numpy as jnp
from diffbio.operators import DifferentiableQualityFilter, QualityFilterConfig

# Configure
config = QualityFilterConfig(initial_threshold=20.0)
filter_op = DifferentiableQualityFilter(config)

# Prepare data
sequence = jnp.eye(4)[jnp.array([0, 1, 2, 3, 0, 1])]  # ACGTAC
quality = jnp.array([30.0, 25.0, 10.0, 35.0, 15.0, 28.0])

# Apply filter
data = {"sequence": sequence, "quality_scores": quality}
result, _, _ = filter_op.apply(data, {}, None)

print(f"Original sum: {sequence.sum():.2f}")
print(f"Filtered sum: {result['sequence'].sum():.2f}")
```

### Access Threshold

```python
# Get current threshold
threshold = filter_op.threshold[...]
print(f"Threshold: {threshold}")

# Update threshold
filter_op.threshold[...] = 25.0
```

### Gradient Computation

```python
import jax

def filter_loss(filter_op, sequence, quality):
    data = {"sequence": sequence, "quality_scores": quality}
    result, _, _ = filter_op.apply(data, {}, None)
    return result["sequence"].sum()

# Gradient w.r.t. threshold
grads = jax.grad(filter_loss)(filter_op, sequence, quality)
print(f"Threshold gradient: {grads.threshold}")
```

## Filter Response

The filter applies sigmoid weighting:

$$w_i = \sigma(Q_i - t) = \frac{1}{1 + e^{-(Q_i - t)}}$$

| Quality vs Threshold | Retention Weight |
|---------------------|------------------|
| Q << t | ~0 (filtered) |
| Q = t | 0.5 |
| Q >> t | ~1 (retained) |

## Input Specifications

### sequence

| Property | Value |
|----------|-------|
| Shape | `(length, alphabet_size)` |
| Type | `Float[Array, ...]` |
| Description | One-hot encoded sequence |

### quality_scores

| Property | Value |
|----------|-------|
| Shape | `(length,)` |
| Type | `Float[Array, ...]` |
| Description | Phred quality scores |

## Output Specifications

### sequence

| Property | Value |
|----------|-------|
| Shape | `(length, alphabet_size)` |
| Type | `Float[Array, ...]` |
| Description | Quality-weighted sequence |

### quality_scores

| Property | Value |
|----------|-------|
| Shape | `(length,)` |
| Type | `Float[Array, ...]` |
| Description | Original quality scores (preserved) |
