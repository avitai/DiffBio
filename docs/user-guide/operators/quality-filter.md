# Quality Filter Operator

The `DifferentiableQualityFilter` operator applies soft quality-based filtering to sequence data using a learnable threshold.

<span class="operator-filter">Filter</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Quality filtering removes or down-weights low-quality bases before downstream analysis. Traditional hard filtering (discard if Q < threshold) is non-differentiable. DiffBio uses a sigmoid function to create a smooth, learnable quality filter.

## Quick Start

```python
import jax.numpy as jnp
from diffbio.operators import DifferentiableQualityFilter, QualityFilterConfig

# Configure filter
config = QualityFilterConfig(initial_threshold=20.0)

# Create operator
filter_op = DifferentiableQualityFilter(config)

# Prepare data
sequence = jnp.eye(4)[jnp.array([0, 1, 2, 3, 0, 1])]  # ACGTAC one-hot
quality = jnp.array([30.0, 25.0, 10.0, 35.0, 15.0, 28.0])  # Phred scores

# Apply filtering
data = {"sequence": sequence, "quality_scores": quality}
result, _, _ = filter_op.apply(data, {}, None)

print(f"Original sequence sum: {sequence.sum():.2f}")
print(f"Filtered sequence sum: {result['sequence'].sum():.2f}")
```

## Configuration

### QualityFilterConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_threshold` | float | 20.0 | Initial Phred quality threshold |
| `stochastic` | bool | False | Whether operator uses randomness |

```python
from diffbio.operators import QualityFilterConfig

config = QualityFilterConfig(
    initial_threshold=20.0,  # Phred 20 = 1% error rate
)
```

### Common Threshold Values

| Phred Score | Error Rate | Typical Use |
|-------------|------------|-------------|
| 10 | 10% | Low-stringency filtering |
| 20 | 1% | Standard filtering |
| 30 | 0.1% | High-stringency filtering |
| 40 | 0.01% | Very high quality only |

## API Reference

### DifferentiableQualityFilter

```python
class DifferentiableQualityFilter(OperatorModule):
    def __init__(
        self,
        config: QualityFilterConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the quality filter with learnable threshold.

        Args:
            config: Quality filter configuration
            rngs: Random number generators (optional)
            name: Optional operator name
        """
```

### Methods

#### apply()

```python
def apply(
    self,
    data: PyTree,
    state: PyTree,
    metadata: dict | None,
    random_params: Any = None,
    stats: dict | None = None,
) -> tuple[PyTree, PyTree, dict | None]:
    """Apply soft quality filtering to sequence data.

    Expected data keys:
        - "sequence": One-hot encoded sequence (length, alphabet_size)
        - "quality_scores": Phred quality scores (length,)

    Output data keys:
        - "sequence": Weighted sequence (positions scaled by quality)
        - "quality_scores": Original quality scores (preserved)

    Formula:
        retention_weight = sigmoid(quality - threshold)
        filtered_sequence = sequence * retention_weight
    """
```

## How It Works

### Soft Threshold

Instead of hard filtering:

```python
# Hard filter (non-differentiable)
mask = quality_scores >= threshold  # Binary
filtered = sequence * mask[:, None]  # All or nothing
```

DiffBio uses sigmoid:

```python
# Soft filter (differentiable)
retention_weight = jax.nn.sigmoid(quality_scores - threshold)
filtered = sequence * retention_weight[:, None]  # Smooth weighting
```

### Sigmoid Response

The sigmoid function creates a smooth transition:

$$w(Q) = \sigma(Q - t) = \frac{1}{1 + e^{-(Q-t)}}$$

| Quality vs Threshold | Weight |
|---------------------|--------|
| Q << t (much below) | ~0 (strongly filtered) |
| Q = t (at threshold) | 0.5 (half weight) |
| Q >> t (much above) | ~1 (fully retained) |

```python
import jax.numpy as jnp
import jax

threshold = 20.0
qualities = jnp.array([10, 15, 20, 25, 30])
weights = jax.nn.sigmoid(qualities - threshold)
# [0.00005, 0.0067, 0.5, 0.9933, 0.99995]
```

## Learnable Threshold

The threshold is a learnable parameter:

```python
filter_op = DifferentiableQualityFilter(config)

# Access the threshold parameter
print(filter_op.threshold)  # nnx.Param with value 20.0

# The threshold can be optimized during training
```

### Training the Threshold

```python
import jax
import optax
from flax import nnx

def pipeline_loss(filter_op, sequences, qualities, targets):
    """Loss function for downstream task."""
    filtered_seqs = []
    for seq, qual in zip(sequences, qualities):
        data = {"sequence": seq, "quality_scores": qual}
        result, _, _ = filter_op.apply(data, {}, None)
        filtered_seqs.append(result['sequence'])

    predictions = downstream_model(filtered_seqs)
    return loss_fn(predictions, targets)

# Compute gradients including w.r.t. threshold
grad_fn = jax.grad(pipeline_loss)
grads = grad_fn(filter_op, train_seqs, train_quals, train_targets)

# The threshold gradient tells us:
# - Positive: increasing threshold improves loss
# - Negative: decreasing threshold improves loss
print(f"Threshold gradient: {grads.threshold}")
```

## Advanced Usage

### Steeper/Softer Transitions

Add temperature scaling for steeper or softer sigmoid:

```python
def soft_filter(sequence, quality, threshold, temperature=1.0):
    """Quality filter with adjustable steepness."""
    # Lower temperature = steeper transition (more like hard filter)
    # Higher temperature = softer transition (more gradual)
    retention = jax.nn.sigmoid((quality - threshold) / temperature)
    return sequence * retention[:, None]

# Sharp transition (almost hard filter)
filtered_sharp = soft_filter(seq, qual, 20.0, temperature=0.1)

# Soft transition (very gradual)
filtered_soft = soft_filter(seq, qual, 20.0, temperature=5.0)
```

### Position-Dependent Threshold

Use different thresholds for different positions:

```python
class PositionalQualityFilter(nnx.Module):
    def __init__(self, sequence_length, initial_threshold=20.0):
        # Different threshold per position
        self.thresholds = nnx.Param(
            jnp.full(sequence_length, initial_threshold)
        )

    def __call__(self, sequence, quality):
        retention = jax.nn.sigmoid(quality - self.thresholds[...])
        return sequence * retention[:, None]
```

### Combining with Other Operators

```python
from diffbio.operators import (
    DifferentiableQualityFilter, QualityFilterConfig,
)
from diffbio.operators.alignment import (
    SmoothSmithWaterman, SmithWatermanConfig,
    create_dna_scoring_matrix,
)

# Create operators
filter_config = QualityFilterConfig(initial_threshold=20.0)
filter_op = DifferentiableQualityFilter(filter_config)

align_config = SmithWatermanConfig(temperature=1.0)
scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
aligner = SmoothSmithWaterman(align_config, scoring_matrix=scoring)

def filtered_alignment(seq1, qual1, seq2, qual2):
    # Filter both sequences
    data1 = {"sequence": seq1, "quality_scores": qual1}
    data2 = {"sequence": seq2, "quality_scores": qual2}

    filtered1, _, _ = filter_op.apply(data1, {}, None)
    filtered2, _, _ = filter_op.apply(data2, {}, None)

    # Align filtered sequences
    result = aligner.align(
        filtered1['sequence'],
        filtered2['sequence']
    )
    return result.score

# Gradient flows through both filter and alignment
grad_fn = jax.grad(filtered_alignment)
grads = grad_fn(seq1, qual1, seq2, qual2)
```

### Batch Processing

```python
def batch_filter(filter_op, sequences, qualities):
    """Filter a batch of sequences."""
    filtered = []
    for seq, qual in zip(sequences, qualities):
        data = {"sequence": seq, "quality_scores": qual}
        result, _, _ = filter_op.apply(data, {}, None)
        filtered.append(result['sequence'])
    return jnp.stack(filtered)

# Or use vmap for efficiency
def single_filter(filter_op, seq, qual):
    data = {"sequence": seq, "quality_scores": qual}
    result, _, _ = filter_op.apply(data, {}, None)
    return result['sequence']

batch_filter_vmap = jax.vmap(
    lambda s, q: single_filter(filter_op, s, q),
    in_axes=(0, 0)
)
```

## Visualization

### Filter Response Curve

```python
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

threshold = 20.0
qualities = jnp.linspace(0, 40, 100)
weights = jax.nn.sigmoid(qualities - threshold)

plt.figure(figsize=(8, 5))
plt.plot(qualities, weights, 'b-', linewidth=2)
plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Phred Quality Score')
plt.ylabel('Retention Weight')
plt.title('Soft Quality Filter Response')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Before/After Comparison

```python
# Visualize filtering effect
sequence = jnp.eye(4)[jnp.array([0, 1, 2, 3, 0, 1, 2, 3])]
quality = jnp.array([35, 30, 15, 40, 10, 25, 8, 32])

data = {"sequence": sequence, "quality_scores": quality}
result, _, _ = filter_op.apply(data, {}, None)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].imshow(sequence.T, aspect='auto', cmap='Blues')
axes[0].set_title('Original Sequence')
axes[0].set_ylabel('Nucleotide (A,C,G,T)')
axes[0].set_xlabel('Position')

axes[1].imshow(result['sequence'].T, aspect='auto', cmap='Blues')
axes[1].set_title('Filtered Sequence')
axes[1].set_xlabel('Position')

plt.tight_layout()
plt.show()
```

## Implementation Details

### Forward Pass

```python
def forward(sequence, quality_scores, threshold):
    # Compute retention weights
    retention_weights = jax.nn.sigmoid(quality_scores - threshold)

    # Apply weights (broadcast over alphabet dimension)
    weighted_sequence = sequence * retention_weights[:, None]

    return weighted_sequence
```

### Gradient Flow

The gradient with respect to the threshold:

$$\frac{\partial L}{\partial t} = -\sum_i \frac{\partial L}{\partial w_i} \cdot \sigma(Q_i - t) \cdot (1 - \sigma(Q_i - t))$$

This gradient is non-zero for positions near the threshold, allowing the model to learn the optimal cutoff.

### Numerical Stability

The sigmoid function is numerically stable in JAX:

```python
# JAX's sigmoid handles large positive/negative inputs gracefully
jax.nn.sigmoid(jnp.array([-100, 0, 100]))
# array([0., 0.5, 1.])
```

## Best Practices

1. **Initialize conservatively**: Start with a moderate threshold (e.g., 20) and let training adjust

2. **Monitor threshold during training**: Track how the threshold changes to understand data quality

3. **Use with downstream tasks**: The optimal threshold depends on the downstream task

4. **Consider temperature**: Add temperature scaling if you need more/less sharp filtering

5. **Preserve quality scores**: The filtered output includes original quality scores for reference

## References

1. Ewing, B. & Green, P. (1998). "Base-Calling of Automated Sequencer Traces Using Phred."

2. Cock, P.J. et al. (2010). "The Sanger FASTQ file format for sequences with quality scores."
