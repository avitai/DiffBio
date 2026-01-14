# Pileup Operator

The `DifferentiablePileup` operator generates soft read pileups for variant calling, aggregating aligned reads into position-wise nucleotide distributions.

<span class="operator-pileup">Pileup</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Pileup generation aggregates aligned sequencing reads at each reference position. Unlike traditional pileup tools that produce integer counts, DiffBio's implementation produces continuous distributions that enable gradient flow for end-to-end training.

## Quick Start

```python
import jax
import jax.numpy as jnp
from diffbio.operators.variant import DifferentiablePileup, PileupConfig

# Configure pileup
config = PileupConfig(
    reference_length=100,
    use_quality_weights=True
)

# Create operator
pileup_op = DifferentiablePileup(config)

# Prepare data
num_reads = 20
read_length = 30

reads = jax.random.uniform(jax.random.PRNGKey(0), (num_reads, read_length, 4))
reads = jax.nn.softmax(reads, axis=-1)  # Soft one-hot
positions = jax.random.randint(jax.random.PRNGKey(1), (num_reads,), 0, 70)
quality = jax.random.uniform(jax.random.PRNGKey(2), (num_reads, read_length), 10, 40)

# Generate pileup
data = {"reads": reads, "positions": positions, "quality": quality}
result, _, _ = pileup_op.apply(data, {}, None)

print(f"Pileup shape: {result['pileup'].shape}")  # (100, 4)
```

## Configuration

### PileupConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_length` | int | 100 | Length of reference sequence |
| `window_size` | int | 21 | Context window size |
| `min_coverage` | int | 1 | Minimum coverage threshold |
| `max_coverage` | int | 100 | Maximum coverage for normalization |
| `use_quality_weights` | bool | True | Weight bases by quality scores |
| `stochastic` | bool | False | Whether operator uses randomness |

```python
from diffbio.operators.variant import PileupConfig

config = PileupConfig(
    reference_length=10000,     # Must match your reference
    use_quality_weights=True,   # Recommended for quality-aware pileup
    min_coverage=5,             # Minimum reads for reliable calls
    max_coverage=200,           # Cap for normalization
)
```

## API Reference

### DifferentiablePileup

```python
class DifferentiablePileup(OperatorModule):
    def __init__(
        self,
        config: PileupConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize differentiable pileup generator.

        Args:
            config: Pileup configuration
            rngs: Random number generators (optional)
            name: Optional operator name
        """
```

### Methods

#### compute_pileup()

```python
def compute_pileup(
    self,
    reads: Float[Array, "num_reads read_length 4"],
    positions: Int[Array, "num_reads"],
    quality: Float[Array, "num_reads read_length"],
    reference_length: int,
) -> Float[Array, "reference_length 4"]:
    """Generate pileup from aligned reads.

    Args:
        reads: One-hot encoded reads
        positions: Starting position of each read
        quality: Quality scores for each base
        reference_length: Length of reference sequence

    Returns:
        Pileup array with nucleotide distributions at each position
    """
```

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
    """Apply pileup generation (Datarax interface).

    Expected data keys:
        - "reads": One-hot encoded reads (num_reads, read_length, 4)
        - "positions": Starting position of each read (num_reads,)
        - "quality": Quality scores (num_reads, read_length)

    Output data keys:
        - "reads", "positions", "quality": Original inputs
        - "pileup": Generated pileup (reference_length, 4)
    """
```

## Input Format

### Reads

One-hot encoded reads with shape `(num_reads, read_length, 4)`:

```python
# Hard one-hot (typical)
read_indices = jnp.array([0, 1, 2, 3, 0, 1])  # ACGTAC
read = jnp.eye(4)[read_indices]  # (6, 4)

# Soft one-hot (for training)
read_soft = jax.nn.softmax(logits, axis=-1)  # (length, 4)
```

### Positions

Integer starting positions for each read:

```python
positions = jnp.array([10, 25, 42, ...])  # (num_reads,)
# Read i covers positions[i] to positions[i] + read_length - 1
```

### Quality Scores

Phred quality scores for each base:

```python
quality = jnp.array([
    [30, 35, 28, 40, ...],  # Read 1 quality scores
    [25, 30, 32, 35, ...],  # Read 2 quality scores
    ...
])  # (num_reads, read_length)
```

## Output Format

The output pileup has shape `(reference_length, 4)`:

```python
pileup = result['pileup']

# At each position, pileup[i] is a probability distribution
# pileup[i, 0] = P(A at position i)
# pileup[i, 1] = P(C at position i)
# pileup[i, 2] = P(G at position i)
# pileup[i, 3] = P(T at position i)

# Sum to 1 at each position
assert jnp.allclose(pileup.sum(axis=-1), 1.0)
```

## Quality Weighting

When `use_quality_weights=True`, bases are weighted by quality:

$$w = 1 - 10^{-Q/10}$$

| Phred Score | Error Rate | Weight |
|-------------|------------|--------|
| 10 | 10% | 0.90 |
| 20 | 1% | 0.99 |
| 30 | 0.1% | 0.999 |
| 40 | 0.01% | 0.9999 |

```python
# Quality weighting in action
def quality_to_weight(phred_scores):
    p_error = jnp.power(10.0, -phred_scores / 10.0)
    return 1.0 - p_error

# High quality base contributes more
weight_q30 = quality_to_weight(30)  # 0.999
weight_q10 = quality_to_weight(10)  # 0.9
```

## Advanced Usage

### Variant Detection from Pileup

```python
def detect_variants(pileup, reference):
    """Identify positions where pileup differs from reference.

    Args:
        pileup: (reference_length, 4) nucleotide distributions
        reference: (reference_length, 4) one-hot reference

    Returns:
        Variant scores at each position
    """
    # Probability of reference base
    ref_prob = (pileup * reference).sum(axis=-1)

    # Variant probability = 1 - reference probability
    variant_prob = 1.0 - ref_prob

    return variant_prob

variant_scores = detect_variants(result['pileup'], reference_onehot)
high_confidence_variants = variant_scores > 0.3
```

### Coverage Analysis

```python
def compute_coverage(pileup, reads, positions, reference_length):
    """Compute soft coverage at each position."""
    # Create coverage array
    read_length = reads.shape[1]
    coverage = jnp.zeros(reference_length)

    for i in range(reads.shape[0]):
        pos = positions[i]
        coverage = coverage.at[pos:pos+read_length].add(1.0)

    return coverage

# Identify low coverage regions
coverage = compute_coverage(reads, positions, config.reference_length)
low_coverage_mask = coverage < config.min_coverage
```

### Gradient-Based Analysis

```python
import jax

def pileup_entropy(reads, positions, quality, config):
    """Compute entropy of pileup (measure of uncertainty)."""
    pileup_op = DifferentiablePileup(config)
    data = {"reads": reads, "positions": positions, "quality": quality}
    result, _, _ = pileup_op.apply(data, {}, None)

    pileup = result['pileup']
    # Entropy: -sum(p * log(p))
    entropy = -(pileup * jnp.log(pileup + 1e-8)).sum(axis=-1)
    return entropy.mean()

# Gradient of entropy w.r.t. quality scores
grad_fn = jax.grad(pileup_entropy, argnums=2)
quality_grads = grad_fn(reads, positions, quality, config)

# Which bases' quality scores most affect uncertainty?
most_influential = jnp.unravel_index(
    jnp.argmax(jnp.abs(quality_grads)),
    quality_grads.shape
)
```

### Integration with Variant Calling

```python
from diffbio.operators.variant import VariantClassifier, VariantClassifierConfig

# Pileup generation
pileup_config = PileupConfig(reference_length=1000)
pileup_op = DifferentiablePileup(pileup_config)

# Variant classification
classifier_config = VariantClassifierConfig(hidden_dims=[64, 32])
classifier = VariantClassifier(classifier_config)

def variant_calling_pipeline(reads, positions, quality):
    # Generate pileup
    data = {"reads": reads, "positions": positions, "quality": quality}
    pileup_result, _, _ = pileup_op.apply(data, {}, None)

    # Classify variants
    pileup = pileup_result['pileup']
    variant_probs = classifier(pileup)

    return variant_probs  # (reference_length, 3) for ref/het/hom

# End-to-end gradient computation
loss_fn = lambda r, p, q, targets: cross_entropy(
    variant_calling_pipeline(r, p, q), targets
)
grads = jax.grad(loss_fn)(reads, positions, quality, true_variants)
```

## Implementation Details

### Aggregation Algorithm

The pileup uses JAX's `segment_sum` for efficient aggregation:

```python
# For each base position in each read, compute absolute reference position
absolute_positions = positions[:, None] + jnp.arange(read_length)

# Flatten and aggregate
flat_positions = absolute_positions.reshape(-1)
flat_reads = reads.reshape(-1, 4) * weights.reshape(-1, 1)

# Aggregate at each reference position
pileup = jax.ops.segment_sum(
    flat_reads,
    flat_positions,
    num_segments=reference_length
)
```

### Normalization

After aggregation, the pileup is normalized to probability distributions:

```python
# Normalize by coverage
coverage = jax.ops.segment_sum(weights, positions, num_segments=ref_len)
pileup_normalized = pileup / jnp.maximum(coverage, 1e-8)

# Apply softmax for valid distribution
pileup_final = jax.nn.softmax(pileup_normalized / temperature, axis=-1)
```

### Out-of-Bounds Handling

Reads extending beyond reference boundaries are automatically clipped:

```python
# Mask positions outside valid range
in_bounds = (positions >= 0) & (positions < reference_length)
weights = weights * in_bounds.astype(jnp.float32)
```

## Performance Considerations

### Memory

| Component | Memory |
|-----------|--------|
| Input reads | O(num_reads × read_length × 4) |
| Intermediate | O(num_reads × read_length) |
| Output pileup | O(reference_length × 4) |

For large datasets, process in chunks:

```python
def chunked_pileup(reads_list, positions_list, quality_list, config, chunk_size=1000):
    pileups = []
    for i in range(0, len(reads_list), chunk_size):
        chunk_reads = jnp.stack(reads_list[i:i+chunk_size])
        chunk_pos = jnp.stack(positions_list[i:i+chunk_size])
        chunk_qual = jnp.stack(quality_list[i:i+chunk_size])

        data = {"reads": chunk_reads, "positions": chunk_pos, "quality": chunk_qual}
        result, _, _ = pileup_op.apply(data, {}, None)
        pileups.append(result['pileup'])

    # Combine chunks (average)
    return jnp.mean(jnp.stack(pileups), axis=0)
```

### GPU Acceleration

```python
# JIT compile for GPU
@jax.jit
def fast_pileup(reads, positions, quality):
    data = {"reads": reads, "positions": positions, "quality": quality}
    result, _, _ = pileup_op.apply(data, {}, None)
    return result['pileup']

# First call compiles, subsequent calls are fast
pileup = fast_pileup(reads, positions, quality)
```

## References

1. Li, H. et al. (2009). "The Sequence Alignment/Map format and SAMtools."

2. Poplin, R. et al. (2018). "A universal SNP and small-indel variant caller using deep neural networks."
