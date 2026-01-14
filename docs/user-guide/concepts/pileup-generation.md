# Pileup Generation

Pileup generation aggregates aligned sequencing reads at each position of a reference genome, creating a summary view for variant calling and other downstream analyses.

## Background

### What is a Pileup?

A pileup shows all reads overlapping each reference position:

```
Reference: A  C  G  T  A  C  G  T
           |  |  |  |  |  |  |  |
Read 1:    A  C  G  T  A  .  .  .
Read 2:    .  C  G  T  A  C  G  .
Read 3:    .  .  G  A  A  C  G  T
           ----------------------
Position:  1  2  3  4  5  6  7  8
```

At position 4: Reference is T, reads show {T, T, A} → potential variant!

### Traditional Pileup

Traditional tools (samtools, etc.) produce discrete counts:

```
Position 4: A=1, C=0, G=0, T=2, Coverage=3
```

This is non-differentiable: small changes in read quality don't change integer counts.

## Differentiable Pileup

DiffBio's `DifferentiablePileup` produces soft nucleotide distributions:

$$
P(i, j) = \frac{\sum_{r \in \text{reads}} w_r \cdot \mathbf{1}[\text{read } r \text{ covers position } i] \cdot s_{r,j}}{\sum_{r} w_r \cdot \mathbf{1}[\text{read } r \text{ covers position } i]}
$$

Where:

- $P(i, j)$ = probability of nucleotide $j$ at position $i$
- $w_r$ = quality-derived weight for read $r$
- $s_{r,j}$ = soft base call for read $r$ at nucleotide $j$

### Quality Weighting

Quality scores are converted to weights:

$$
w = 1 - 10^{-Q/10}
$$

Where $Q$ is the Phred quality score. Higher quality → higher weight.

```python
# Phred 30 (99.9% accuracy) → weight ≈ 0.999
# Phred 20 (99% accuracy)   → weight ≈ 0.99
# Phred 10 (90% accuracy)   → weight ≈ 0.9
```

## Usage

### Basic Pileup Generation

```python
import jax.numpy as jnp
from diffbio.operators import DifferentiablePileup, PileupConfig

# Configure pileup
config = PileupConfig(
    reference_length=100,    # Length of reference
    use_quality_weights=True, # Weight by quality scores
    min_coverage=1,
    max_coverage=100
)

pileup_op = DifferentiablePileup(config)

# Prepare input data
num_reads = 20
read_length = 50

# One-hot encoded reads: (num_reads, read_length, 4)
reads = jnp.eye(4)[jax.random.randint(
    jax.random.PRNGKey(0),
    (num_reads, read_length),
    0, 4
)]

# Starting positions for each read
positions = jax.random.randint(
    jax.random.PRNGKey(1),
    (num_reads,),
    0, 50
)

# Quality scores: (num_reads, read_length)
quality = jax.random.uniform(
    jax.random.PRNGKey(2),
    (num_reads, read_length),
    minval=10.0,
    maxval=40.0
)

# Generate pileup
data = {"reads": reads, "positions": positions, "quality": quality}
result, _, _ = pileup_op.apply(data, {}, None)

pileup = result['pileup']  # Shape: (reference_length, 4)
print(f"Pileup shape: {pileup.shape}")
```

### Understanding the Output

The pileup output is a probability distribution over nucleotides at each position:

```python
# Check distribution at position 25
pos_25 = pileup[25]
print(f"Position 25: A={pos_25[0]:.3f}, C={pos_25[1]:.3f}, "
      f"G={pos_25[2]:.3f}, T={pos_25[3]:.3f}")
print(f"Sum (should be 1): {pos_25.sum():.3f}")
```

### Interpreting Pileup for Variants

```python
# Reference sequence (one-hot encoded)
reference = jnp.eye(4)[reference_indices]  # (reference_length, 4)

# Compute divergence from reference
def variant_likelihood(pileup, reference):
    # Probability of observing non-reference base
    ref_prob = (pileup * reference).sum(axis=-1)  # P(reference base)
    alt_prob = 1 - ref_prob  # P(any alternate base)
    return alt_prob

variant_scores = variant_likelihood(pileup, reference)
# High scores indicate potential variants
```

## Configuration Options

### PileupConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_length` | int | 100 | Length of reference sequence |
| `window_size` | int | 21 | Context window size |
| `min_coverage` | int | 1 | Minimum coverage threshold |
| `max_coverage` | int | 100 | Maximum coverage for normalization |
| `use_quality_weights` | bool | True | Weight by quality scores |

```python
config = PileupConfig(
    reference_length=10000,
    use_quality_weights=True,
    min_coverage=5,
    max_coverage=200
)
```

## Implementation Details

### Efficient Aggregation

DiffBio uses JAX's `segment_sum` for efficient pileup computation:

```python
# Aggregate nucleotide contributions at each position
pileup = jax.ops.segment_sum(
    weighted_reads,           # (num_bases, 4)
    clipped_positions,        # (num_bases,)
    num_segments=reference_length
)
```

This is differentiable and efficient on GPU.

### Handling Variable Read Lengths

If reads have different lengths, pad to maximum length:

```python
def pad_reads(reads_list, max_length):
    padded = []
    for read in reads_list:
        padding = max_length - len(read)
        padded_read = jnp.pad(read, ((0, padding), (0, 0)))
        padded.append(padded_read)
    return jnp.stack(padded)
```

### Out-of-Bounds Handling

Reads extending beyond reference are automatically clipped:

```python
# Mask out-of-bounds positions
in_bounds = (positions >= 0) & (positions < reference_length)
weights = weights * in_bounds.astype(jnp.float32)
```

## Gradient Flow

The pileup is fully differentiable:

```python
import jax

def pileup_loss(reads, positions, quality, target_pileup, config):
    pileup_op = DifferentiablePileup(config)
    data = {"reads": reads, "positions": positions, "quality": quality}
    result, _, _ = pileup_op.apply(data, {}, None)
    return jnp.mean((result['pileup'] - target_pileup) ** 2)

# Compute gradients w.r.t. reads (for adversarial analysis, etc.)
grad_fn = jax.grad(pileup_loss)
grads = grad_fn(reads, positions, quality, target, config)
```

## Comparison: Hard vs Soft Pileup

| Aspect | Hard Pileup | Soft Pileup (DiffBio) |
|--------|-------------|----------------------|
| Output | Integer counts | Probability distribution |
| Quality handling | Binary filter | Continuous weights |
| Differentiable | No | Yes |
| Gradient flow | Blocked | Enabled |
| Memory | Lower | Higher |

## Use Cases

### 1. Variant Calling Training

```python
# Train variant caller end-to-end
def train_variant_caller(reads, positions, quality, true_variants):
    pileup = generate_pileup(reads, positions, quality)
    predictions = variant_classifier(pileup)
    loss = cross_entropy(predictions, true_variants)
    return loss  # Gradients flow back through pileup!
```

### 2. Read Quality Optimization

```python
# Learn optimal quality weighting
class LearnedQualityWeighting(nnx.Module):
    def __init__(self):
        self.weight_fn = nnx.Linear(1, 1)

    def __call__(self, quality):
        return jax.nn.sigmoid(self.weight_fn(quality[:, None]))[:, 0]
```

### 3. Coverage Analysis

```python
# Analyze coverage patterns
def compute_coverage(pileup):
    # Sum probabilities gives soft coverage
    return pileup.sum(axis=-1)

coverage = compute_coverage(pileup)
low_coverage_positions = coverage < threshold
```

## Performance Considerations

### Memory Usage

Pileup generation requires storing:

- All reads in memory: O(num_reads × read_length × 4)
- Intermediate computations: O(num_reads × read_length)
- Output pileup: O(reference_length × 4)

For large datasets, process in chunks:

```python
def chunked_pileup(reads, positions, quality, chunk_size=10000):
    num_reads = reads.shape[0]
    pileups = []
    for i in range(0, num_reads, chunk_size):
        chunk_pileup = compute_pileup(
            reads[i:i+chunk_size],
            positions[i:i+chunk_size],
            quality[i:i+chunk_size]
        )
        pileups.append(chunk_pileup)
    return sum(pileups) / len(pileups)  # Average
```

### GPU Acceleration

```python
# JIT compile for speed
@jax.jit
def fast_pileup(reads, positions, quality):
    data = {"reads": reads, "positions": positions, "quality": quality}
    result, _, _ = pileup_op.apply(data, {}, None)
    return result['pileup']

# First call compiles, subsequent calls are fast
pileup = fast_pileup(reads, positions, quality)
```

## References

1. Li, H. et al. (2009). "The Sequence Alignment/Map format and SAMtools." *Bioinformatics*, 25(16), 2078-2079.

2. Danecek, P. et al. (2021). "Twelve years of SAMtools and BCFtools." *GigaScience*, 10(2).
