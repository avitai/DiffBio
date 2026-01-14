# Smith-Waterman Operator

The `SmoothSmithWaterman` operator provides a differentiable implementation of the Smith-Waterman local alignment algorithm.

<span class="operator-alignment">Alignment</span> <span class="diff-high">Fully Differentiable</span>

## Overview

The Smith-Waterman algorithm finds the optimal local alignment between two sequences. DiffBio's implementation uses the logsumexp relaxation to make the algorithm differentiable, enabling gradient-based optimization of alignment parameters.

## Quick Start

```python
import jax.numpy as jnp
from diffbio.operators.alignment import (
    SmoothSmithWaterman,
    SmithWatermanConfig,
    create_dna_scoring_matrix,
)

# Create scoring matrix
scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

# Configure aligner
config = SmithWatermanConfig(
    temperature=1.0,
    gap_open=-10.0,
    gap_extend=-1.0
)

# Create operator
aligner = SmoothSmithWaterman(config, scoring_matrix=scoring)

# One-hot encode sequences
seq1 = jnp.eye(4)[jnp.array([0, 1, 2, 3])]  # ACGT
seq2 = jnp.eye(4)[jnp.array([0, 1, 0, 3])]  # ACAT

# Perform alignment
result = aligner.align(seq1, seq2)
print(f"Score: {result.score:.2f}")
```

## Configuration

### SmithWatermanConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Smoothness of logsumexp approximation |
| `gap_open` | float | -10.0 | Penalty for opening a gap |
| `gap_extend` | float | -1.0 | Penalty for extending a gap |
| `stochastic` | bool | False | Whether operator uses randomness |

```python
from diffbio.operators.alignment import SmithWatermanConfig

config = SmithWatermanConfig(
    temperature=1.0,      # Lower = sharper, higher = smoother
    gap_open=-10.0,       # Penalty for starting a gap
    gap_extend=-1.0,      # Penalty per additional gap position
)
```

### Temperature Effects

<div class="benchmark-results">
<div class="benchmark-title">Temperature Trade-offs</div>

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.1 | Near-discrete, sparse gradients | Final inference |
| 1.0 | Balanced | General training |
| 5.0 | Very smooth, dense gradients | Initial training |

</div>

## API Reference

### SmoothSmithWaterman

```python
class SmoothSmithWaterman(OperatorModule):
    def __init__(
        self,
        config: SmithWatermanConfig,
        scoring_matrix: Array,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the smooth Smith-Waterman aligner.

        Args:
            config: Alignment configuration
            scoring_matrix: Scoring matrix (alphabet_size, alphabet_size)
            rngs: Random number generators (optional)
            name: Optional operator name
        """
```

### Methods

#### align()

```python
def align(
    self,
    seq1: Float[Array, "len1 alphabet"],
    seq2: Float[Array, "len2 alphabet"],
) -> AlignmentResult:
    """Perform smooth Smith-Waterman local alignment.

    Args:
        seq1: First sequence, one-hot encoded (len1, alphabet_size)
        seq2: Second sequence, one-hot encoded (len2, alphabet_size)

    Returns:
        AlignmentResult with score, alignment_matrix, and soft_alignment
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
    """Apply alignment to sequence pair data (Datarax interface).

    Expected data keys:
        - "seq1": First sequence, one-hot encoded
        - "seq2": Second sequence, one-hot encoded

    Output data keys:
        - "seq1", "seq2": Original sequences
        - "score": Alignment score
        - "alignment_matrix": DP matrix
        - "soft_alignment": Position correspondence probabilities
    """
```

### AlignmentResult

```python
class AlignmentResult(NamedTuple):
    score: Float[Array, ""]           # Soft alignment score
    alignment_matrix: Float[Array, "len1_plus1 len2_plus1"]  # DP matrix
    soft_alignment: Float[Array, "len1 len2"]  # Position correspondences
```

## Scoring Matrices

### Pre-defined Matrices

```python
from diffbio.operators.alignment import (
    DNA_SIMPLE,           # Simple DNA match/mismatch
    RNA_SIMPLE,           # Simple RNA match/mismatch
    BLOSUM62,             # Protein substitution matrix
    PROTEIN_ALPHABET,     # "ARNDCQEGHILKMFPSTWYV"
)
```

### Creating Custom Matrices

```python
from diffbio.operators.alignment import create_dna_scoring_matrix

# Simple match/mismatch
scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

# Custom matrix
custom = jnp.array([
    [5, -4, -4, -4],  # A matches
    [-4, 5, -4, -4],  # C matches
    [-4, -4, 5, -4],  # G matches
    [-4, -4, -4, 5],  # T matches
])
```

## Learnable Parameters

The operator has four learnable parameters:

```python
aligner = SmoothSmithWaterman(config, scoring_matrix=scoring)

# Access learnable parameters
print(aligner.temperature)      # nnx.Param
print(aligner.scoring_matrix)   # nnx.Param
print(aligner.gap_open)         # nnx.Param
print(aligner.gap_extend)       # nnx.Param
```

### Training Example

```python
import jax
import optax
from flax import nnx

# Define loss function
def alignment_loss(aligner, seq_pairs, target_scores):
    total_loss = 0.0
    for (s1, s2), target in zip(seq_pairs, target_scores):
        result = aligner.align(s1, s2)
        total_loss += (result.score - target) ** 2
    return total_loss / len(seq_pairs)

# Get parameters
params = nnx.state(aligner, nnx.Param)

# Create optimizer
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

# Training step
@jax.jit
def train_step(aligner, seq_pairs, targets, opt_state):
    loss, grads = jax.value_and_grad(alignment_loss)(
        aligner, seq_pairs, targets
    )
    params = nnx.state(aligner, nnx.Param)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    nnx.update(aligner, optax.apply_updates(params, updates))
    return loss, opt_state

# Train
for epoch in range(100):
    loss, opt_state = train_step(aligner, train_pairs, train_targets, opt_state)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")
```

## Advanced Usage

### Gradient Analysis

Analyze which parameters affect alignment most:

```python
import jax

def score_fn(scoring_matrix, gap_open, gap_extend, temp, seq1, seq2):
    config = SmithWatermanConfig(
        temperature=temp,
        gap_open=gap_open,
        gap_extend=gap_extend
    )
    aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix)
    return aligner.align(seq1, seq2).score

# Gradients w.r.t. all parameters
grad_fn = jax.grad(score_fn, argnums=(0, 1, 2, 3))
grads = grad_fn(scoring, -10.0, -1.0, 1.0, seq1, seq2)

print(f"Scoring matrix gradient norm: {jnp.linalg.norm(grads[0]):.4f}")
print(f"Gap open gradient: {grads[1]:.4f}")
print(f"Gap extend gradient: {grads[2]:.4f}")
print(f"Temperature gradient: {grads[3]:.4f}")
```

### Soft Alignment Visualization

```python
import matplotlib.pyplot as plt

result = aligner.align(seq1, seq2)

plt.figure(figsize=(8, 6))
plt.imshow(result.soft_alignment, cmap='viridis')
plt.colorbar(label='Alignment probability')
plt.xlabel('Sequence 2 position')
plt.ylabel('Sequence 1 position')
plt.title('Soft Alignment Matrix')
plt.show()
```

### Batch Processing

```python
# Using Datarax interface
data = {"seq1": seq1, "seq2": seq2}
result_data, state, metadata = aligner.apply(data, {}, None)

# Using vmap for batches
def align_pair(pair):
    return aligner.align(pair['seq1'], pair['seq2'])

batch_align = jax.vmap(align_pair)
batch_results = batch_align(batch_data)
```

## Implementation Details

### Algorithm

The smooth Smith-Waterman replaces the standard recurrence:

$$H(i,j) = \max(0, H(i-1,j-1) + s, H(i-1,j) + g, H(i,j-1) + g)$$

With the logsumexp relaxation:

$$H(i,j) = \tau \cdot \log\sum_{k} \exp(v_k / \tau)$$

Where $v_k$ are the candidate values and $\tau$ is the temperature.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Forward pass | O(nm) | O(nm) |
| Backward pass | O(nm) | O(nm) |
| Total | O(nm) | O(nm) |

Where n, m are sequence lengths.

### JAX Optimization

The implementation uses:

- `jax.lax.fori_loop` for efficient row iteration
- `jax.lax.scan` for column iteration
- Automatic XLA compilation for GPU acceleration

## References

1. Smith, T.F. & Waterman, M.S. (1981). "Identification of common molecular subsequences."

2. Petti, S. et al. (2023). "End-to-end learning of multiple sequence alignments with differentiable Smith-Waterman."
