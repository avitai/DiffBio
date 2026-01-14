# Simple Alignment Example

This example demonstrates basic differentiable sequence alignment using DiffBio's Smith-Waterman operator.

## Setup

```python
import jax
import jax.numpy as jnp
from diffbio.operators.alignment import (
    SmoothSmithWaterman,
    SmithWatermanConfig,
    create_dna_scoring_matrix,
)
```

## Create the Aligner

```python
# Create a DNA scoring matrix
# match=2.0 for identical nucleotides, mismatch=-1.0 for different ones
scoring_matrix = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

# Configure the Smith-Waterman aligner
config = SmithWatermanConfig(
    temperature=1.0,      # Smoothness parameter (1.0 is balanced)
    gap_open=-10.0,       # Penalty for starting a gap
    gap_extend=-1.0,      # Penalty per additional gap position
)

# Create the aligner
aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix)
```

## Encode Sequences

DiffBio uses one-hot encoding for sequences. The DNA alphabet is A=0, C=1, G=2, T=3.

```python
def one_hot_dna(sequence_string):
    """Convert DNA sequence string to one-hot encoding."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    indices = jnp.array([mapping[c] for c in sequence_string])
    return jnp.eye(4)[indices]

# Example sequences
seq1 = one_hot_dna("ACGTACGT")  # 8 nucleotides
seq2 = one_hot_dna("ACATACGT")  # Different at position 3 (G->A)

print(f"Sequence 1 shape: {seq1.shape}")  # (8, 4)
print(f"Sequence 2 shape: {seq2.shape}")  # (8, 4)
```

## Perform Alignment

```python
# Run the alignment
result = aligner.align(seq1, seq2)

print(f"Alignment score: {result.score:.4f}")
print(f"Alignment matrix shape: {result.alignment_matrix.shape}")  # (9, 9)
print(f"Soft alignment shape: {result.soft_alignment.shape}")      # (8, 8)
```

## Interpret Results

### Alignment Score

The alignment score indicates overall similarity:

```python
# Higher score = better alignment
print(f"Score: {result.score:.4f}")
```

### Soft Alignment Matrix

The soft alignment shows position correspondences as probabilities:

```python
# Find most likely corresponding positions
best_matches = jnp.argmax(result.soft_alignment, axis=1)
print(f"Position correspondences: {best_matches}")
# Position i in seq1 corresponds to position best_matches[i] in seq2
```

### Visualization (Optional)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.imshow(result.soft_alignment, cmap='viridis')
plt.colorbar(label='Alignment probability')
plt.xlabel('Sequence 2 position')
plt.ylabel('Sequence 1 position')
plt.title('Soft Alignment Matrix')
plt.show()
```

## Compute Gradients

The key feature of DiffBio is differentiability:

```python
# Define a loss function based on alignment score
def alignment_loss(scoring_matrix, seq1, seq2):
    config = SmithWatermanConfig(temperature=1.0)
    aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix)
    result = aligner.align(seq1, seq2)
    return -result.score  # Negative because we want to maximize

# Compute gradient w.r.t. scoring matrix
grad_fn = jax.grad(alignment_loss)
grads = grad_fn(scoring_matrix, seq1, seq2)

print("Scoring matrix gradients:")
print(grads)
```

The gradients tell us how to adjust the scoring matrix to improve alignment.

## Using the Datarax Interface

For batch processing, use the `apply()` method:

```python
# Prepare data as dictionary
data = {
    "seq1": seq1,
    "seq2": seq2,
}

# Apply operator
result_data, state, metadata = aligner.apply(data, {}, None)

# Access results
print(f"Score: {result_data['score']:.4f}")
print(f"Keys: {result_data.keys()}")
```

## Batch Processing with vmap

Process multiple sequence pairs in parallel:

```python
# Create batch of sequence pairs
batch_seq1 = jnp.stack([
    one_hot_dna("ACGTACGT"),
    one_hot_dna("TTTTAAAA"),
    one_hot_dna("GCGCGCGC"),
])

batch_seq2 = jnp.stack([
    one_hot_dna("ACATACGT"),
    one_hot_dna("TTTTAAAG"),
    one_hot_dna("GCGCATGC"),
])

# Vectorize alignment
def align_pair(s1, s2):
    return aligner.align(s1, s2)

batch_align = jax.vmap(align_pair)
batch_results = batch_align(batch_seq1, batch_seq2)

print(f"Batch scores: {batch_results.score}")
```

## Temperature Effects

The temperature parameter controls the smoothness of the alignment:

```python
temperatures = [0.1, 1.0, 5.0, 10.0]

for temp in temperatures:
    config = SmithWatermanConfig(temperature=temp)
    aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix)
    result = aligner.align(seq1, seq2)
    print(f"Temperature {temp}: score = {result.score:.4f}")
```

- **Low temperature** (0.1): Near-discrete, sharp alignment
- **High temperature** (10.0): Very smooth, gradients flow more easily

## Learning Optimal Parameters

Train the scoring matrix for a specific task:

```python
import optax
from flax import nnx

# Initial scoring matrix
initial_scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

# Create aligner with learnable scoring
config = SmithWatermanConfig(temperature=1.0)
aligner = SmoothSmithWaterman(config, scoring_matrix=initial_scoring)

# Define target (we want seq1 and seq2 to align with score > 10)
target_score = 10.0

def loss(aligner, seq1, seq2, target):
    result = aligner.align(seq1, seq2)
    return (result.score - target) ** 2

# Optimizer
params = nnx.state(aligner, nnx.Param)
optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(params)

# Training loop
for step in range(100):
    loss_val, grads = jax.value_and_grad(loss)(aligner, seq1, seq2, target_score)

    params = nnx.state(aligner, nnx.Param)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    nnx.update(aligner, optax.apply_updates(params, updates))

    if step % 20 == 0:
        print(f"Step {step}: loss = {loss_val:.4f}")

# Final score
final_result = aligner.align(seq1, seq2)
print(f"Final score: {final_result.score:.4f} (target: {target_score})")
```

## Summary

This example demonstrated:

1. Creating a differentiable Smith-Waterman aligner
2. One-hot encoding DNA sequences
3. Performing smooth alignments
4. Computing gradients for optimization
5. Using vmap for batch processing
6. Learning optimal alignment parameters

## Next Steps

- Try [Pileup Generation](pileup-generation.md) for variant calling
- See [Variant Calling Pipeline](../advanced/variant-calling.md) for a complete workflow
