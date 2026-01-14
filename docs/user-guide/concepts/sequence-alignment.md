# Sequence Alignment

Sequence alignment is the process of arranging two or more sequences to identify regions of similarity. DiffBio provides differentiable implementations of alignment algorithms that enable gradient-based optimization.

## Background

### The Alignment Problem

Given two sequences, find the optimal arrangement that:

- Maximizes matches between similar characters
- Minimizes penalties for mismatches and gaps
- Reveals evolutionary or functional relationships

### Types of Alignment

| Type | Description | Use Case |
|------|-------------|----------|
| **Global** | Aligns entire sequences end-to-end | Comparing full genes |
| **Local** | Finds best matching subsequences | Finding domains, motifs |
| **Semi-global** | Global in one sequence, local in other | Mapping reads to reference |

DiffBio currently implements **local alignment** (Smith-Waterman).

## Smith-Waterman Algorithm

The Smith-Waterman algorithm finds the optimal local alignment using dynamic programming.

### Recurrence Relation

For sequences $A$ (length $m$) and $B$ (length $n$), define matrix $H$:

$$
H(i, j) = \max \begin{cases}
0 \\
H(i-1, j-1) + s(A_i, B_j) \\
H(i-1, j) + g \\
H(i, j-1) + g
\end{cases}
$$

Where:

- $s(a, b)$ is the scoring function (match/mismatch)
- $g$ is the gap penalty (usually negative)
- $H(i, 0) = H(0, j) = 0$ (local alignment allows free ends)

The optimal score is $\max_{i,j} H(i,j)$.

### The Differentiability Challenge

The `max` operation in the recurrence is **non-differentiable**:

- Gradient only flows through the winning branch
- Most gradients are zero (vanishing gradient problem)
- Cannot learn which branch *should* win

## Smooth Smith-Waterman

DiffBio's `SmoothSmithWaterman` replaces `max` with `logsumexp`:

$$
H(i, j) = \tau \cdot \text{logsumexp}\left(\frac{1}{\tau} \begin{bmatrix} 0 \\ H(i-1, j-1) + s(A_i, B_j) \\ H(i-1, j) + g \\ H(i, j-1) + g \end{bmatrix}\right)
$$

This is a smooth approximation where gradients flow through all branches, weighted by their probability.

### Temperature Control

The temperature $\tau$ controls the smoothness:

```python
# Low temperature (τ = 0.1): Sharp, near-discrete
# Medium temperature (τ = 1.0): Balanced
# High temperature (τ = 10.0): Very smooth

config = SmithWatermanConfig(temperature=1.0)
```

<div class="technique-card">
<div class="technique-name">Temperature Annealing</div>
<div class="technique-description">
Start with high temperature for exploration, gradually decrease for convergence to discrete solutions.
</div>
<div class="technique-use-cases">
Use when training aligners from scratch or fine-tuning on new domains.
</div>
</div>

## Scoring Matrices

The scoring function $s(a, b)$ is typically defined by a **substitution matrix**:

### DNA Scoring

For DNA sequences (A, C, G, T):

```python
from diffbio.operators.alignment import create_dna_scoring_matrix

# Simple match/mismatch scoring
scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
# Result: 4x4 matrix with 2.0 on diagonal, -1.0 elsewhere
```

### Learned Scoring

DiffBio allows learning the scoring matrix:

```python
aligner = SmoothSmithWaterman(config, scoring_matrix=initial_scoring)

# The scoring matrix is a learnable parameter
print(aligner.scoring_matrix)  # nnx.Param

# Train to optimize for your objective
def loss_fn(aligner, seq1, seq2, target_score):
    result = aligner.align(seq1, seq2)
    return (result.score - target_score) ** 2

grads = jax.grad(loss_fn)(aligner, seq1, seq2, target)
```

## Gap Penalties

Gaps represent insertions or deletions (indels) in the alignment.

### Linear Gap Penalty

Simple penalty per gap position:

$$
\text{gap\_cost}(k) = g \cdot k
$$

### Affine Gap Penalty

Separate penalties for opening and extending gaps:

$$
\text{gap\_cost}(k) = g_o + g_e \cdot (k - 1)
$$

Where:

- $g_o$ is the gap opening penalty
- $g_e$ is the gap extension penalty

```python
config = SmithWatermanConfig(
    gap_open=-10.0,    # Penalty for starting a gap
    gap_extend=-1.0,   # Penalty per additional gap position
)
```

!!! tip "Learnable Gap Penalties"
    Gap penalties are learnable parameters in DiffBio. The optimal penalties depend on your data and can be learned end-to-end.

## Soft Alignment Output

The `SmoothSmithWaterman` operator returns three outputs:

### 1. Alignment Score

The soft maximum score:

```python
result.score  # Scalar, differentiable
```

### 2. Alignment Matrix

The full DP matrix showing scores at each position:

```python
result.alignment_matrix  # Shape: (len1+1, len2+1)
```

### 3. Soft Alignment

A probability matrix showing the correspondence between positions:

```python
result.soft_alignment  # Shape: (len1, len2)
# soft_alignment[i, j] = probability that position i aligns to position j
```

## Usage Example

```python
import jax.numpy as jnp
from diffbio.operators import SmoothSmithWaterman, SmithWatermanConfig
from diffbio.operators.alignment import create_dna_scoring_matrix

# Setup
config = SmithWatermanConfig(
    temperature=1.0,
    gap_open=-10.0,
    gap_extend=-1.0
)
scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
aligner = SmoothSmithWaterman(config, scoring_matrix=scoring)

# One-hot encode sequences
def one_hot(indices, vocab_size=4):
    return jnp.eye(vocab_size)[indices]

seq1 = one_hot(jnp.array([0, 1, 2, 3, 0, 1, 2, 3]))  # ACGTACGT
seq2 = one_hot(jnp.array([0, 1, 0, 3, 0, 1, 2, 3]))  # ACATACGT

# Align
result = aligner.align(seq1, seq2)

print(f"Score: {result.score:.2f}")
print(f"Best aligned positions: {jnp.argmax(result.soft_alignment, axis=1)}")
```

## Batch Processing

Use the Datarax interface for batch processing:

```python
# Prepare batch data
data = {"seq1": seq1, "seq2": seq2}
state = {}
metadata = None

# Apply operator
result_data, state, metadata = aligner.apply(data, state, metadata)

# Access results
print(result_data['score'])
print(result_data['alignment_matrix'])
print(result_data['soft_alignment'])
```

## Advanced: Gradient Analysis

Analyze how alignment parameters affect the score:

```python
import jax

# Gradient w.r.t. scoring matrix
def alignment_score(scoring_matrix, seq1, seq2, config):
    aligner = SmoothSmithWaterman(config, scoring_matrix=scoring_matrix)
    return aligner.align(seq1, seq2).score

grad_scoring = jax.grad(alignment_score)(scoring, seq1, seq2, config)
print("Scoring matrix gradients:")
print(grad_scoring)

# Which positions in scoring matrix matter most for this alignment?
important_pairs = jnp.unravel_index(jnp.argmax(jnp.abs(grad_scoring)), grad_scoring.shape)
print(f"Most influential nucleotide pair: {important_pairs}")
```

## References

1. Smith, T.F. & Waterman, M.S. (1981). "Identification of common molecular subsequences." *Journal of Molecular Biology*, 147(1), 195-197.

2. Petti, S. et al. (2023). "End-to-end learning of multiple sequence alignments with differentiable Smith-Waterman." *Bioinformatics*, 39(1).
