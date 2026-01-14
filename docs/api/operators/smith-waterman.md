# Smith-Waterman API

Differentiable Smith-Waterman local alignment operator.

## SmoothSmithWaterman

::: diffbio.operators.alignment.smith_waterman.SmoothSmithWaterman
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - align
        - apply

## SmithWatermanConfig

::: diffbio.operators.alignment.smith_waterman.SmithWatermanConfig
    options:
      show_root_heading: true
      members: []

## AlignmentResult

::: diffbio.operators.alignment.smith_waterman.AlignmentResult
    options:
      show_root_heading: true
      members: []

## Scoring Matrices

### create_dna_scoring_matrix

::: diffbio.operators.alignment.scoring.create_dna_scoring_matrix
    options:
      show_root_heading: true

### create_rna_scoring_matrix

::: diffbio.operators.alignment.scoring.create_rna_scoring_matrix
    options:
      show_root_heading: true

### ScoringMatrix

::: diffbio.operators.alignment.scoring.ScoringMatrix
    options:
      show_root_heading: true

### Pre-defined Matrices

```python
from diffbio.operators.alignment import (
    DNA_SIMPLE,         # 4x4 DNA scoring matrix
    RNA_SIMPLE,         # 4x4 RNA scoring matrix
    BLOSUM62,           # 20x20 protein substitution matrix
    PROTEIN_ALPHABET,   # "ARNDCQEGHILKMFPSTWYV"
)
```

## Usage Examples

### Basic Alignment

```python
import jax.numpy as jnp
from diffbio.operators.alignment import (
    SmoothSmithWaterman,
    SmithWatermanConfig,
    create_dna_scoring_matrix,
)

# Setup
config = SmithWatermanConfig(temperature=1.0, gap_open=-10.0)
scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
aligner = SmoothSmithWaterman(config, scoring_matrix=scoring)

# One-hot encode sequences
seq1 = jnp.eye(4)[jnp.array([0, 1, 2, 3])]  # ACGT
seq2 = jnp.eye(4)[jnp.array([0, 1, 0, 3])]  # ACAT

# Align
result = aligner.align(seq1, seq2)
print(f"Score: {result.score}")
```

### Datarax Interface

```python
data = {"seq1": seq1, "seq2": seq2}
result_data, state, metadata = aligner.apply(data, {}, None)
print(result_data["score"])
```

### Gradient Computation

```python
import jax

def alignment_loss(aligner, seq1, seq2):
    return -aligner.align(seq1, seq2).score

grads = jax.grad(alignment_loss)(aligner, seq1, seq2)
```
