# RNA Structure Operators API

Differentiable operators for RNA secondary structure prediction.

## DifferentiableRNAFold

::: diffbio.operators.rna_structure.rna_folding.DifferentiableRNAFold
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## RNAFoldConfig

::: diffbio.operators.rna_structure.rna_folding.RNAFoldConfig
    options:
      show_root_heading: true
      members: []

## Factory Function

### create_rna_fold_predictor

::: diffbio.operators.rna_structure.rna_folding.create_rna_fold_predictor
    options:
      show_root_heading: true

## Helper Functions

### compute_pair_energy_matrix

::: diffbio.operators.rna_structure.rna_folding.compute_pair_energy_matrix
    options:
      show_root_heading: true

### compute_base_pair_probabilities

::: diffbio.operators.rna_structure.rna_folding.compute_base_pair_probabilities
    options:
      show_root_heading: true

## Usage Examples

### Basic Usage

```python
from diffbio.operators.rna_structure import create_rna_fold_predictor
import jax
import jax.numpy as jnp

# Create predictor
predictor = create_rna_fold_predictor(temperature=1.0)

# Prepare one-hot encoded RNA sequence
sequence = jax.nn.one_hot(
    jax.random.randint(jax.random.PRNGKey(0), (50,), 0, 4),
    num_classes=4,
)

# Apply
result, _, _ = predictor.apply({"sequence": sequence}, {}, None)
bp_probs = result["bp_probs"]  # (50, 50)
```

### Full Configuration

```python
from diffbio.operators.rna_structure import (
    DifferentiableRNAFold,
    RNAFoldConfig,
)
from flax import nnx

config = RNAFoldConfig(
    temperature=0.5,       # Sharper predictions
    min_hairpin_loop=3,    # Standard hairpin constraint
    bp_energy_au=-2.0,     # A-U pair energy
    bp_energy_gc=-3.0,     # G-C pair energy
    bp_energy_gu=-1.0,     # G-U wobble energy
)

predictor = DifferentiableRNAFold(config, rngs=nnx.Rngs(42))
```

### Batched Processing

```python
import jax.numpy as jnp

# Batch of RNA sequences
batch_size = 8
seq_len = 50
sequences = jax.nn.one_hot(
    jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, 4),
    num_classes=4,
)

result, _, _ = predictor.apply({"sequence": sequences}, {}, None)
bp_probs = result["bp_probs"]  # (8, 50, 50)
```

### Gradient Computation

```python
import jax
from flax import nnx

predictor = create_rna_fold_predictor()

def loss_fn(model, sequence):
    result, _, _ = model.apply({"sequence": sequence}, {}, None)
    # Example: maximize specific base pair probability
    return -result["bp_probs"][10, 40]

# Compute gradients w.r.t. model parameters
_, grads = nnx.value_and_grad(loss_fn)(predictor, sequence)
```

## Input Specifications

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `sequence` | (length, 4) or (batch, length, 4) | float32 | One-hot encoded RNA (A=0, C=1, G=2, U=3) |

## Output Specifications

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `sequence` | same as input | float32 | Original input sequence |
| `bp_probs` | (length, length) or (batch, length, length) | float32 | Base pair probability matrix |
| `partition_function` | () or (batch,) | float32 | Log partition function |

## Base Pair Energies

| Pair | Default Energy | Description |
|------|---------------|-------------|
| A-U | -2.0 | Watson-Crick (2 H-bonds) |
| G-C | -3.0 | Watson-Crick (3 H-bonds) |
| G-U | -1.0 | Wobble pair |
