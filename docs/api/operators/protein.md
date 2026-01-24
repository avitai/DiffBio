# Protein Structure Operators API

Differentiable operators for protein structure analysis.

## DifferentiableSecondaryStructure

::: diffbio.operators.protein.secondary_structure.DifferentiableSecondaryStructure
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply
        - compute_hbond_energy
        - compute_hbond_map
        - detect_helix_pattern
        - detect_strand_pattern
        - assign_secondary_structure

## SecondaryStructureConfig

::: diffbio.operators.protein.secondary_structure.SecondaryStructureConfig
    options:
      show_root_heading: true
      members: []

## Factory Function

### create_secondary_structure_predictor

::: diffbio.operators.protein.secondary_structure.create_secondary_structure_predictor
    options:
      show_root_heading: true

## Helper Functions

### compute_hydrogen_position

::: diffbio.operators.protein.secondary_structure.compute_hydrogen_position
    options:
      show_root_heading: true

## Usage Examples

### Basic Usage

```python
from diffbio.operators.protein import create_secondary_structure_predictor
import jax
import jax.numpy as jnp

# Create predictor
predictor = create_secondary_structure_predictor()

# Prepare coordinates (batch, n_residues, 4 atoms, xyz)
coords = jax.random.uniform(jax.random.PRNGKey(0), (1, 50, 4, 3)) * 10

# Apply
result, _, _ = predictor.apply({"coordinates": coords}, {}, None)
ss_probs = result["ss_onehot"]  # (1, 50, 3)
```

### Full Configuration

```python
from diffbio.operators.protein import (
    DifferentiableSecondaryStructure,
    SecondaryStructureConfig,
)
from flax import nnx

config = SecondaryStructureConfig(
    margin=1.0,
    cutoff=-0.5,
    min_helix_length=4,
    temperature=0.5,  # Sharper assignments
)

predictor = DifferentiableSecondaryStructure(config, rngs=nnx.Rngs(42))
```

### Gradient Computation

```python
import jax

def loss_fn(coords):
    result, _, _ = predictor.apply({"coordinates": coords}, {}, None)
    # Maximize helix content
    helix_prob = result["ss_onehot"][:, :, 1]  # Index 1 = helix
    return -helix_prob.mean()

# Compute gradients w.r.t. coordinates
grads = jax.grad(loss_fn)(coords)
```

## Input Specifications

| Key | Shape | Description |
|-----|-------|-------------|
| `coordinates` | (batch, length, 4, 3) | Backbone atoms (N, CA, C, O) in Angstroms |

## Output Specifications

| Key | Shape | Description |
|-----|-------|-------------|
| `coordinates` | (batch, length, 4, 3) | Original input coordinates |
| `ss_onehot` | (batch, length, 3) | Soft SS probabilities |
| `ss_indices` | (batch, length) | Hard SS assignments |
| `hbond_map` | (batch, length, length) | Continuous H-bond matrix |

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CONST_Q1Q2` | 0.084 | Partial charge product |
| `CONST_F` | 332.0 | Conversion to kcal/mol |
| `DEFAULT_CUTOFF` | -0.5 | H-bond energy threshold |
| `DEFAULT_MARGIN` | 1.0 | Smoothing margin |
