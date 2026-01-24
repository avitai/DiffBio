# RNA Structure Operators

DiffBio provides differentiable operators for RNA secondary structure prediction, following the McCaskill partition function algorithm for computing base pair probabilities.

<span class="operator-rna">RNA Structure</span> <span class="diff-high">Fully Differentiable</span>

## Overview

RNA structure operators enable gradient-based optimization for RNA design and analysis:

- **DifferentiableRNAFold**: McCaskill-style partition function for base pair probabilities

## DifferentiableRNAFold

Differentiable implementation of the McCaskill partition function algorithm for computing RNA base pair probabilities.

### Algorithm

The McCaskill algorithm (1990) computes the partition function Z = Σ_P exp(-E(P)/RT) over all possible secondary structures:

```
One-Hot RNA Sequence (length, 4)
         │
         ▼
┌─────────────────────────────┐
│ Compute Base Pair Energies  │
│ A-U: -2.0 (2 H-bonds)       │
│ G-C: -3.0 (3 H-bonds)       │
│ G-U: -1.0 (wobble)          │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Boltzmann Weights           │
│ w(i,j) = exp(-E(i,j)/T)     │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Partition Function          │
│ Z = Σ w(i,j) over valid     │
│     base pairs              │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Base Pair Probabilities     │
│ P(i,j) = w(i,j) / Z         │
└─────────────────────────────┘
         │
         ▼
    BP Probability Matrix
       (length x length)
```

### Quick Start

```python
from flax import nnx
import jax
import jax.numpy as jnp
from diffbio.operators.rna_structure import (
    DifferentiableRNAFold,
    RNAFoldConfig,
    create_rna_fold_predictor,
)

# Create predictor
predictor = create_rna_fold_predictor(
    temperature=1.0,      # Boltzmann temperature
    min_hairpin_loop=3,   # Minimum unpaired bases in hairpin
)

# Prepare one-hot encoded RNA sequence
# A=0, C=1, G=2, U=3
seq_len = 50
seq_indices = jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4)
sequence = jax.nn.one_hot(seq_indices, num_classes=4)

# Apply predictor
data = {"sequence": sequence}
result, state, metadata = predictor.apply(data, {}, None)

# Get results
bp_probs = result["bp_probs"]           # (50, 50) base pair probabilities
log_z = result["partition_function"]    # Log partition function
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Boltzmann temperature (RT). Lower = sharper probabilities |
| `min_hairpin_loop` | int | 3 | Minimum unpaired nucleotides in hairpin loop |
| `alphabet_size` | int | 4 | Nucleotide alphabet size (A, C, G, U) |
| `bp_energy_au` | float | -2.0 | Energy for A-U base pair |
| `bp_energy_gc` | float | -3.0 | Energy for G-C base pair |
| `bp_energy_gu` | float | -1.0 | Energy for G-U wobble pair |

### Input/Output Formats

**Input**

| Key | Shape | Description |
|-----|-------|-------------|
| `sequence` | (length, 4) or (batch, length, 4) | One-hot encoded RNA sequence (A=0, C=1, G=2, U=3) |

**Output**

| Key | Shape | Description |
|-----|-------|-------------|
| `sequence` | same as input | Original input sequence |
| `bp_probs` | (length, length) or (batch, length, length) | Base pair probability matrix |
| `partition_function` | () or (batch,) | Log partition function |

### Base Pair Probability Matrix

The output `bp_probs[i,j]` gives the probability that nucleotides at positions i and j form a base pair in the thermodynamic ensemble:

- **Symmetric**: `bp_probs[i,j] = bp_probs[j,i]`
- **Diagonal zero**: `bp_probs[i,i] = 0` (can't pair with self)
- **Hairpin constraint**: `bp_probs[i,j] = 0` if `|i-j| <= min_hairpin_loop`
- **Valid pairs only**: Non-zero only for A-U, G-C, G-U pairs

### Base Pairing Rules

Watson-Crick and wobble base pairs:

| Pair | Energy | H-bonds | Description |
|------|--------|---------|-------------|
| G-C | -3.0 | 3 | Strongest pair |
| A-U | -2.0 | 2 | Standard pair |
| G-U | -1.0 | 2 | Wobble pair (weaker) |

### Temperature Effect

The temperature parameter controls the sharpness of base pair probabilities:

```python
# Low temperature = sharper predictions
predictor_low = create_rna_fold_predictor(temperature=0.1)

# High temperature = more uniform predictions
predictor_high = create_rna_fold_predictor(temperature=5.0)
```

- **Low temperature (< 1)**: Probabilities concentrated on most stable pairs
- **High temperature (> 1)**: More uniform distribution over all valid pairs
- **Unit temperature (= 1)**: Standard Boltzmann distribution

### Training Example

```python
import optax
from flax import nnx

predictor = create_rna_fold_predictor(temperature=1.0)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(nnx.state(predictor, nnx.Param))

def loss_fn(model, sequence, target_bp):
    """MSE loss for target base pair probabilities."""
    data = {"sequence": sequence}
    result, _, _ = model.apply(data, {}, None)
    return jnp.mean((result["bp_probs"] - target_bp) ** 2)

@nnx.jit
def train_step(model, opt_state, sequence, target):
    loss, grads = nnx.value_and_grad(loss_fn)(model, sequence, target)
    params = nnx.state(model, nnx.Param)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    return loss, opt_state
```

### Gradient Flow for RNA Design

Use soft (probabilistic) sequences for gradient-based optimization:

```python
# Initialize soft sequence
logits = jax.random.normal(jax.random.PRNGKey(0), (seq_len, 4))
soft_sequence = jax.nn.softmax(logits, axis=-1)

def design_loss(seq):
    """Loss for designing sequences with target structure."""
    result, _, _ = predictor.apply({"sequence": seq}, {}, None)
    # Maximize probability of target base pairs
    target_pairs = [...]  # Define target (i,j) pairs
    loss = 0.0
    for i, j in target_pairs:
        loss -= result["bp_probs"][i, j]
    return loss

# Compute gradients w.r.t. sequence
grads = jax.grad(design_loss)(soft_sequence)
```

### Visualizing Base Pair Probabilities

```python
import matplotlib.pyplot as plt

result, _, _ = predictor.apply({"sequence": sequence}, {}, None)
bp_probs = result["bp_probs"]

plt.figure(figsize=(8, 8))
plt.imshow(bp_probs, cmap='viridis', origin='lower')
plt.colorbar(label='Base Pair Probability')
plt.xlabel('Position j')
plt.ylabel('Position i')
plt.title('RNA Base Pair Probability Matrix')
```

## Use Cases

| Application | Description |
|-------------|-------------|
| RNA design | Optimize sequences for target secondary structures |
| Structure prediction | Compute ensemble-averaged structure properties |
| Riboswitch design | Design RNA switches with specific folding behavior |
| mRNA optimization | Improve mRNA stability through structure design |
| Aptamer engineering | Design RNA aptamers with desired binding properties |

## References

1. McCaskill, J. S. (1990). "The equilibrium partition function and base pair binding probabilities for RNA secondary structure." *Biopolymers* 29, 1105-1119.

2. Matthies, M. C. et al. (2024). "Differentiable partition function calculation for RNA." *Nucleic Acids Research* 52(3), e14.

3. Krueger, R. et al. (2025). "JAX-RNAfold: Scalable differentiable folding." *Bioinformatics* 41(5), btaf203.

## Next Steps

- See [Language Model Operators](language-models.md) for sequence embedding
- Explore [Statistical Operators](statistical.md) for related analysis methods
- Check [RNA-seq Operators](rnaseq.md) for transcriptomics analysis
