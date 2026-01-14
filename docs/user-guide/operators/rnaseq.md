# RNA-seq Operators

DiffBio provides differentiable operators for RNA-seq analysis, including splicing quantification and motif discovery.

<span class="operator-rnaseq">RNA-seq</span> <span class="diff-high">Fully Differentiable</span>

## Overview

RNA-seq operators enable end-to-end optimization of:

- **SplicingPSI**: Percent Spliced In (PSI) calculation for alternative splicing
- **DifferentiableMotifDiscovery**: Position Weight Matrix (PWM) based motif discovery

## SplicingPSI

Differentiable PSI calculation for alternative splicing analysis.

### Quick Start

```python
from flax import nnx
from diffbio.operators.rnaseq import SplicingPSI, SplicingPSIConfig

# Configure PSI calculator
config = SplicingPSIConfig(
    temperature=1.0,
    min_coverage=1.0,
    num_exons=3,
)

# Create operator
rngs = nnx.Rngs(42)
psi_calc = SplicingPSI(config, rngs=rngs)

# Apply to junction counts
data = {
    "inclusion_counts": inclusion,   # Reads supporting exon inclusion
    "exclusion_counts": exclusion,   # Reads supporting exon skipping
}
result, state, metadata = psi_calc.apply(data, {}, None)

# Get PSI values
psi = result["psi"]                  # PSI values [0, 1]
confidence = result["confidence"]    # Confidence based on coverage
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Temperature for soft operations |
| `min_coverage` | float | 1.0 | Minimum read coverage |
| `num_exons` | int | 3 | Number of exons in event |

### PSI Calculation

PSI (Percent Spliced In) measures exon inclusion:

$$PSI = \frac{Inclusion}{Inclusion + Exclusion}$$

DiffBio's implementation uses soft division with temperature scaling for gradient flow.

### Alternative Splicing Events

| Event Type | Description | PSI Interpretation |
|------------|-------------|-------------------|
| Exon Skipping | Cassette exon | 1.0 = included, 0.0 = skipped |
| Intron Retention | Retained intron | 1.0 = retained, 0.0 = spliced |
| Alt 5' SS | Alternative 5' splice site | Relative usage |
| Alt 3' SS | Alternative 3' splice site | Relative usage |

## DifferentiableMotifDiscovery

PWM-based motif discovery with learnable position weight matrices.

### Quick Start

```python
from diffbio.operators.rnaseq import DifferentiableMotifDiscovery, MotifDiscoveryConfig

# Configure motif discovery
config = MotifDiscoveryConfig(
    num_motifs=10,
    motif_length=8,
    alphabet_size=4,
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
motif_finder = DifferentiableMotifDiscovery(config, rngs=rngs)

# Apply to sequences
data = {"sequences": sequences}  # (n_seqs, seq_len, alphabet_size)
result, state, metadata = motif_finder.apply(data, {}, None)

# Get results
pwms = result["pwms"]              # Learned PWMs (num_motifs, motif_len, alphabet)
scores = result["motif_scores"]    # Per-position scores
best_motif = result["best_motif"]  # Best matching motif per position
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_motifs` | int | 10 | Number of motifs to discover |
| `motif_length` | int | 8 | Length of each motif PWM |
| `alphabet_size` | int | 4 | Alphabet size (4 for DNA) |
| `temperature` | float | 1.0 | Temperature for soft max |

### PWM Scoring

Position Weight Matrix scoring:

$$Score(s, p) = \sum_{i=0}^{L-1} \log \frac{PWM[i, s_{p+i}]}{background[s_{p+i}]}$$

DiffBio uses soft PWM matching with temperature-controlled softmax.

### Training for Motif Discovery

```python
import jax
from flax import nnx

def motif_loss(finder, sequences, labels):
    """Train motif finder with sequence labels."""
    data = {"sequences": sequences}
    result, _, _ = finder.apply(data, {}, None)

    # Use motif scores for classification
    scores = result["motif_scores"]
    logits = scores.max(axis=-1)  # Max over positions

    # Cross-entropy loss
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

# Train
grads = nnx.grad(motif_loss)(motif_finder, train_seqs, train_labels)
```

### Visualizing Learned Motifs

```python
import matplotlib.pyplot as plt
import logomaker

# Extract PWMs from trained model
pwms = motif_finder.pwms.value

# Convert to DataFrame for logomaker
for i, pwm in enumerate(pwms):
    df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])

    # Create sequence logo
    logo = logomaker.Logo(df, shade_below=0.5)
    plt.title(f'Motif {i+1}')
    plt.show()
```

## Differentiability Techniques

### Soft PSI

Standard PSI is non-differentiable at boundaries. DiffBio uses:

```python
# Soft PSI with numerical stability
psi = inclusion / (inclusion + exclusion + epsilon)

# Temperature-scaled for sharper gradients during training
psi = jax.nn.sigmoid((inclusion - exclusion) / temperature)
```

### Soft PWM Matching

Instead of argmax for best position:

```python
# Soft position selection via attention
position_weights = jax.nn.softmax(scores / temperature, axis=-1)
soft_position = jnp.sum(position_weights * positions, axis=-1)
```

## Use Cases

| Application | Operator | Description |
|-------------|----------|-------------|
| Splicing analysis | SplicingPSI | Quantify alternative splicing |
| Splice site prediction | DifferentiableMotifDiscovery | Learn splice site motifs |
| RBP binding | DifferentiableMotifDiscovery | Discover binding motifs |
| Regulatory elements | DifferentiableMotifDiscovery | Find enhancer/silencer motifs |

## Next Steps

- See [Statistical Operators](statistical.md) for differential expression
- Explore [Single-Cell Operators](singlecell.md) for scRNA-seq analysis
