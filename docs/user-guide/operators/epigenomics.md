# Epigenomics Operators

DiffBio provides differentiable operators for epigenomic analysis, including peak calling and chromatin state annotation.

<span class="operator-epigenomics">Epigenomics</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Epigenomics operators enable end-to-end optimization of:

- **DifferentiablePeakCaller**: CNN-based peak detection for ChIP-seq/ATAC-seq
- **ChromatinStateAnnotator**: HMM-based chromatin state classification

## DifferentiablePeakCaller

CNN-based peak calling with learnable filters for ChIP-seq and ATAC-seq data.

### Quick Start

```python
from flax import nnx
from diffbio.operators.epigenomics import DifferentiablePeakCaller, PeakCallerConfig

# Configure peak caller
config = PeakCallerConfig(
    num_filters=32,
    kernel_sizes=[3, 5, 7],
    threshold=0.5,
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

# Apply to signal track
data = {"signal": signal}  # (length,) or (batch, length)
result, state, metadata = peak_caller.apply(data, {}, None)

# Get peaks
peak_scores = result["peak_scores"]      # Per-position peak probability
peak_calls = result["peak_calls"]        # Soft peak calls
boundaries = result["peak_boundaries"]   # Peak start/end positions
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_filters` | int | 32 | Number of convolutional filters |
| `kernel_sizes` | list | [3, 5, 7] | Multi-scale kernel sizes |
| `threshold` | float | 0.5 | Peak calling threshold |
| `temperature` | float | 1.0 | Temperature for soft operations |

### VAE Denoising Mode

The peak caller supports an optional VAE denoising step that preprocesses the coverage signal before peak detection. When enabled, a Poisson decoder (following the SCALE approach) models count data, and the denoised signal is used for downstream peak calling. This is configured via the `use_vae_denoising` parameter.

### Architecture

```
Input Signal → [VAE Denoising (optional)] → [Conv1D + ReLU] × 3 scales → Concat → Dense → Sigmoid → Peak Scores
                                                     ↓
                                              Multi-scale features capture peaks of different widths
```

### Training for Peak Calling

```python
import jax
from flax import nnx

def peak_loss(caller, signals, labels):
    """Binary cross-entropy for peak calling."""
    data = {"signal": signals}
    result, _, _ = caller.apply(data, {}, None)
    peak_probs = result["peak_scores"]

    # Binary cross-entropy
    loss = -jnp.mean(
        labels * jnp.log(peak_probs + 1e-8) +
        (1 - labels) * jnp.log(1 - peak_probs + 1e-8)
    )
    return loss

# Compute gradients
grads = nnx.grad(peak_loss)(peak_caller, train_signals, train_labels)
```

## ChromatinStateAnnotator

HMM-based chromatin state annotation from histone modification data.

### Quick Start

```python
from diffbio.operators.epigenomics import ChromatinStateAnnotator, ChromatinStateConfig

# Configure annotator
config = ChromatinStateConfig(
    num_states=15,      # e.g., ChromHMM default
    num_marks=6,        # Number of histone marks
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
annotator = ChromatinStateAnnotator(config, rngs=rngs)

# Apply to histone mark data
data = {"histone_marks": marks}  # (length, num_marks)
result, state, metadata = annotator.apply(data, {}, None)

# Get chromatin states
state_probs = result["state_probabilities"]   # (length, num_states)
posteriors = result["state_posteriors"]       # Forward-backward posteriors
viterbi_path = result["viterbi_path"]         # Most likely state sequence
log_likelihood = result["log_likelihood"]     # Sequence log-likelihood
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_states` | int | 15 | Number of chromatin states |
| `num_marks` | int | 6 | Number of histone marks |
| `temperature` | float | 1.0 | Temperature for soft Viterbi |

### HMM Components

The ChromatinStateAnnotator learns:

- **Transition Matrix**: State-to-state transition probabilities
- **Emission Matrix**: Mark presence probability per state
- **Initial Distribution**: Starting state probabilities

### Chromatin State Interpretation

Common chromatin states learned by the model:

| State Type | Typical Marks | Function |
|------------|---------------|----------|
| Active Promoter | H3K4me3, H3K27ac | Gene activation |
| Strong Enhancer | H3K4me1, H3K27ac | Distal regulation |
| Weak Enhancer | H3K4me1 | Poised enhancer |
| Transcribed | H3K36me3 | Actively transcribed |
| Heterochromatin | H3K9me3 | Repressed regions |
| Polycomb | H3K27me3 | Developmental silencing |

## Differentiability Techniques

### Soft Peak Boundaries

Instead of discrete peak start/end detection:

```python
# Soft peak start: rising edge detection
start_probs = jax.nn.sigmoid((scores[1:] - scores[:-1]) / temperature)

# Soft peak end: falling edge detection
end_probs = jax.nn.sigmoid((scores[:-1] - scores[1:]) / temperature)
```

### Forward-Backward Algorithm

ChromatinStateAnnotator uses log-space forward-backward:

- **Forward**: Compute P(observations up to t, state at t)
- **Backward**: Compute P(observations after t | state at t)
- **Posteriors**: Combine for P(state at t | all observations)

All operations use logsumexp for numerical stability.

## Use Cases

| Application | Operator | Description |
|-------------|----------|-------------|
| ChIP-seq peak calling | DifferentiablePeakCaller | Find protein binding sites |
| ATAC-seq analysis | DifferentiablePeakCaller | Identify open chromatin |
| Chromatin annotation | ChromatinStateAnnotator | Classify regulatory elements |
| Enhancer prediction | ChromatinStateAnnotator | Find active enhancers |

## Next Steps

- See [Statistical Operators](statistical.md) for more HMM applications
- Explore [Multi-omics Operators](multiomics.md) for Hi-C analysis
