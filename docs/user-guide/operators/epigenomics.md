# Epigenomics Operators

DiffBio provides differentiable operators for epigenomic analysis, including
peak calling, chromatin state annotation, and contextual sequence models that
combine sequence, TF context, and chromatin contacts.

<span class="operator-epigenomics">Epigenomics</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Epigenomics operators enable end-to-end optimization of:

- **DifferentiablePeakCaller**: CNN-based peak detection for ChIP-seq/ATAC-seq
- **ChromatinStateAnnotator**: HMM-based chromatin state classification
- **ContextualEpigenomicsOperator**: Artifex-transformer-based contextual
  sequence model with TF conditioning and optional chromatin-guidance loss

## ContextualEpigenomicsOperator

Configurable contextual epigenomics operator that supports three ablation modes
from one code path:

- sequence-only
- sequence plus TF context
- sequence plus TF context plus chromatin-guidance loss

The operator reuses Artifex's `TransformerEncoder` for sequence modeling and
adds a structured chromatin-consistency term instead of introducing a separate
local attention stack for contact guidance.

### Quick Start

```python
from flax import nnx

from diffbio.operators.epigenomics import (
    ContextualEpigenomicsConfig,
    ContextualEpigenomicsOperator,
    compute_contextual_epigenomics_loss,
)
from diffbio.sources import build_synthetic_contextual_epigenomics_dataset

data = build_synthetic_contextual_epigenomics_dataset(
    n_examples=4,
    sequence_length=24,
    num_tf_features=3,
    target_semantics="binary_peak_mask",
    num_output_classes=1,
)

config = ContextualEpigenomicsConfig(
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
    intermediate_dim=256,
    max_length=24,
    num_tf_features=3,
    num_outputs=1,
    use_tf_context=True,
    use_chromatin_guidance=True,
    chromatin_guidance_weight=0.1,
)
operator = ContextualEpigenomicsOperator(config, rngs=nnx.Rngs(42))

result, state, metadata = operator.apply(data, {}, None)
losses = compute_contextual_epigenomics_loss(operator, data)

logits = result["logits"]                         # (batch, length)
token_embeddings = result["token_embeddings"]     # (batch, length, hidden_dim)
guidance_loss = result["chromatin_guidance_loss"] # scalar
total_loss = losses["total"]
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | int | 64 | Token embedding width |
| `num_layers` | int | 2 | Number of Artifex transformer layers |
| `num_heads` | int | 4 | Number of attention heads |
| `intermediate_dim` | int | 256 | Feed-forward hidden dimension |
| `max_length` | int | 512 | Maximum supported sequence length |
| `num_tf_features` | int | 8 | TF-context feature count |
| `num_outputs` | int | 1 | Output channels per genomic position |
| `use_tf_context` | bool | True | Enable TF-conditioning path |
| `use_chromatin_guidance` | bool | False | Enable chromatin-consistency term |
| `chromatin_guidance_weight` | float | 0.1 | Weight on chromatin guidance |

### Inputs And Outputs

Expected data keys:

- `sequence`: one-hot sequence tensor `(batch, length, 4)` or `(length, 4)`
- `tf_context`: TF features `(batch, n_tf_features)` or `(n_tf_features,)`
- `chromatin_contacts`: contact map `(batch, length, length)` or `(length, length)`
- `targets`: supervised targets `(batch, length)` or `(length,)`
- `sequence_mask`: optional binary mask for padded positions

Main outputs:

- `embeddings`: pooled sequence embeddings
- `token_embeddings`: per-position contextualized embeddings
- `logits`: per-position prediction logits
- `chromatin_guidance_loss`: unweighted structured contact-consistency term

### Training Notes

`compute_contextual_epigenomics_loss()` returns three values:

- `supervised`: BCE or multiclass cross-entropy depending on `num_outputs`
- `chromatin_guidance`: structured consistency term derived from token-embedding
  similarity and the supplied contact map
- `total`: `supervised + chromatin_guidance_weight * chromatin_guidance`

This keeps the three ablation modes on one implementation path instead of
forking separate operator classes.

### Benchmark Coverage

The contextual epigenomics benchmark family now evaluates the real operator in
three reproducible ablations:

- `sequence_only`
- `tf_context`
- `tf_plus_chromatin`

The quick-suite reports track downstream task metrics plus a bounded
chromatin-consistency score derived from the structured guidance loss. This is
the supported benchmark path for comparing whether TF conditioning improves the
task metric and whether chromatin guidance improves structural consistency
without introducing a second operator implementation.

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

```mermaid
graph LR
    A["Input Signal"] --> B["VAE Denoising<br/>(optional)"]
    B --> C["[Conv1D + ReLU]<br/>× 3 scales"]
    C --> D["Concat"]
    D --> E["Dense"]
    E --> F["Sigmoid"]
    F --> G["Peak Scores"]

    style A fill:#d1fae5,stroke:#059669,color:#064e3b
    style B fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    style C fill:#e0e7ff,stroke:#4338ca,color:#312e81
    style D fill:#e0e7ff,stroke:#4338ca,color:#312e81
    style E fill:#e0e7ff,stroke:#4338ca,color:#312e81
    style F fill:#e0e7ff,stroke:#4338ca,color:#312e81
    style G fill:#d1fae5,stroke:#059669,color:#064e3b
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
| Context-aware regulatory modeling | ContextualEpigenomicsOperator | Combine sequence, TF context, and chromatin guidance |

## Next Steps

- See [Statistical Operators](statistical.md) for more HMM applications
- Explore [Multi-omics Operators](multiomics.md) for Hi-C analysis
