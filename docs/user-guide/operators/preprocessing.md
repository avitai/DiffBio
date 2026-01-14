# Preprocessing Operators

DiffBio provides differentiable preprocessing operators for read quality control, adapter removal, and error correction.

<span class="operator-preprocessing">Preprocessing</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Preprocessing operators enable end-to-end optimization of:

- **SoftAdapterRemoval**: Differentiable adapter trimming with soft alignment
- **DifferentiableDuplicateWeighting**: Probabilistic duplicate detection and weighting
- **SoftErrorCorrection**: Neural network-based sequencing error correction

## SoftAdapterRemoval

Differentiable adapter trimming using soft sequence alignment.

### Quick Start

```python
from flax import nnx
from diffbio.operators.preprocessing import SoftAdapterRemoval, AdapterRemovalConfig

# Configure adapter removal
config = AdapterRemovalConfig(
    adapter_length=20,
    alphabet_size=4,
    temperature=1.0,
    min_match_score=0.8,
)

# Create operator
rngs = nnx.Rngs(42)
adapter_removal = SoftAdapterRemoval(config, rngs=rngs)

# Apply to reads
data = {
    "sequence": read_sequence,    # (read_length, alphabet_size)
    "quality_scores": quality,    # (read_length,)
}
result, state, metadata = adapter_removal.apply(data, {}, None)

# Get trimmed sequence
trimmed_seq = result["sequence"]
trimmed_quality = result["quality_scores"]
trim_position = result["trim_position"]
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adapter_length` | int | 20 | Length of adapter sequence |
| `alphabet_size` | int | 4 | Alphabet size (4 for DNA) |
| `temperature` | float | 1.0 | Temperature for soft matching |
| `min_match_score` | float | 0.8 | Minimum match threshold |

### Soft Trimming

Instead of hard trimming at a discrete position:

```python
# Soft trim position via weighted average
position_weights = jax.nn.softmax(match_scores / temperature)
soft_trim = jnp.sum(position_weights * positions)

# Soft mask applied to sequence
mask = jax.nn.sigmoid((positions - soft_trim) / temperature)
trimmed = sequence * mask[..., None]
```

## DifferentiableDuplicateWeighting

Probabilistic duplicate detection and weighting for PCR duplicate handling.

### Quick Start

```python
from diffbio.operators.preprocessing import (
    DifferentiableDuplicateWeighting,
    DuplicateWeightingConfig,
)

# Configure duplicate weighting
config = DuplicateWeightingConfig(
    embedding_dim=32,
    similarity_threshold=0.9,
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
dup_weighting = DifferentiableDuplicateWeighting(config, rngs=rngs)

# Apply to batch of reads
data = {
    "sequences": read_batch,     # (n_reads, read_length, alphabet_size)
    "quality_scores": quality,   # (n_reads, read_length)
}
result, state, metadata = dup_weighting.apply(data, {}, None)

# Get duplicate weights
weights = result["weights"]              # (n_reads,) weights in [0, 1]
similarity_matrix = result["similarity"] # Pairwise similarities
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | int | 32 | Read embedding dimension |
| `similarity_threshold` | float | 0.9 | Duplicate similarity threshold |
| `temperature` | float | 1.0 | Softmax temperature |

### Duplicate Detection

Uses learned sequence embeddings for soft duplicate detection:

1. **Embed reads**: Neural network encodes reads to fixed-size vectors
2. **Compute similarity**: Cosine similarity between all pairs
3. **Weight calculation**: Higher weight for unique reads, lower for duplicates

## SoftErrorCorrection

Neural network-based sequencing error correction with soft base substitution.

### Quick Start

```python
from diffbio.operators.preprocessing import SoftErrorCorrection, ErrorCorrectionConfig

# Configure error correction
config = ErrorCorrectionConfig(
    alphabet_size=4,
    hidden_dim=64,
    context_window=5,
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
error_correction = SoftErrorCorrection(config, rngs=rngs)

# Apply to sequence
data = {
    "sequence": read_sequence,    # (read_length, alphabet_size)
    "quality_scores": quality,    # (read_length,)
}
result, state, metadata = error_correction.apply(data, {}, None)

# Get corrected sequence
corrected = result["sequence"]
correction_probs = result["correction_probabilities"]
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alphabet_size` | int | 4 | Alphabet size (4 for DNA) |
| `hidden_dim` | int | 64 | Network hidden dimension |
| `context_window` | int | 5 | Context bases on each side |
| `temperature` | float | 1.0 | Correction sharpness |

### Error Correction Model

```
Context Window → Convolution → Dense → Corrected Base Probabilities
     [--5--|X|--5--]    →    →    →    P(A), P(C), P(G), P(T)
```

The model uses quality scores to weight corrections:
- Low quality positions: more likely to be corrected
- High quality positions: corrections are suppressed

## Using the Preprocessing Pipeline

All preprocessing operators can be combined into a pipeline:

```python
from diffbio.pipelines import PreprocessingPipeline, PreprocessingPipelineConfig

# Configure full pipeline
config = PreprocessingPipelineConfig(
    enable_adapter_removal=True,
    enable_duplicate_weighting=True,
    enable_error_correction=True,
    quality_threshold=20.0,
)

# Create pipeline
rngs = nnx.Rngs(42)
pipeline = PreprocessingPipeline(config, rngs=rngs)

# Process reads
data = {
    "reads": reads,      # (n_reads, read_length, alphabet_size)
    "quality": quality,  # (n_reads, read_length)
}
result, state, metadata = pipeline.apply(data, {}, None)

preprocessed = result["preprocessed_reads"]
read_weights = result["read_weights"]
```

## Training Example

```python
from flax import nnx

def preprocessing_loss(pipeline, reads, quality, corrected_target):
    """Train preprocessing to match corrected reads."""
    data = {"reads": reads, "quality": quality}
    result, _, _ = pipeline.apply(data, {}, None)

    # Reconstruction loss
    loss = jnp.mean((result["preprocessed_reads"] - corrected_target) ** 2)
    return loss

# Compute gradients
grads = nnx.grad(preprocessing_loss)(
    pipeline, train_reads, train_quality, train_corrected
)
```

## Use Cases

| Application | Operator | Description |
|-------------|----------|-------------|
| Adapter trimming | SoftAdapterRemoval | Remove sequencing adapters |
| PCR duplicate handling | DifferentiableDuplicateWeighting | Weight down duplicates |
| Error correction | SoftErrorCorrection | Correct sequencing errors |
| Full preprocessing | PreprocessingPipeline | Combined preprocessing |

## Next Steps

- See [Preprocessing Pipeline](../pipelines/preprocessing.md) for the full pipeline
- Explore [Quality Filter](quality-filter.md) for quality-based filtering
