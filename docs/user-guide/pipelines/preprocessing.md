# Preprocessing Pipeline

The `PreprocessingPipeline` chains multiple preprocessing operators for end-to-end differentiable read preprocessing.

<span class="pipeline-preprocessing">Preprocessing</span> <span class="diff-high">Fully Differentiable</span>

## Overview

The preprocessing pipeline combines:

1. **Quality Filtering**: Soft quality-based read weighting
2. **Adapter Removal**: Differentiable adapter trimming
3. **Duplicate Weighting**: Probabilistic PCR duplicate handling
4. **Error Correction**: Neural network-based error correction

## Quick Start

```python
from flax import nnx
from diffbio.pipelines import PreprocessingPipeline, PreprocessingPipelineConfig

# Configure pipeline
config = PreprocessingPipelineConfig(
    read_length=150,
    quality_threshold=20.0,
    adapter_sequence="AGATCGGAAGAG",
    enable_adapter_removal=True,
    enable_duplicate_weighting=True,
    enable_error_correction=True,
)

# Create pipeline
rngs = nnx.Rngs(42)
pipeline = PreprocessingPipeline(config, rngs=rngs)

# Process reads
data = {
    "reads": read_sequences,    # (n_reads, read_length, 4)
    "quality": quality_scores,  # (n_reads, read_length)
}
result, state, metadata = pipeline.apply(data, {}, None)

# Get preprocessed data
preprocessed = result["preprocessed_reads"]  # Cleaned reads
weights = result["read_weights"]              # Per-read weights
```

## Configuration

### PreprocessingPipelineConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `read_length` | int | 150 | Expected read length for initialization |
| `adapter_sequence` | str | `"AGATCGGAAGAG"` | Adapter sequence to remove (Illumina universal default) |
| `quality_threshold` | float | 20.0 | Initial quality score threshold for filtering |
| `adapter_match_threshold` | float | 0.8 | Threshold for adapter matching |
| `adapter_temperature` | float | 1.0 | Temperature for soft adapter trimming |
| `duplicate_similarity_threshold` | float | 0.95 | Similarity threshold for duplicate detection |
| `error_correction_window` | int | 11 | Window size for error correction |
| `error_correction_hidden_dim` | int | 64 | Hidden dimension for error correction network |
| `enable_adapter_removal` | bool | True | Enable adapter trimming |
| `enable_duplicate_weighting` | bool | True | Enable duplicate detection |
| `enable_error_correction` | bool | True | Enable error correction |

### Detailed Configuration

```python
config = PreprocessingPipelineConfig(
    # General
    read_length=150,

    # Quality filtering
    quality_threshold=20.0,

    # Adapter removal
    enable_adapter_removal=True,
    adapter_sequence="AGATCGGAAGAG",
    adapter_match_threshold=0.8,
    adapter_temperature=1.0,

    # Duplicate handling
    enable_duplicate_weighting=True,
    duplicate_similarity_threshold=0.95,

    # Error correction
    enable_error_correction=True,
    error_correction_window=11,
    error_correction_hidden_dim=64,
)
```

## Pipeline Stages

### Stage 1: Quality Filtering

Applies soft quality-based filtering using sigmoid weights:

```python
# Per-read quality weight
mean_quality = jnp.mean(quality_scores, axis=-1)
weights = jax.nn.sigmoid((mean_quality - threshold) / temperature)
```

Reads with low average quality get lower weights, but all reads contribute to gradient computation.

### Stage 2: Adapter Removal

Finds and softly trims adapter sequences:

```python
# For each read, compute adapter match scores
match_scores = soft_align(read, adapter_sequence)

# Soft trim position (returns probability distribution over positions)
trim_pos = soft_ops.argmax(match_scores, axis=-1, softness=0.1)

# Apply soft mask
trimmed = read * soft_mask(positions, trim_pos)
```

### Stage 3: Duplicate Weighting

Identifies potential PCR duplicates using learned embeddings:

```python
# Embed reads
embeddings = read_encoder(reads)

# Compute pairwise similarity
similarity = cosine_similarity(embeddings, embeddings.T)

# Weight based on uniqueness
weights = 1.0 - max_similarity_to_others
```

### Stage 4: Error Correction

Neural network predicts corrected base probabilities:

```python
# Context-aware correction
for position in read:
    context = read[pos-5:pos+6]  # Context window
    correction = correction_network(context, quality[pos])
    corrected[pos] = mix(read[pos], correction, quality_weight)
```

## Training

### Loss Function

```python
from flax import nnx

def preprocessing_loss(pipeline, reads, quality, corrected_targets):
    """Train preprocessing pipeline end-to-end."""
    data = {"reads": reads, "quality": quality}
    result, _, _ = pipeline.apply(data, {}, None)

    # Reconstruction loss
    recon_loss = jnp.mean(
        (result["preprocessed_reads"] - corrected_targets) ** 2
    )

    return recon_loss

# Compute gradients through entire pipeline
grads = nnx.grad(preprocessing_loss)(
    pipeline, train_reads, train_quality, train_corrected
)
```

### Training Loop

```python
import optax

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(nnx.state(pipeline, nnx.Param))

@jax.jit
def train_step(pipeline, batch, opt_state):
    def loss_fn(pipe):
        result, _, _ = pipe.apply(batch, {}, None)
        return jnp.mean((result["preprocessed_reads"] - batch["target"]) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(pipeline)

    params = nnx.state(pipeline, nnx.Param)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    nnx.update(pipeline, optax.apply_updates(params, updates))

    return loss, opt_state

# Train
for epoch in range(100):
    for batch in data_loader:
        loss, opt_state = train_step(pipeline, batch, opt_state)
```

## Selective Stages

Enable only specific preprocessing stages:

```python
# Only quality filtering
config = PreprocessingPipelineConfig(
    enable_adapter_removal=False,
    enable_duplicate_weighting=False,
    enable_error_correction=False,
)

# Quality + Error correction (no adapter/duplicate handling)
config = PreprocessingPipelineConfig(
    enable_adapter_removal=False,
    enable_duplicate_weighting=False,
    enable_error_correction=True,
)
```

## Integration with Other Pipelines

Chain preprocessing with downstream analysis:

```python
from diffbio.pipelines import PreprocessingPipeline, VariantCallingPipeline

# Create pipelines
preprocess = PreprocessingPipeline(preprocess_config, rngs=rngs)
variant_caller = VariantCallingPipeline(variant_config, rngs=rngs)

# Chain execution
def full_pipeline(data):
    # Preprocess
    prep_result, _, _ = preprocess.apply(data, {}, None)

    # Variant calling
    variant_data = {
        "reads": prep_result["preprocessed_reads"],
        "positions": data["positions"],
        "quality": data["quality"],
    }
    var_result, _, _ = variant_caller.apply(variant_data, {}, None)

    return var_result

# End-to-end gradient
grads = jax.grad(lambda d: full_pipeline(d)["logits"].sum())(data)
```

## Performance Tips

### JIT Compilation

```python
@jax.jit
def process_batch(pipeline, reads, quality):
    data = {"reads": reads, "quality": quality}
    result, _, _ = pipeline.apply(data, {}, None)
    return result["preprocessed_reads"], result["read_weights"]

# Fast batch processing
for batch in batches:
    preprocessed, weights = process_batch(pipeline, batch["reads"], batch["quality"])
```

### Memory Optimization

For large datasets:

```python
# Process in chunks
chunk_size = 1000
for i in range(0, len(reads), chunk_size):
    chunk_reads = reads[i:i+chunk_size]
    chunk_quality = quality[i:i+chunk_size]
    result, _, _ = pipeline.apply(
        {"reads": chunk_reads, "quality": chunk_quality},
        {}, None
    )
```

## Next Steps

- See [Preprocessing Operators](../operators/preprocessing.md) for individual components
- Explore [Variant Calling Pipeline](variant-calling.md) for downstream analysis
- Check [Training Overview](../training/overview.md) for training guidance
