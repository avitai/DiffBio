# Variant Calling Pipeline API

End-to-end differentiable variant calling pipeline.

## VariantCallingPipeline

::: diffbio.pipelines.variant_calling.VariantCallingPipeline
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply
        - set_training
        - train_mode
        - eval_mode
        - call_variants

## VariantCallingPipelineConfig

::: diffbio.pipelines.variant_calling.VariantCallingPipelineConfig
    options:
      show_root_heading: true
      members: []

## Factory Function

### create_variant_calling_pipeline

::: diffbio.pipelines.variant_calling.create_variant_calling_pipeline
    options:
      show_root_heading: true

## Usage Examples

### Quick Start

```python
from diffbio.pipelines import create_variant_calling_pipeline
import jax
import jax.numpy as jnp

# Create pipeline
pipeline = create_variant_calling_pipeline(
    reference_length=100,
    num_classes=3,
)

# Prepare data
data = {
    "reads": jax.nn.softmax(
        jax.random.uniform(jax.random.PRNGKey(0), (20, 30, 4)),
        axis=-1
    ),
    "positions": jax.random.randint(jax.random.PRNGKey(1), (20,), 0, 70),
    "quality": jax.random.uniform(jax.random.PRNGKey(2), (20, 30), minval=10, maxval=40),
}

# Run pipeline
result, _, _ = pipeline.apply(data, {}, None)
predictions = jnp.argmax(result["probabilities"], axis=-1)
```

### Full Configuration

```python
from diffbio.pipelines import VariantCallingPipeline, VariantCallingPipelineConfig
from flax import nnx

config = VariantCallingPipelineConfig(
    reference_length=10000,
    num_classes=3,
    quality_threshold=20.0,
    pileup_window_size=21,
    classifier_hidden_dim=128,
    use_quality_weights=True,
)

pipeline = VariantCallingPipeline(config, rngs=nnx.Rngs(42))
pipeline.eval_mode()
```

### Training Mode

```python
# Enable dropout
pipeline.train_mode()

# Training loop
for batch in dataloader:
    loss = train_step(pipeline, batch)

# Disable dropout for inference
pipeline.eval_mode()
```

### Access Components

```python
# Quality filter threshold
pipeline.quality_filter.threshold[...]

# Pileup temperature
pipeline.pileup.config.temperature

# Classifier network
pipeline.classifier
```

## Input Specifications

| Key | Shape | Description |
|-----|-------|-------------|
| `reads` | (num_reads, read_length, 4) | One-hot encoded reads |
| `positions` | (num_reads,) | Read start positions |
| `quality` | (num_reads, read_length) | Phred quality scores |

## Output Specifications

| Key | Shape | Description |
|-----|-------|-------------|
| `reads` | (num_reads, read_length, 4) | Original reads |
| `positions` | (num_reads,) | Original positions |
| `quality` | (num_reads, read_length) | Original quality |
| `filtered_reads` | (num_reads, read_length, 4) | Quality-filtered reads |
| `filtered_quality` | (num_reads, read_length) | Filtered quality scores |
| `pileup` | (reference_length, 4) | Aggregated pileup |
| `logits` | (reference_length, num_classes) | Raw predictions |
| `probabilities` | (reference_length, num_classes) | Class probabilities |
