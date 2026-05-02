# Enhanced Variant Calling Pipeline API

DeepVariant-style end-to-end differentiable variant calling pipeline with CNN classifier and quality recalibration.

## EnhancedVariantCallingPipeline

::: diffbio.pipelines.enhanced_variant_calling.EnhancedVariantCallingPipeline
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## EnhancedVariantCallingPipelineConfig

::: diffbio.pipelines.enhanced_variant_calling.EnhancedVariantCallingPipelineConfig
    options:
      show_root_heading: true
      members: []

## Factory Function

### create_enhanced_variant_calling_pipeline

::: diffbio.pipelines.enhanced_variant_calling.create_enhanced_variant_calling_pipeline
    options:
      show_root_heading: true

## Usage Examples

### Quick Start

```python
from diffbio.pipelines import create_enhanced_variant_calling_pipeline
import jax
import jax.numpy as jnp

# Create pipeline
pipeline = create_enhanced_variant_calling_pipeline(
    reference_length=100,
    num_classes=3,
    pileup_window_size=11,
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
from diffbio.pipelines import EnhancedVariantCallingPipeline, EnhancedVariantCallingPipelineConfig
from flax import nnx

config = EnhancedVariantCallingPipelineConfig(
    reference_length=10000,
    num_classes=3,
    quality_threshold=20.0,
    pileup_window_size=21,
    cnn_hidden_channels=(64, 128, 256),
    cnn_fc_dims=(256, 128),
    cnn_dropout_rate=0.2,
    enable_preprocessing=True,
    enable_quality_recalibration=True,
)

pipeline = EnhancedVariantCallingPipeline(config, rngs=nnx.Rngs(42))
# Note: this pipeline has no training-mode toggle; dropout state is managed
# by submodules directly when applicable.
```

### Training Mode

```python
# EnhancedVariantCallingPipeline does not expose train_mode/eval_mode toggles.
# Submodules that use dropout manage their own state during apply().
for batch in dataloader:
    loss = train_step(pipeline, batch)
```

### Access Components

```python
# Quality filter (if enabled)
if pipeline.quality_filter is not None:
    pipeline.quality_filter.threshold[...]

# Pileup generator
pipeline.pileup.config.temperature

# CNN classifier
pipeline.cnn_classifier

# Quality recalibration (if enabled)
if pipeline.quality_recalibration is not None:
    pipeline.quality_recalibration
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
| `pileup` | (reference_length, 4) | Aggregated pileup |
| `logits` | (reference_length, num_classes) | Raw predictions |
| `probabilities` | (reference_length, num_classes) | Class probabilities |
| `quality_scores` | (reference_length,) | Recalibrated quality* |
| `filter_weights` | (reference_length,) | Soft filter weights* |

*Only present when `enable_quality_recalibration=True`
