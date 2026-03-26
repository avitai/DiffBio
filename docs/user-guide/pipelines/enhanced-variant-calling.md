# Enhanced Variant Calling Pipeline

The `EnhancedVariantCallingPipeline` is an end-to-end differentiable pipeline for advanced variant calling, featuring DeepVariant-style CNN classification and VQSR-style quality recalibration.

## Overview

The pipeline processes sequencing reads through four stages:

```mermaid
graph LR
    A[Reads] --> B[Quality Filter]
    B --> C[Pileup Generation]
    C --> D[CNN Classifier]
    D --> E[Quality Recalibration]
    E --> F[Predictions]

    style A fill:#d1fae5,stroke:#059669,color:#064e3b
    style B fill:#e0e7ff,stroke:#4338ca,color:#312e81
    style C fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    style D fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    style E fill:#fef3c7,stroke:#d97706,color:#78350f
    style F fill:#d1fae5,stroke:#059669,color:#064e3b
```

1. **Quality Filter** (optional): Soft-masks low-quality bases using learnable threshold
2. **Pileup Generation**: Aggregates filtered reads into position-wise distributions
3. **CNN Classifier**: DeepVariant-style convolutional neural network for variant classification
4. **Quality Recalibration** (optional): VQSR-style GMM-based quality score recalibration

## Quick Start

```python
from diffbio.pipelines import create_enhanced_variant_calling_pipeline
import jax
import jax.numpy as jnp

# Create pipeline
pipeline = create_enhanced_variant_calling_pipeline(
    reference_length=100,
    num_classes=3,  # ref/snp/indel
    pileup_window_size=11,
)

# Prepare data
num_reads = 20
read_length = 30

data = {
    "reads": jax.random.uniform(jax.random.PRNGKey(0), (num_reads, read_length, 4)),
    "positions": jax.random.randint(jax.random.PRNGKey(1), (num_reads,), 0, 70),
    "quality": jax.random.uniform(jax.random.PRNGKey(2), (num_reads, read_length), 10, 40),
}
data["reads"] = jax.nn.softmax(data["reads"], axis=-1)  # Normalize

# Run pipeline
result, _, _ = pipeline.apply(data, {}, None)

print(f"Pileup shape: {result['pileup'].shape}")              # (100, 4)
print(f"Predictions shape: {result['probabilities'].shape}")   # (100, 3)
print(f"Quality scores shape: {result['quality_scores'].shape}")  # (100,)
```

## Configuration

### EnhancedVariantCallingPipelineConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_length` | int | 1000 | Length of reference sequence |
| `num_classes` | int | 3 | Number of variant classes (ref/SNP/indel) |
| `quality_threshold` | float | 20.0 | Initial quality threshold for filtering |
| `pileup_window_size` | int | 11 | Context window for pileup and CNN |
| `cnn_input_height` | int | 100 | Height of pileup image for CNN |
| `cnn_hidden_channels` | list[int] | [64, 128, 256] | CNN hidden channels |
| `cnn_fc_dims` | list[int] | [256, 128] | Fully connected layer dimensions |
| `cnn_dropout_rate` | float | 0.1 | Dropout rate for CNN classifier |
| `quality_recal_n_components` | int | 3 | GMM components for quality recalibration |
| `quality_recal_n_features` | int | 4 | Features for quality recalibration |
| `quality_recal_threshold` | float | 0.5 | Threshold for quality filtering |
| `enable_preprocessing` | bool | True | Enable quality filtering preprocessing |
| `enable_quality_recalibration` | bool | True | Enable quality recalibration |

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
```

## Input Format

The pipeline expects a dictionary with three keys:

### reads

One-hot encoded reads with shape `(num_reads, read_length, 4)`:

```python
# Hard one-hot encoding
read_indices = jnp.array([0, 1, 2, 3, ...])  # A=0, C=1, G=2, T=3
reads = jnp.eye(4)[read_indices]

# Or soft probabilities from base calling
reads = base_caller_output  # Already (num_reads, read_length, 4)
```

### positions

Starting positions for each read:

```python
positions = jnp.array([100, 250, 430, ...])  # (num_reads,)
# Read i covers positions[i] to positions[i] + read_length - 1
```

### quality

Phred quality scores for each base:

```python
quality = jnp.array([
    [30, 35, 28, 40, ...],  # Read 1 qualities
    [25, 30, 32, 35, ...],  # Read 2 qualities
])  # (num_reads, read_length)
```

## Output Format

The pipeline returns a dictionary with:

| Key | Shape | Description |
|-----|-------|-------------|
| `reads` | (num_reads, read_length, 4) | Original reads |
| `positions` | (num_reads,) | Original positions |
| `quality` | (num_reads, read_length) | Original quality |
| `pileup` | (reference_length, 4) | Aggregated nucleotide distribution |
| `logits` | (reference_length, num_classes) | Raw classifier output |
| `probabilities` | (reference_length, num_classes) | Softmax probabilities |
| `quality_scores` | (reference_length,) | Recalibrated quality scores* |
| `filter_weights` | (reference_length,) | Soft filter weights* |

*Only present when `enable_quality_recalibration=True`

## Pipeline Stages

### Stage 1: Quality Filtering (Optional)

The quality filter applies soft masking based on quality scores:

```python
# Internal operation
retention_weight = sigmoid(quality - threshold)
filtered_reads = reads * retention_weight
```

Access the threshold:

```python
if pipeline.quality_filter is not None:
    print(pipeline.quality_filter.threshold[...])  # e.g., 20.0
```

### Stage 2: Pileup Generation

Aggregates reads into position-wise nucleotide distributions:

```python
# At each reference position, compute weighted sum of read bases
# weighted by quality and position overlap
pileup[pos] = softmax(weighted_base_counts[pos])
```

### Stage 3: CNN Classification

DeepVariant-style convolutional neural network classifier:

```python
# For each position, create a pileup image
# Extract context window around position
# Apply CNN layers: Conv2D → BatchNorm → ReLU → Pool
# Classify using fully connected layers
logits = cnn_classifier(pileup_image)  # (num_classes,)
probs = softmax(logits)
```

The CNN architecture:

- Input: Pileup images `(reference_length, height, window_size, 4)`
- Convolutional layers with increasing channels
- Batch normalization and ReLU activations
- Max pooling for spatial reduction
- Fully connected layers with dropout

Classes:

- **0**: Reference (no variant)
- **1**: SNP (single nucleotide polymorphism)
- **2**: Indel (insertion/deletion)

### Stage 4: Quality Recalibration (Optional)

VQSR-style quality score recalibration using a Gaussian Mixture Model:

```python
# Extract variant features
features = [depth, max_prob, entropy, strand_balance]

# Apply GMM-based quality recalibration
quality_scores = gmm_quality_model(features)
filter_weights = sigmoid(quality_scores - threshold)
```

Features used for recalibration:

- **Depth**: Total coverage at each position
- **Max probability**: Confidence of prediction
- **Entropy**: Uncertainty in predictions
- **Strand balance**: Proxy for strand bias

## Training

### Using the Trainer Class

```python
from diffbio.pipelines import create_enhanced_variant_calling_pipeline
from diffbio.utils.training import (
    Trainer, TrainingConfig, cross_entropy_loss,
    create_synthetic_training_data, data_iterator
)

# Create pipeline
pipeline = create_enhanced_variant_calling_pipeline(reference_length=100)

# Create trainer
trainer = Trainer(
    pipeline,
    TrainingConfig(
        learning_rate=1e-3,
        num_epochs=50,
        log_every=10,
        grad_clip_norm=1.0,
    )
)

# Generate training data
inputs, targets = create_synthetic_training_data(
    num_samples=100,
    reference_length=100,
)

# Define loss function
def loss_fn(predictions, targets):
    return cross_entropy_loss(
        predictions["logits"],
        targets["labels"],
        num_classes=3,
    )

# Train
trainer.train(
    data_iterator_fn=lambda: data_iterator(inputs, targets),
    loss_fn=loss_fn,
)

# Get trained pipeline
trained_pipeline = trainer.pipeline
```

### Train vs Eval Mode

```python
# Enable dropout during training
pipeline.train_mode()

for batch in train_dataloader:
    loss = train_step(pipeline, batch)

# Disable dropout for inference
pipeline.eval_mode()

result, _, _ = pipeline.apply(test_data, {}, None)
```

## Inference

### Single Sample

```python
pipeline.eval_mode()

result, _, _ = pipeline.apply(data, {}, None)
predictions = jnp.argmax(result["probabilities"], axis=-1)

# Find variant positions
variant_positions = jnp.where(predictions > 0)[0]
print(f"Variants at positions: {variant_positions}")

# Get quality-filtered variants (if recalibration enabled)
if "filter_weights" in result:
    high_quality = result["filter_weights"] > 0.5
    high_quality_variants = jnp.where(predictions > 0 & high_quality)[0]
```

### JIT Compilation for Performance

```python
@jax.jit
def fast_inference(pipeline, data):
    result, _, _ = pipeline.apply(data, {}, None)
    return result["probabilities"], result.get("quality_scores")

# Pre-compile
_ = fast_inference(pipeline, sample_data)

# Fast subsequent calls
preds, quality = fast_inference(pipeline, real_data)
```

## Accessing Components

The pipeline exposes its sub-components for inspection and modification:

```python
# Quality filter (if enabled)
if pipeline.quality_filter is not None:
    pipeline.quality_filter.threshold[...]  # Current threshold

# Pileup generator
pipeline.pileup.temperature[...]  # Pileup temperature

# CNN classifier
pipeline.cnn_classifier  # Neural network module

# Quality recalibration (if enabled)
if pipeline.quality_recalibration is not None:
    pipeline.quality_recalibration  # GMM-based quality model
```

### Modifying Components

```python
# Update quality threshold
if pipeline.quality_filter is not None:
    pipeline.quality_filter.threshold[...] = 25.0

# Access all learnable parameters
from flax import nnx
params = nnx.state(pipeline, nnx.Param)
print(jax.tree.map(lambda x: x.shape, params))
```

## Comparison with Basic Pipeline

| Feature | VariantCallingPipeline | EnhancedVariantCallingPipeline |
|---------|----------------------|------------------------------|
| Classifier | MLP | CNN (DeepVariant-style) |
| Quality Recalibration | No | Yes (VQSR-style) |
| Input Representation | Flat window | Pileup images |
| Dropout | No | Yes |
| Configurable Stages | No | Yes (enable_* flags) |

## References

1. Poplin, R. et al. (2018). "A universal SNP and small-indel variant caller using deep neural networks." *Nature Biotechnology*.

2. Van der Auwera, G.A. & O'Connor, B.D. (2020). "Genomics in the Cloud: Using Docker, GATK, and WDL in Terra." - VQSR methodology.

3. DePristo, M.A. et al. (2011). "A framework for variation discovery and genotyping using next-generation DNA sequencing data." *Nature Genetics*.
