# Variant Calling Pipeline

The `VariantCallingPipeline` is an end-to-end differentiable pipeline for calling genetic variants from sequencing reads.

## Overview

The pipeline processes sequencing reads through three stages:

```mermaid
graph LR
    A[Reads] --> B[Quality Filter]
    B --> C[Pileup Generation]
    C --> D[Variant Classifier]
    D --> E[Predictions]

    style A fill:#d1fae5,stroke:#059669,color:#064e3b
    style B fill:#e0e7ff,stroke:#4338ca,color:#312e81
    style C fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    style D fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    style E fill:#d1fae5,stroke:#059669,color:#064e3b
```

1. **Quality Filter**: Soft-masks low-quality bases using learnable threshold
2. **Pileup Generation**: Aggregates filtered reads into position-wise distributions
3. **Variant Classifier**: Neural network predicts variant class at each position

## Quick Start

```python
from diffbio.pipelines import create_variant_calling_pipeline
import jax
import jax.numpy as jnp

# Create pipeline
pipeline = create_variant_calling_pipeline(
    reference_length=100,
    num_classes=3,  # ref/snp/indel
    quality_threshold=20.0,
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

print(f"Pileup shape: {result['pileup'].shape}")           # (100, 4)
print(f"Predictions shape: {result['probabilities'].shape}")  # (100, 3)
```

## Configuration

### VariantCallingPipelineConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_length` | int | 100 | Length of reference sequence |
| `num_classes` | int | 3 | Number of variant classes |
| `quality_threshold` | float | 20.0 | Initial quality threshold |
| `pileup_window_size` | int | 11 | Context window for classifier |
| `classifier_hidden_dim` | int | 64 | Hidden dimension of classifier MLP |
| `use_quality_weights` | bool | True | Weight pileup by quality |

```python
from diffbio.pipelines import VariantCallingPipelineConfig

config = VariantCallingPipelineConfig(
    reference_length=10000,
    num_classes=3,
    quality_threshold=20.0,
    pileup_window_size=21,
    classifier_hidden_dim=128,
    use_quality_weights=True,
)
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
| `filtered_reads` | (num_reads, read_length, 4) | Quality-filtered reads |
| `filtered_quality` | (num_reads, read_length) | Filtered quality scores |
| `pileup` | (reference_length, 4) | Aggregated nucleotide distribution |
| `logits` | (reference_length, num_classes) | Raw classifier output |
| `probabilities` | (reference_length, num_classes) | Softmax probabilities |

## Pipeline Stages

### Stage 1: Quality Filtering

The quality filter applies soft masking based on quality scores:

```python
# Internal operation
retention_weight = sigmoid(quality - threshold)
filtered_reads = reads * retention_weight
```

Access the threshold:

```python
print(pipeline.quality_filter.threshold[...])  # e.g., 20.0
```

### Stage 2: Pileup Generation

Aggregates reads into position-wise nucleotide distributions:

```python
# At each reference position, compute weighted sum of read bases
# weighted by quality and position overlap
pileup[pos] = softmax(weighted_base_counts[pos])
```

### Stage 3: Variant Classification

Neural network classifier with sliding window:

```python
# For each position, extract context window
window = pileup[pos-k:pos+k+1]  # (window_size, 4)

# Classify using MLP
logits = classifier(window.flatten())  # (num_classes,)
probs = softmax(logits)
```

Classes:

- **0**: Reference (no variant)
- **1**: SNP (single nucleotide polymorphism)
- **2**: Indel (insertion/deletion)

## Training

### Using the Trainer Class

```python
from diffbio.pipelines import create_variant_calling_pipeline
from diffbio.utils.training import (
    Trainer, TrainingConfig, cross_entropy_loss,
    create_synthetic_training_data, data_iterator
)

# Create pipeline
pipeline = create_variant_calling_pipeline(reference_length=100)

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

### Custom Training Loop

```python
import jax
import optax
from flax import nnx

# Setup
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(nnx.state(pipeline, nnx.Param))

@jax.jit
def train_step(pipeline, opt_state, batch_data, targets):
    def loss_fn(pipeline):
        result, _, _ = pipeline.apply(batch_data, {}, None)
        return cross_entropy_loss(result["logits"], targets["labels"])

    loss, grads = jax.value_and_grad(loss_fn)(pipeline)

    params = nnx.state(pipeline, nnx.Param)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    nnx.update(pipeline, optax.apply_updates(params, updates))

    return loss, opt_state

# Training loop
pipeline.train_mode()
for epoch in range(num_epochs):
    for batch, targets in dataloader:
        loss, opt_state = train_step(pipeline, opt_state, batch, targets)
pipeline.eval_mode()
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
```

### Batch Processing

```python
from datarax.core.element_batch import Batch, Element

# Create batch
elements = [Element(data=d, state={}, metadata={}) for d in samples]
batch = Batch.from_elements(elements)

# Process batch
result_batch = pipeline.apply_batch(batch)
```

### Using call_variants()

```python
results = pipeline.call_variants(batch, threshold=0.5)

print(results["predictions"])    # Predicted classes
print(results["probabilities"])  # Class probabilities
print(results["pileup"])         # Pileup for inspection
```

## Accessing Components

The pipeline exposes its sub-components for inspection and modification:

```python
# Quality filter
pipeline.quality_filter.threshold[...]  # Current threshold

# Pileup generator
pipeline.pileup.temperature[...]  # Pileup temperature

# Classifier
pipeline.classifier  # Neural network module
```

### Modifying Components

```python
# Update quality threshold
pipeline.quality_filter.threshold[...] = 25.0

# Access all learnable parameters
params = nnx.state(pipeline, nnx.Param)
print(jax.tree.map(lambda x: x.shape, params))
```

## Advanced Usage

### Temperature Annealing

```python
def train_with_annealing(pipeline, data_fn, num_epochs):
    for epoch in range(num_epochs):
        # Anneal temperature from 5.0 to 0.5
        temp = 5.0 * (0.5 / 5.0) ** (epoch / num_epochs)
        pipeline.pileup.temperature[...] = temp

        # Train epoch
        for batch, targets in data_fn():
            loss = train_step(pipeline, batch, targets)

        print(f"Epoch {epoch}: temp={temp:.2f}, loss={loss:.4f}")
```

### Custom Classifier

```python
from diffbio.operators.variant import VariantClassifier, VariantClassifierConfig

# Create custom classifier
custom_classifier = VariantClassifier(
    VariantClassifierConfig(
        num_classes=4,  # More classes
        hidden_dim=256,  # Larger network
        input_window=21,  # Larger context
        dropout_rate=0.2,
    ),
    rngs=nnx.Rngs(42),
)

# Replace in pipeline
pipeline.classifier = custom_classifier
```

### Multi-GPU Training

```python
# Replicate pipeline across devices
pipeline_replicated = jax.device_put_replicated(
    pipeline, jax.local_devices()
)

@jax.pmap
def parallel_train_step(pipeline, batch):
    # Each device processes its shard
    result, _, _ = pipeline.apply(batch, {}, None)
    return result
```

## Performance Tips

1. **JIT compile** the apply method for inference
2. **Batch reads** from the same region together
3. **Use appropriate window size** - larger windows capture more context but are slower
4. **Reduce classifier hidden dim** for faster inference

```python
# Fast inference
@jax.jit
def fast_inference(pipeline, data):
    result, _, _ = pipeline.apply(data, {}, None)
    return result["probabilities"]

# Pre-compile
_ = fast_inference(pipeline, sample_data)

# Fast subsequent calls
preds = fast_inference(pipeline, real_data)
```

## Evaluation

```python
def evaluate_pipeline(pipeline, test_data, test_labels):
    pipeline.eval_mode()

    all_preds = []
    all_labels = []

    for data, labels in zip(test_data, test_labels):
        result, _, _ = pipeline.apply(data, {}, None)
        preds = jnp.argmax(result["probabilities"], axis=-1)
        all_preds.append(preds)
        all_labels.append(labels["labels"])

    preds = jnp.concatenate(all_preds)
    labels = jnp.concatenate(all_labels)

    # Compute metrics
    accuracy = (preds == labels).mean()
    precision = compute_precision(preds, labels)
    recall = compute_recall(preds, labels)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
```

## References

1. Poplin, R. et al. (2018). "A universal SNP and small-indel variant caller using deep neural networks."

2. Luo, R. et al. (2020). "Exploring the limit of using a deep neural network on pileup data for germline variant calling."
