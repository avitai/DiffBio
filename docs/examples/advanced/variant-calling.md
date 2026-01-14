# Variant Calling Pipeline Example

This example demonstrates the complete end-to-end variant calling workflow using DiffBio.

## Overview

We'll build and train a variant calling pipeline that:

1. Filters reads by quality
2. Generates pileups
3. Classifies variants

```mermaid
graph LR
    A[Reads] --> B[Quality Filter]
    B --> C[Pileup]
    C --> D[Classifier]
    D --> E[Variants]
```

## Setup

```python
import jax
import jax.numpy as jnp
from flax import nnx

from diffbio.pipelines import (
    create_variant_calling_pipeline,
    VariantCallingPipeline,
    VariantCallingPipelineConfig,
)
from diffbio.utils.training import (
    Trainer,
    TrainingConfig,
    cross_entropy_loss,
    create_synthetic_training_data,
    data_iterator,
)
```

## Create the Pipeline

### Quick Creation

```python
# Simple factory function
pipeline = create_variant_calling_pipeline(
    reference_length=100,
    num_classes=3,        # ref/snp/indel
    quality_threshold=20.0,
    hidden_dim=64,
    seed=42,
)
```

### Full Configuration

```python
# Full control over parameters
config = VariantCallingPipelineConfig(
    reference_length=100,
    num_classes=3,
    quality_threshold=20.0,
    pileup_window_size=11,
    classifier_hidden_dim=128,
    use_quality_weights=True,
)

rngs = nnx.Rngs(seed=42)
pipeline = VariantCallingPipeline(config, rngs=rngs)
pipeline.eval_mode()  # Disable dropout for inference
```

## Generate Training Data

```python
# Create synthetic training data
train_inputs, train_targets = create_synthetic_training_data(
    num_samples=500,
    num_reads=20,
    read_length=30,
    reference_length=100,
    variant_rate=0.05,
    seed=42,
)

# Split into train/val
val_split = 400
train_inputs, val_inputs = train_inputs[:val_split], train_inputs[val_split:]
train_targets, val_targets = train_targets[:val_split], train_targets[val_split:]

print(f"Training samples: {len(train_inputs)}")
print(f"Validation samples: {len(val_inputs)}")
```

## Inspect Training Data

```python
# Look at one sample
sample = train_inputs[0]
target = train_targets[0]

print("Input keys:", sample.keys())
print(f"  reads: {sample['reads'].shape}")
print(f"  positions: {sample['positions'].shape}")
print(f"  quality: {sample['quality'].shape}")

print("Target keys:", target.keys())
print(f"  labels: {target['labels'].shape}")

# Count variants in this sample
num_variants = (target['labels'] > 0).sum()
print(f"Variants in sample: {num_variants}")
```

## Run Inference (Before Training)

```python
# Apply pipeline to one sample
result, _, _ = pipeline.apply(sample, {}, None)

print("Output keys:", result.keys())
print(f"  pileup: {result['pileup'].shape}")
print(f"  logits: {result['logits'].shape}")
print(f"  probabilities: {result['probabilities'].shape}")

# Get predictions
predictions = jnp.argmax(result['probabilities'], axis=-1)
print(f"Predicted variants (before training): {(predictions > 0).sum()}")
```

## Train the Pipeline

### Using the Trainer Class

```python
# Create trainer
trainer = Trainer(
    pipeline,
    TrainingConfig(
        learning_rate=1e-3,
        num_epochs=30,
        log_every=50,
        grad_clip_norm=1.0,
    ),
)

# Define loss function
def loss_fn(predictions, targets):
    return cross_entropy_loss(
        predictions["logits"],
        targets["labels"],
        num_classes=3,
    )

# Train
print("Starting training...")
trainer.train(
    data_iterator_fn=lambda: data_iterator(train_inputs, train_targets),
    loss_fn=loss_fn,
)
print(f"Best training loss: {trainer.training_state.best_loss:.4f}")
```

### Custom Training Loop (Alternative)

```python
import optax

# If not using Trainer, here's a custom loop:
def custom_train(pipeline, train_data, num_epochs=30):
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(nnx.state(pipeline, nnx.Param))

    @jax.jit
    def train_step(pipeline, opt_state, batch, targets):
        def compute_loss(model):
            result, _, _ = model.apply(batch, {}, None)
            return cross_entropy_loss(result["logits"], targets["labels"])

        loss, grads = jax.value_and_grad(compute_loss)(pipeline)
        params = nnx.state(pipeline, nnx.Param)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        nnx.update(pipeline, optax.apply_updates(params, updates))
        return loss, opt_state

    pipeline.train_mode()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inp, tgt in zip(*train_data):
            loss, opt_state = train_step(pipeline, opt_state, inp, tgt)
            epoch_loss += float(loss)
        print(f"Epoch {epoch}: loss = {epoch_loss/len(train_data[0]):.4f}")

    pipeline.eval_mode()
    return pipeline
```

## Evaluate the Model

```python
def evaluate(pipeline, inputs, targets):
    """Evaluate pipeline on a dataset."""
    pipeline.eval_mode()

    all_preds = []
    all_labels = []

    for inp, tgt in zip(inputs, targets):
        result, _, _ = pipeline.apply(inp, {}, None)
        preds = jnp.argmax(result["probabilities"], axis=-1)
        all_preds.append(preds)
        all_labels.append(tgt["labels"])

    preds = jnp.concatenate(all_preds)
    labels = jnp.concatenate(all_labels)

    # Accuracy
    accuracy = (preds == labels).mean()

    # Per-class accuracy
    for cls in range(3):
        mask = labels == cls
        cls_acc = (preds[mask] == cls).mean() if mask.sum() > 0 else 0.0
        print(f"  Class {cls} accuracy: {cls_acc:.4f}")

    # Variant detection metrics
    true_variants = labels > 0
    pred_variants = preds > 0

    tp = (pred_variants & true_variants).sum()
    fp = (pred_variants & ~true_variants).sum()
    fn = (~pred_variants & true_variants).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

# Evaluate on validation set
print("\nValidation metrics:")
val_metrics = evaluate(trainer.pipeline, val_inputs, val_targets)
print(f"Accuracy: {val_metrics['accuracy']:.4f}")
print(f"Precision: {val_metrics['precision']:.4f}")
print(f"Recall: {val_metrics['recall']:.4f}")
print(f"F1: {val_metrics['f1']:.4f}")
```

## Analyze Results

```python
# Run inference on a specific sample
pipeline.eval_mode()
result, _, _ = pipeline.apply(val_inputs[0], {}, None)

# Get predictions
probs = result["probabilities"]
preds = jnp.argmax(probs, axis=-1)
true_labels = val_targets[0]["labels"]

# Compare predictions to truth
print(f"\nSample analysis:")
print(f"  True variants: {(true_labels > 0).sum()}")
print(f"  Predicted variants: {(preds > 0).sum()}")

# Find specific positions
variant_positions = jnp.where(true_labels > 0)[0]
print(f"\nTrue variant positions: {variant_positions}")
print(f"Predictions at those positions: {preds[variant_positions]}")
print(f"Confidence at those positions: {probs[variant_positions].max(axis=-1)}")
```

## Inspect Learned Parameters

```python
# Quality filter threshold
print(f"Learned quality threshold: {pipeline.quality_filter.threshold[...]:.2f}")

# Pileup temperature
print(f"Pileup temperature: {pipeline.pileup.temperature[...]:.4f}")
```

## Save and Load Model

```python
import pickle

# Save
state = nnx.state(pipeline, nnx.Param)
with open("variant_calling_model.pkl", "wb") as f:
    pickle.dump(state, f)
print("Model saved!")

# Load
with open("variant_calling_model.pkl", "rb") as f:
    loaded_state = pickle.load(f)
nnx.update(pipeline, loaded_state)
print("Model loaded!")
```

## Production Inference

```python
@jax.jit
def call_variants(pipeline, reads, positions, quality):
    """JIT-compiled variant calling."""
    data = {
        "reads": reads,
        "positions": positions,
        "quality": quality,
    }
    result, _, _ = pipeline.apply(data, {}, None)
    return {
        "predictions": jnp.argmax(result["probabilities"], axis=-1),
        "probabilities": result["probabilities"],
        "pileup": result["pileup"],
    }

# Use in production
pipeline.eval_mode()
results = call_variants(
    pipeline,
    val_inputs[0]["reads"],
    val_inputs[0]["positions"],
    val_inputs[0]["quality"],
)

# Find high-confidence variants
confidence = results["probabilities"].max(axis=-1)
variant_mask = results["predictions"] > 0
high_conf_variants = variant_mask & (confidence > 0.8)
print(f"High-confidence variants: {high_conf_variants.sum()}")
```

## Summary

This example demonstrated:

1. Creating a variant calling pipeline
2. Generating and preparing training data
3. Training the pipeline end-to-end
4. Evaluating performance metrics
5. Analyzing learned parameters
6. Saving and loading models
7. Production inference patterns

## Next Steps

- Experiment with different hyperparameters
- Try real sequencing data (BAM/VCF files)
- Add custom loss functions for class imbalance
- Implement temperature annealing during training
