# Training Overview

DiffBio provides utilities for end-to-end gradient-based training of differentiable bioinformatics pipelines.

## Key Concepts

### End-to-End Differentiability

Unlike traditional pipelines where each component is optimized separately, DiffBio enables joint optimization:

```
Traditional: [Opt A] → [Opt B] → [Opt C]  (cascaded errors)
DiffBio:     [A → B → C]                   (joint optimization)
                 ↑
            single loss
```

### Gradient Flow

Gradients flow through all pipeline components:

```python
def pipeline_loss(pipeline, data, targets):
    # Gradients flow through:
    # - Quality threshold
    # - Pileup weighting
    # - Classifier weights
    result, _, _ = pipeline.apply(data, {}, None)
    return cross_entropy_loss(result["logits"], targets)

grads = jax.grad(pipeline_loss)(pipeline, data, targets)
```

## Training Utilities

DiffBio provides training utilities in `diffbio.utils.training`:

| Class/Function | Description |
|----------------|-------------|
| `Trainer` | High-level training loop |
| `TrainingConfig` | Training hyperparameters |
| `TrainingState` | Training progress tracking |
| `cross_entropy_loss` | Classification loss function |
| `create_optax_optimizer` | Optimizer factory |
| `create_synthetic_training_data` | Synthetic data generation |

## Quick Start

```python
from diffbio.pipelines import create_variant_calling_pipeline
from diffbio.utils.training import (
    Trainer,
    TrainingConfig,
    cross_entropy_loss,
    create_synthetic_training_data,
    data_iterator,
)

# 1. Create pipeline
pipeline = create_variant_calling_pipeline(
    reference_length=100,
    num_classes=3,
)

# 2. Create trainer
config = TrainingConfig(
    learning_rate=1e-3,
    num_epochs=50,
    log_every=10,
    grad_clip_norm=1.0,
)
trainer = Trainer(pipeline, config)

# 3. Generate training data
inputs, targets = create_synthetic_training_data(
    num_samples=100,
    reference_length=100,
)

# 4. Define loss function
def loss_fn(predictions, targets):
    return cross_entropy_loss(
        predictions["logits"],
        targets["labels"],
        num_classes=3,
    )

# 5. Train
trainer.train(
    data_iterator_fn=lambda: data_iterator(inputs, targets),
    loss_fn=loss_fn,
)

# 6. Get trained model
trained_pipeline = trainer.pipeline
```

## Training Workflow

### 1. Data Preparation

```python
# Real data loading (example)
def load_training_data(bam_files, vcf_files):
    inputs = []
    targets = []

    for bam, vcf in zip(bam_files, vcf_files):
        reads, positions, quality = parse_bam(bam)
        labels = parse_vcf(vcf)

        inputs.append({
            "reads": reads,
            "positions": positions,
            "quality": quality,
        })
        targets.append({"labels": labels})

    return inputs, targets

# Synthetic data for testing
inputs, targets = create_synthetic_training_data(
    num_samples=1000,
    num_reads=20,
    read_length=50,
    reference_length=100,
    variant_rate=0.05,
)
```

### 2. Pipeline Configuration

```python
from diffbio.pipelines import (
    VariantCallingPipeline,
    VariantCallingPipelineConfig,
)
from flax import nnx

config = VariantCallingPipelineConfig(
    reference_length=100,
    num_classes=3,
    quality_threshold=20.0,
    classifier_hidden_dim=128,
)

pipeline = VariantCallingPipeline(config, rngs=nnx.Rngs(42))
```

### 3. Training Configuration

```python
from diffbio.utils.training import TrainingConfig

training_config = TrainingConfig(
    learning_rate=1e-3,   # Adam learning rate
    num_epochs=100,       # Training epochs
    log_every=10,         # Logging frequency
    grad_clip_norm=1.0,   # Gradient clipping (None to disable)
)
```

### 4. Loss Function Definition

```python
from diffbio.utils.training import cross_entropy_loss

def variant_loss(predictions, targets):
    """Custom loss with class weighting."""
    logits = predictions["logits"]
    labels = targets["labels"]

    # Class weights (variants are rare)
    class_weights = jnp.array([1.0, 5.0, 5.0])

    # Weighted cross entropy
    one_hot = jax.nn.one_hot(labels.astype(jnp.int32), 3)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    weighted = one_hot * class_weights
    return -jnp.mean(jnp.sum(weighted * log_probs, axis=-1))
```

### 5. Training Loop

```python
from diffbio.utils.training import Trainer

trainer = Trainer(pipeline, training_config)

# Train with progress logging
trainer.train(
    data_iterator_fn=lambda: data_iterator(inputs, targets),
    loss_fn=variant_loss,
)

# Access training history
print(f"Best loss: {trainer.training_state.best_loss:.4f}")
print(f"Loss history: {trainer.training_state.loss_history[-10:]}")
```

## Advanced Training

### Custom Training Loop

For more control, implement your own training loop:

```python
import jax
import optax
from flax import nnx

# Create optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
)

# Initialize optimizer state
params = nnx.state(pipeline, nnx.Param)
opt_state = optimizer.init(params)

@jax.jit
def train_step(pipeline, opt_state, batch, targets):
    def loss_fn(model):
        result, _, _ = model.apply(batch, {}, None)
        return cross_entropy_loss(result["logits"], targets["labels"])

    loss, grads = jax.value_and_grad(loss_fn)(pipeline)

    params = nnx.state(pipeline, nnx.Param)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    nnx.update(pipeline, optax.apply_updates(params, updates))

    return loss, opt_state

# Training loop
pipeline.train_mode()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch, targets in dataloader:
        loss, opt_state = train_step(pipeline, opt_state, batch, targets)
        epoch_loss += float(loss)
    print(f"Epoch {epoch}: loss = {epoch_loss / len(dataloader):.4f}")

pipeline.eval_mode()
```

### Learning Rate Scheduling

```python
# Warmup + cosine decay
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=100,
    decay_steps=10000,
    end_value=1e-5,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(schedule),
)
```

### Validation Monitoring

```python
def train_with_validation(pipeline, train_data, val_data, config):
    trainer = Trainer(pipeline, config)
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        # Train epoch
        trainer.train_epoch(train_data, loss_fn)

        # Validate
        pipeline.eval_mode()
        val_loss = 0.0
        for batch, targets in val_data:
            result, _, _ = pipeline.apply(batch, {}, None)
            val_loss += float(loss_fn(result, targets))
        val_loss /= len(val_data)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(pipeline, f"best_model.pkl")

        pipeline.train_mode()
        print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")

    return best_val_loss
```

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def should_stop(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

early_stopping = EarlyStopping(patience=20)

for epoch in range(max_epochs):
    loss = train_epoch(...)
    if early_stopping.should_stop(loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

## Checkpointing

### Save Checkpoints

```python
import pickle
from flax import nnx

def save_checkpoint(pipeline, path):
    state = nnx.state(pipeline, nnx.Param)
    with open(path, 'wb') as f:
        pickle.dump(state, f)
```

### Load Checkpoints

```python
def load_checkpoint(pipeline, path):
    with open(path, 'rb') as f:
        state = pickle.load(f)
    nnx.update(pipeline, state)
```

### Automatic Checkpointing

```python
class CheckpointCallback:
    def __init__(self, save_dir, save_every=100):
        self.save_dir = save_dir
        self.save_every = save_every
        self.step = 0

    def __call__(self, pipeline, loss):
        self.step += 1
        if self.step % self.save_every == 0:
            path = f"{self.save_dir}/checkpoint_{self.step}.pkl"
            save_checkpoint(pipeline, path)
            print(f"Saved checkpoint to {path}")
```

## Distributed Training

### Multi-GPU with pmap

```python
# Replicate model across devices
devices = jax.local_devices()
pipeline_replicated = jax.device_put_replicated(pipeline, devices)

@jax.pmap
def parallel_train_step(pipeline, batch, targets):
    def loss_fn(model):
        result, _, _ = model.apply(batch, {}, None)
        return cross_entropy_loss(result["logits"], targets["labels"])

    loss, grads = jax.value_and_grad(loss_fn)(pipeline)

    # All-reduce gradients
    grads = jax.lax.pmean(grads, axis_name='devices')

    return loss, grads

# Shard data across devices
def shard_batch(batch, num_devices):
    return jax.tree.map(
        lambda x: x.reshape(num_devices, -1, *x.shape[1:]),
        batch
    )
```

## Best Practices

1. **Start with synthetic data** to verify pipeline works
2. **Use gradient clipping** to prevent exploding gradients
3. **Monitor training metrics** (loss, gradient norms)
4. **Validate frequently** to catch overfitting
5. **Save checkpoints** regularly
6. **Use temperature annealing** for smooth → discrete transition

## Next Steps

- See [Training Utilities](utilities.md) for API reference
- Explore [Examples](../../examples/overview.md) for complete training scripts
- Read about [Variant Calling Pipeline](../pipelines/variant-calling.md) for specific use cases
