# Training Utilities

The `diffbio.utils.training` module provides utilities for training differentiable bioinformatics pipelines.

## TrainingConfig

Configuration dataclass for training hyperparameters.

```python
from diffbio.utils.training import TrainingConfig
from substrax.optim import OptimizerConfig

config = TrainingConfig(
    optimizer=OptimizerConfig(
        optimizer_type="adam", learning_rate=1e-3, gradient_clip_norm=1.0
    ),
    num_epochs=100,         # Number of training epochs
    log_every=10,           # Log every N steps
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | `substrax.optim.OptimizerConfig` | Adam at 1e-3, global-norm clip 1.0 | The optimizer the trainer builds: type, learning rate or schedule, clipping, weight decay; substrax refuses invalid values |
| `num_epochs` | int | 100 | Number of training epochs |
| `log_every` | int | 10 | Log metrics every N steps |

## TrainingState

Dataclass tracking training progress.

```python
from diffbio.utils.training import TrainingState

state = TrainingState()

# Access during training
print(state.step)          # Current step
print(state.epoch)         # Current epoch
print(state.loss_history)  # List of all losses
print(state.best_loss)     # Best loss seen
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `step` | int | Current training step |
| `epoch` | int | Current epoch |
| `loss_history` | list[float] | History of all loss values |
| `best_loss` | float | Best (lowest) loss observed |

## Trainer

High-level training class using Flax NNX patterns.

```python
from diffbio.utils.training import Trainer, TrainingConfig
from diffbio.pipelines import create_variant_calling_pipeline

# Create pipeline
pipeline = create_variant_calling_pipeline(reference_length=100)

# Create trainer
trainer = Trainer(
    pipeline=pipeline,
    config=TrainingConfig(num_epochs=50),
)
```

### Methods

#### train()

Run the full training loop.

```python
trainer.train(
    data_iterator_fn=lambda: iter(zip(inputs, targets)),
    loss_fn=my_loss_function,
)
```

**Parameters:**

- `data_iterator_fn`: Function returning a fresh iterator of (batch, targets) tuples
- `loss_fn`: Function taking (predictions, targets) and returning scalar loss

#### train_epoch()

Train for a single epoch.

```python
metrics = trainer.train_epoch(
    data_iterator=iter(zip(inputs, targets)),
    loss_fn=my_loss_function,
)

print(metrics["avg_loss"])  # Average epoch loss
print(metrics["min_loss"])  # Minimum batch loss
print(metrics["max_loss"])  # Maximum batch loss
```

### Accessing Results

```python
# After training
trained_pipeline = trainer.pipeline
training_history = trainer.training_state.loss_history
best_loss = trainer.training_state.best_loss
```

## Loss Functions

### softmax_cross_entropy

The classification loss is calibrax's `softmax_cross_entropy`; the class count is the logits'
last axis.

```python
from calibrax.metrics.functional import softmax_cross_entropy

loss = softmax_cross_entropy(
    logits,       # (batch, ..., num_classes)
    labels,       # (batch, ...) integer labels
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `logits` | Array | Raw model predictions |
| `labels` | Array | Integer class labels |
| `mask` | Array, optional | Boolean mask selecting the positions that count |
| `weights` | Array, optional | Per-position weights for a weighted mean |
| `reduction` | str | `"mean"` (default), `"sum"` or `"none"` |

**Returns:** Scalar loss value (or per-position losses with `reduction="none"`)

### Custom Loss Functions

Define custom losses for specific objectives:

```python
def focal_loss(logits, labels, gamma=2.0, num_classes=3):
    """Focal loss for class imbalance."""
    one_hot = jax.nn.one_hot(labels.astype(jnp.int32), num_classes)
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    focal_weight = (1 - probs) ** gamma
    return -jnp.mean(jnp.sum(one_hot * focal_weight * log_probs, axis=-1))


def weighted_cross_entropy(logits, labels, class_weights, num_classes=3):
    """Cross-entropy with class weights."""
    one_hot = jax.nn.one_hot(labels.astype(jnp.int32), num_classes)
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    weights = class_weights[labels.astype(jnp.int32)]
    return -jnp.mean(weights * jnp.sum(one_hot * log_probs, axis=-1))
```

## Optimizer Utilities

### The optimizer spec

`TrainingConfig.optimizer` is a `substrax.optim.OptimizerConfig`, and the trainer
builds its `nnx.Optimizer` from it with `substrax.optim.create_optimizer`; DiffBio keeps
no optimizer-building code of its own. `default_training_optimizer()` returns the
default spec (Adam at 1e-3 with a unit global-norm clip).

```python
from diffbio.utils.training import TrainingConfig, default_training_optimizer
from substrax.optim import OptimizerConfig

assert TrainingConfig().optimizer == default_training_optimizer()

config = TrainingConfig(
    optimizer=OptimizerConfig(
        optimizer_type="adam", learning_rate=1e-3, gradient_clip_norm=1.0
    ),
)
```

### Custom Optimizers

Every optax alias, schedule and clip the spec supports is described in the spec itself,
so a custom optimizer is a different `OptimizerConfig`:

```python
import optax
from substrax.optim import OptimizerConfig

# AdamW with weight decay
adamw = OptimizerConfig(optimizer_type="adamw", learning_rate=1e-3, weight_decay=0.01)

# A schedule is the learning rate; it reaches the update as optax's own step count
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=1000,
    decay_steps=10000,
)
warmed = OptimizerConfig(optimizer_type="adam", learning_rate=schedule)

# Clipping comes first in the chain substrax builds
clipped = OptimizerConfig(optimizer_type="adam", learning_rate=1e-3, gradient_clip_norm=1.0)

config = TrainingConfig(optimizer=warmed)
```

## Data Utilities

### create_synthetic_training_data

Generate synthetic training data for testing and development.

```python
from diffbio.utils.training import create_synthetic_training_data

inputs, targets = create_synthetic_training_data(
    num_samples=100,        # Number of samples
    num_reads=20,           # Reads per sample
    read_length=50,         # Length of each read
    reference_length=100,   # Reference sequence length
    variant_rate=0.1,       # Probability of variant at each position
    seed=42,                # Random seed
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_samples` | int | 100 | Number of samples to generate |
| `num_reads` | int | 10 | Number of reads per sample |
| `read_length` | int | 50 | Length of each read |
| `reference_length` | int | 100 | Length of reference sequence |
| `variant_rate` | float | 0.1 | Probability of variant at each position |
| `seed` | int | 42 | Random seed for reproducibility |

**Returns:** Tuple of (inputs, targets)

- `inputs`: List of dicts with keys "reads", "positions", "quality"
- `targets`: List of dicts with key "labels"

### data_iterator

Create an iterator over training data.

```python
from diffbio.utils.training import data_iterator

# Create iterator
iterator = data_iterator(inputs, targets, batch_size=1)

for batch_data, batch_targets in iterator:
    # Process batch
    pass
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | list[dict] | - | List of input dictionaries |
| `targets` | list[dict] | - | List of target dictionaries |
| `batch_size` | int | 1 | Batch size (currently only 1 supported) |

## Complete Example

```python
from diffbio.pipelines import create_variant_calling_pipeline
from calibrax.metrics.functional import softmax_cross_entropy
from diffbio.utils.training import (
    Trainer,
    TrainingConfig,
    TrainingState,
    create_synthetic_training_data,
    data_iterator,
)

# 1. Create pipeline
pipeline = create_variant_calling_pipeline(
    reference_length=100,
    num_classes=3,
)

# 2. Generate data
inputs, targets = create_synthetic_training_data(
    num_samples=500,
    reference_length=100,
    variant_rate=0.05,
)

# 3. Split into train/val
train_inputs, val_inputs = inputs[:400], inputs[400:]
train_targets, val_targets = targets[:400], targets[400:]

# 4. Configure training
config = TrainingConfig(
    optimizer=OptimizerConfig(
        optimizer_type="adam", learning_rate=1e-3, gradient_clip_norm=1.0
    ),
    num_epochs=50,
    log_every=10,
)

# 5. Create trainer
trainer = Trainer(pipeline, config)

# 6. Define loss
def loss_fn(predictions, targets):
    return softmax_cross_entropy(
        predictions["logits"],
        targets["labels"],
    )

# 7. Train
trainer.train(
    data_iterator_fn=lambda: data_iterator(train_inputs, train_targets),
    loss_fn=loss_fn,
)

# 8. Evaluate
pipeline.eval_mode()
val_loss = 0.0
for inp, tgt in zip(val_inputs, val_targets):
    result, _, _ = pipeline.apply(inp, {}, None)
    val_loss += float(loss_fn(result, tgt))
val_loss /= len(val_inputs)

print(f"Final validation loss: {val_loss:.4f}")
print(f"Best training loss: {trainer.training_state.best_loss:.4f}")

# 9. Save model
import pickle
from flax import nnx
from substrax.optim import OptimizerConfig

state = nnx.state(trainer.pipeline, nnx.Param)
with open("trained_model.pkl", "wb") as f:
    pickle.dump(state, f)
```

## API Reference

```python
# All exports from diffbio.utils.training
from calibrax.metrics.functional import softmax_cross_entropy
from diffbio.utils.training import (
    # Configuration
    TrainingConfig,
    TrainingState,

    # Trainer
    Trainer,

    # Loss functions

    # Optimizer spec
    default_training_optimizer,

    # Data utilities
    create_synthetic_training_data,
    data_iterator,
)
```
