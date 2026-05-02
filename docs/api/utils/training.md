# Training Utilities API

Training utilities for differentiable bioinformatics pipelines.

## Trainer

::: diffbio.utils.training.Trainer
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - train
        - train_epoch

## Configuration

### TrainingConfig

::: diffbio.utils.training.TrainingConfig
    options:
      show_root_heading: true
      members: []

### TrainingState

::: diffbio.utils.training.TrainingState
    options:
      show_root_heading: true
      members: []

## Loss Functions

### cross_entropy_loss

::: diffbio.utils.training.cross_entropy_loss
    options:
      show_root_heading: true

## Optimizer Utilities

### create_optax_optimizer

::: diffbio.utils.training.create_optax_optimizer
    options:
      show_root_heading: true

## Data Utilities

### create_synthetic_training_data

::: diffbio.utils.training.create_synthetic_training_data
    options:
      show_root_heading: true

### data_iterator

::: diffbio.utils.training.data_iterator
    options:
      show_root_heading: true

## Usage Examples

### Complete Training Example

```python
from diffbio.pipelines import create_variant_calling_pipeline
from diffbio.utils.training import (
    Trainer,
    TrainingConfig,
    cross_entropy_loss,
    create_synthetic_training_data,
    data_iterator,
)

# Create pipeline
pipeline = create_variant_calling_pipeline(reference_length=100)

# Create trainer
config = TrainingConfig(
    learning_rate=1e-3,
    num_epochs=50,
    log_every=10,
)
trainer = Trainer(pipeline, config)

# Generate data
inputs, targets = create_synthetic_training_data(
    num_samples=100,
    reference_length=100,
)

# Define loss
def loss_fn(predictions, targets):
    return cross_entropy_loss(
        predictions["logits"],
        targets["labels"],
    )

# Train
trainer.train(
    data_iterator_fn=lambda: data_iterator(inputs, targets),
    loss_fn=loss_fn,
)

# Get trained model
trained = trainer.pipeline
```

### Custom Training Loop

```python
import jax
import optax
from flax import nnx

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(nnx.state(pipeline, nnx.Param))

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
```

## Module Exports

```python
from diffbio.utils.training import (
    # Core
    Trainer,
    TrainingConfig,
    TrainingState,

    # Loss
    cross_entropy_loss,

    # Optimizer
    create_optax_optimizer,

    # Data
    create_synthetic_training_data,
    data_iterator,
)
```
