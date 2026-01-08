# DiffBio

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
  <a href="https://jax.readthedocs.io/"><img src="https://img.shields.io/badge/JAX-0.4.35+-green.svg" alt="JAX"></a>
  <a href="https://flax.readthedocs.io/"><img src="https://img.shields.io/badge/Flax-0.10+-orange.svg" alt="Flax"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <strong>End-to-End Differentiable Bioinformatics Pipelines</strong>
</p>

<p align="center">
  Built on <a href="https://github.com/mahdi-shafiei/workshop-data">Datarax</a> | Powered by <a href="https://jax.readthedocs.io/">JAX</a> & <a href="https://flax.readthedocs.io/">Flax NNX</a>
</p>

---

## Overview

DiffBio is a framework for building **end-to-end differentiable bioinformatics pipelines**. By replacing discrete operations with differentiable relaxations, DiffBio enables gradient-based optimization through entire analysis workflows.

Traditional bioinformatics pipelines use discrete operations (hard thresholds, argmax decisions) that block gradient flow. DiffBio addresses this by:

- **Soft quality filtering** using sigmoid-based weights instead of hard cutoffs
- **Differentiable pileup** with soft position assignments via temperature-controlled softmax
- **Soft alignment scoring** replacing discrete Smith-Waterman with continuous relaxations
- **End-to-end training** of complete pipelines using gradient descent

This enables learning optimal pipeline parameters directly from data, rather than manual tuning.

## Features

### Differentiable Operators

| Operator | Description |
|----------|-------------|
| `DifferentiableQualityFilter` | Sigmoid-based soft quality filtering with learnable threshold |
| `DifferentiablePileup` | Soft pileup generation with temperature-controlled position assignments |
| `SoftSmithWaterman` | Differentiable sequence alignment with soft max operations |
| `VariantClassifier` | Neural network for variant classification (ref/SNP/indel) |

### Pipelines

| Pipeline | Description |
|----------|-------------|
| `VariantCallingPipeline` | End-to-end differentiable variant calling from reads to predictions |

### Training Infrastructure

- **Flax NNX patterns** for stateful model management
- **Gradient clipping** and configurable optimizers via Optax
- **Synthetic data generation** for development and testing
- **Cross-entropy loss** for variant classification

## Installation

```bash
# Clone the repository
git clone https://github.com/mahdi-shafiei/DiffBio.git
cd DiffBio

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Using Individual Operators

```python
import jax
import jax.numpy as jnp
from flax import nnx

from diffbio.operators import (
    DifferentiableQualityFilter,
    DifferentiablePileup,
    SoftSmithWaterman,
)

# Quality filtering with learnable threshold
quality_filter = DifferentiableQualityFilter(
    threshold=20.0,
    temperature=1.0,
    rngs=nnx.Rngs(0),
)

# Apply to reads
quality_scores = jnp.array([35.0, 15.0, 28.0, 10.0])
reads = jax.nn.one_hot(jnp.array([[0, 1, 2, 3]] * 4), 4)
data = {"reads": reads, "quality": quality_scores}

filtered_data, _, _ = quality_filter.apply(data, {}, None)
# filtered_data["weights"] contains soft weights for each read
```

### Using the Variant Calling Pipeline

```python
from diffbio.pipelines import (
    VariantCallingPipeline,
    VariantCallingPipelineConfig,
    create_variant_calling_pipeline,
)

# Create pipeline with default configuration
pipeline = create_variant_calling_pipeline(
    reference_length=100,
    num_classes=3,  # ref, SNP, indel
    hidden_dim=32,
    seed=42,
)

# Process reads
batch_data = {
    "reads": reads,           # (num_reads, read_length, 4)
    "positions": positions,   # (num_reads,)
    "quality": quality,       # (num_reads, read_length)
}

result, _, _ = pipeline.apply(batch_data, {}, None)
# result["logits"] contains per-position variant predictions
# result["probabilities"] contains class probabilities
```

### Training a Pipeline

```python
from diffbio.utils import (
    Trainer,
    TrainingConfig,
    cross_entropy_loss,
    create_synthetic_training_data,
    data_iterator,
)

# Generate synthetic training data
inputs, targets = create_synthetic_training_data(
    num_samples=100,
    num_reads=10,
    read_length=50,
    reference_length=100,
    variant_rate=0.1,
)

# Configure training
config = TrainingConfig(
    learning_rate=1e-3,
    num_epochs=50,
    log_every=10,
    grad_clip_norm=1.0,
)

# Create trainer
trainer = Trainer(pipeline, config)

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

# Access trained pipeline
trained_pipeline = trainer.pipeline
```

## Architecture

DiffBio is built on [Datarax](https://github.com/mahdi-shafiei/workshop-data), a framework for composable data processing operators. Each DiffBio operator inherits from `OperatorModule` and implements:

```
apply(data, state, metadata) -> (output_data, output_state, output_metadata)
```

This enables:
- **Composition**: Chain operators into pipelines
- **Batch processing**: Automatic vectorization via `apply_batch()`
- **Gradient flow**: End-to-end differentiability through the pipeline

### Operator Composition

```python
from datarax import Batch

# Create batch from data
batch = Batch.from_data(batch_data)

# Apply operators in sequence
batch = quality_filter.apply_batch(batch)
batch = pileup.apply_batch(batch)
batch = classifier.apply_batch(batch)

# Extract results
results = batch.data.get_value()
```

## Testing

```bash
# Run all tests
uv run pytest -vv

# Run with coverage
uv run pytest -vv --cov=src/ --cov-report=term-missing

# Run specific test modules
uv run pytest tests/operators/ -vv
uv run pytest tests/pipelines/ -vv
uv run pytest tests/integration/ -vv
```

## Project Structure

```
DiffBio/
├── src/diffbio/
│   ├── operators/          # Differentiable operators
│   │   ├── quality_filter.py
│   │   ├── pileup.py
│   │   ├── smith_waterman.py
│   │   └── classifier.py
│   ├── pipelines/          # End-to-end pipelines
│   │   └── variant_calling.py
│   └── utils/              # Training utilities
│       └── training.py
└── tests/
    ├── operators/          # Unit tests for operators
    ├── pipelines/          # Pipeline tests
    └── integration/        # Integration tests
```

## Requirements

- Python 3.12+
- JAX 0.4.35+
- Flax 0.10+
- Optax 0.2.4+
- jaxtyping 0.2.36+
- Datarax (installed automatically)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

DiffBio builds on ideas from:
- [SMURF](https://www.biorxiv.org/content/10.1101/2024.04.09.588712): Differentiable sequence alignment
- [Datarax](https://github.com/mahdi-shafiei/workshop-data): Composable data processing framework
- [Flax NNX](https://flax.readthedocs.io/): Neural network library for JAX
