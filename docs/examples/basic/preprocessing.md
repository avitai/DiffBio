# Preprocessing Example

This example demonstrates the differentiable preprocessing pipeline for sequence read data.

## Setup

```python
import jax
import jax.numpy as jnp
from flax import nnx
from diffbio.pipelines import PreprocessingPipeline, PreprocessingPipelineConfig
```

## Generate Synthetic Read Data

```python
def generate_synthetic_reads(n_reads=100, read_length=50, seed=42):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 3)

    # One-hot encoded sequences
    sequence_indices = jax.random.randint(keys[0], (n_reads, read_length), 0, 4)
    sequences = jax.nn.one_hot(sequence_indices, 4)

    # Quality scores (Phred scale)
    quality_scores = jax.random.uniform(keys[1], (n_reads, read_length), minval=10.0, maxval=40.0)

    return sequences, quality_scores

sequences, quality_scores = generate_synthetic_reads(n_reads=100, read_length=50)
print(f"Sequences shape: {sequences.shape}")        # (100, 50, 4)
print(f"Quality scores shape: {quality_scores.shape}")  # (100, 50)
```

## Create Preprocessing Pipeline

```python
config = PreprocessingPipelineConfig(
    read_length=50,
    quality_threshold=20.0,
    enable_adapter_removal=True,
    enable_duplicate_weighting=True,
    enable_error_correction=True,
)

rngs = nnx.Rngs(42)
pipeline = PreprocessingPipeline(config, rngs=rngs)
```

## Run Preprocessing

```python
data = {
    "reads": sequences,
    "quality": quality_scores,
}

result, state, metadata = pipeline.apply(data, {}, None)

# Access outputs
read_weights = result["read_weights"]
preprocessed_reads = result["preprocessed_reads"]

print(f"Read weights range: [{read_weights.min():.3f}, {read_weights.max():.3f}]")
print(f"Preprocessed reads shape: {preprocessed_reads.shape}")
```

## Differentiability

```python
def preprocessing_loss(pipeline, reads, quality):
    data = {"reads": reads, "quality": quality}
    result, _, _ = pipeline.apply(data, {}, None)
    # Use read weights as quality metric
    return -result["read_weights"].mean()

loss_val, grads = nnx.value_and_grad(preprocessing_loss)(pipeline, sequences, quality_scores)
print(f"Loss: {loss_val:.4f}")
```

## Next Steps

- See [Variant Calling Pipeline](../advanced/variant-calling.md) for downstream analysis
- Explore [Preprocessing Operators](../../user-guide/operators/preprocessing.md) for details
