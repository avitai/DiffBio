# Preprocessing Pipeline API

End-to-end differentiable preprocessing pipeline.

## PreprocessingPipeline

::: diffbio.pipelines.preprocessing.PreprocessingPipeline
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## PreprocessingPipelineConfig

::: diffbio.pipelines.preprocessing.PreprocessingPipelineConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Basic Usage

```python
from flax import nnx
from diffbio.pipelines import PreprocessingPipeline, PreprocessingPipelineConfig

config = PreprocessingPipelineConfig(
    quality_threshold=20.0,
    enable_adapter_removal=True,
    enable_duplicate_weighting=True,
    enable_error_correction=True,
)

pipeline = PreprocessingPipeline(config, rngs=nnx.Rngs(42))

data = {
    "reads": read_sequences,    # (n_reads, read_length, 4)
    "quality": quality_scores,  # (n_reads, read_length)
}
result, _, _ = pipeline.apply(data, {}, None)

preprocessed = result["preprocessed_reads"]
weights = result["read_weights"]
```

### Training

```python
from flax import nnx

def loss_fn(pipeline, reads, quality, target):
    data = {"reads": reads, "quality": quality}
    result, _, _ = pipeline.apply(data, {}, None)
    return jnp.mean((result["preprocessed_reads"] - target) ** 2)

grads = nnx.grad(loss_fn)(pipeline, reads, quality, corrected_target)
```
