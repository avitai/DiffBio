# Mapping Operators API

Differentiable operators for read mapping using neural networks.

## NeuralReadMapper

::: diffbio.operators.mapping.neural_mapper.NeuralReadMapper
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## NeuralReadMapperConfig

::: diffbio.operators.mapping.neural_mapper.NeuralReadMapperConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Neural Read Mapping

```python
from flax import nnx
from diffbio.operators.mapping import NeuralReadMapper, NeuralReadMapperConfig

config = NeuralReadMapperConfig(
    read_length=150,
    reference_window=1000,
    embedding_dim=128,
    num_layers=4,
    num_heads=8,
)
mapper = NeuralReadMapper(config, rngs=nnx.Rngs(42))

data = {
    "reads": read_sequences,       # (n_reads, read_length, alphabet_size)
    "reference": reference_seq,    # (ref_length, alphabet_size)
}
result, _, _ = mapper.apply(data, {}, None)
positions = result["positions"]
mapping_scores = result["scores"]
```

### Batch Read Mapping

```python
# Process multiple reads at once
reads = jnp.stack([read1, read2, read3])  # (3, read_length, 4)

data = {"reads": reads, "reference": reference}
result, _, _ = mapper.apply(data, {}, None)

# Get mapping positions for all reads
all_positions = result["positions"]  # (3,)
```
