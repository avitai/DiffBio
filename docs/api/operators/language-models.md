# Language Model Operators API

Differentiable transformer-based operators for DNA/RNA sequence embedding.

## TransformerSequenceEncoder

::: diffbio.operators.language_models.transformer_encoder.TransformerSequenceEncoder
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply
        - get_positional_encoding

## TransformerSequenceEncoderConfig

::: diffbio.operators.language_models.transformer_encoder.TransformerSequenceEncoderConfig
    options:
      show_root_heading: true
      members: []

## Factory Functions

### create_dna_encoder

::: diffbio.operators.language_models.transformer_encoder.create_dna_encoder
    options:
      show_root_heading: true

### create_rna_encoder

::: diffbio.operators.language_models.transformer_encoder.create_rna_encoder
    options:
      show_root_heading: true

## DifferentiableFoundationModel

::: diffbio.operators.language_models.foundation_model.DifferentiableFoundationModel
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## FoundationModelConfig

::: diffbio.operators.language_models.foundation_model.FoundationModelConfig
    options:
      show_root_heading: true
      members: []

## GeneTokenizer

::: diffbio.operators.language_models.foundation_model.GeneTokenizer
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## Usage Examples

### Basic Usage

```python
from diffbio.operators.language_models import create_dna_encoder
import jax
import jax.numpy as jnp

# Create encoder
encoder = create_dna_encoder(hidden_dim=256, num_layers=4)

# Prepare one-hot encoded sequence
sequence = jax.nn.one_hot(
    jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 4),
    num_classes=4,
)

# Apply
result, _, _ = encoder.apply({"sequence": sequence}, {}, None)
embedding = result["embedding"]  # (256,)
```

### Full Configuration

```python
from diffbio.operators.language_models import (
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
)
from flax import nnx

config = TransformerSequenceEncoderConfig(
    hidden_dim=640,
    num_layers=12,
    num_heads=20,
    intermediate_dim=5120,
    max_length=1024,
    alphabet_size=4,
    dropout_rate=0.1,
    pooling="cls",
)

encoder = TransformerSequenceEncoder(config, rngs=nnx.Rngs(42))
```

### Batched Processing

```python
import jax.numpy as jnp

# Batch of sequences
batch_size = 8
seq_len = 100
sequences = jax.nn.one_hot(
    jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, 4),
    num_classes=4,
)

result, _, _ = encoder.apply({"sequence": sequences}, {}, None)
embeddings = result["embedding"]  # (8, 256)
```

### Gradient Computation

```python
import jax
from flax import nnx

encoder = create_dna_encoder()

def loss_fn(model, sequence):
    result, _, _ = model.apply({"sequence": sequence}, {}, None)
    return result["embedding"].sum()

# Compute gradients w.r.t. model parameters
_, grads = nnx.value_and_grad(loss_fn)(encoder, sequence)
```

## Input Specifications

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `sequence` | (length, 4) or (batch, length, 4) | float32 | One-hot encoded nucleotide sequence |
| `attention_mask` | (length,) or (batch, length) | float32 | Optional mask (1=valid, 0=padded) |

## Output Specifications

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `sequence` | same as input | float32 | Original input sequence |
| `embedding` | (hidden_dim,) or (batch, hidden_dim) | float32 | Global sequence embedding |
| `position_embeddings` | (length, hidden_dim) or (batch, length, hidden_dim) | float32 | Per-position hidden states |

## Reference Configurations

| Model | hidden_dim | num_layers | num_heads | intermediate_dim |
|-------|------------|------------|-----------|------------------|
| DNABERT | 768 | 12 | 12 | 3072 |
| RNA-FM | 640 | 12 | 20 | 5120 |
| Default | 256 | 4 | 4 | 1024 |
