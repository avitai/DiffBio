# Normalization Operators API

Differentiable normalization operators for count data, dimensionality reduction, and embeddings.

## VAENormalizer

::: diffbio.operators.normalization.vae_normalizer.VAENormalizer
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## VAENormalizerConfig

::: diffbio.operators.normalization.vae_normalizer.VAENormalizerConfig
    options:
      show_root_heading: true
      members: []

## DifferentiableUMAP

::: diffbio.operators.normalization.umap.DifferentiableUMAP
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## UMAPConfig

::: diffbio.operators.normalization.umap.UMAPConfig
    options:
      show_root_heading: true
      members: []

## SequenceEmbedding

::: diffbio.operators.normalization.embedding.SequenceEmbedding
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## SequenceEmbeddingConfig

::: diffbio.operators.normalization.embedding.SequenceEmbeddingConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### VAE Normalization

```python
from flax import nnx
from diffbio.operators.normalization import VAENormalizer, VAENormalizerConfig

config = VAENormalizerConfig(n_genes=2000, latent_dim=10)
vae = VAENormalizer(config, rngs=nnx.Rngs(42))

data = {"counts": raw_counts}  # (n_cells, n_genes)
result, _, _ = vae.apply(data, {}, None)
normalized = result["normalized"]
```

### UMAP Dimensionality Reduction

```python
from diffbio.operators.normalization import DifferentiableUMAP, UMAPConfig

config = UMAPConfig(n_components=2, n_neighbors=15, input_features=50)
umap = DifferentiableUMAP(config, rngs=nnx.Rngs(42))

data = {"features": high_dim_data}  # (n_samples, n_features)
result, _, _ = umap.apply(data, {}, None)
embedding = result["embedding"]
```

### Sequence Embedding

```python
from diffbio.operators.normalization import SequenceEmbedding, SequenceEmbeddingConfig

config = SequenceEmbeddingConfig(embedding_dim=64, max_length=100)
seq_embed = SequenceEmbedding(config, rngs=nnx.Rngs(42))

data = {"sequences": sequences}
result, _, _ = seq_embed.apply(data, {}, None)
embeddings = result["embeddings"]
```
