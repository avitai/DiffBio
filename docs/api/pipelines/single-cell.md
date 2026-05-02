# Single-Cell Pipeline API

End-to-end differentiable single-cell RNA-seq analysis pipeline with scVI-style VAE, Harmony batch correction, and soft clustering.

## SingleCellPipeline

::: diffbio.pipelines.single_cell.SingleCellPipeline
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## SingleCellPipelineConfig

::: diffbio.pipelines.single_cell.SingleCellPipelineConfig
    options:
      show_root_heading: true
      members: []

## Factory Function

### create_single_cell_pipeline

::: diffbio.pipelines.single_cell.create_single_cell_pipeline
    options:
      show_root_heading: true

## Usage Examples

### Quick Start

```python
from diffbio.pipelines import create_single_cell_pipeline
import jax
import jax.numpy as jnp

# Create pipeline
pipeline = create_single_cell_pipeline(
    n_genes=2000,
    n_clusters=10,
    latent_dim=64,
)

# Prepare data
n_cells = 100
n_genes = 2000

data = {
    "counts": jax.random.poisson(
        jax.random.PRNGKey(0), lam=5.0, shape=(n_cells, n_genes)
    ).astype(jnp.float32),
    "ambient_profile": jnp.ones(n_genes) / n_genes,
    "batch_labels": jax.random.randint(jax.random.PRNGKey(1), (n_cells,), 0, 3),
}

# Run pipeline
result, _, _ = pipeline.apply(data, {}, None)
clusters = jnp.argmax(result["cluster_assignments"], axis=-1)
```

### Full Configuration

```python
from diffbio.pipelines import SingleCellPipeline, SingleCellPipelineConfig
from flax import nnx

config = SingleCellPipelineConfig(
    n_genes=5000,
    n_clusters=15,
    latent_dim=128,
    hidden_dims=(256, 128),
    umap_n_components=2,
    batch_correction_clusters=50,
    batch_correction_iterations=20,
    clustering_temperature=0.5,
    enable_ambient_removal=True,
    enable_batch_correction=True,
    enable_dim_reduction=True,
    enable_clustering=True,
)

pipeline = SingleCellPipeline(config, rngs=nnx.Rngs(42))
# Note: this pipeline has no training-mode toggle; submodules manage their
# own dropout/training state when applicable.
```

### Training Mode

```python
# SingleCellPipeline does not expose train_mode/eval_mode toggles.
# Submodules that use dropout manage their own state during apply().
for batch in dataloader:
    loss = train_step(pipeline, batch)
```

### Access Components

```python
# Ambient removal (if enabled)
if pipeline.ambient_removal is not None:
    pipeline.ambient_removal

# VAE normalizer (always available)
pipeline.vae_normalizer

# Batch correction (if enabled)
if pipeline.batch_correction is not None:
    pipeline.batch_correction

# Dimensionality reduction (if enabled)
if pipeline.dim_reduction is not None:
    pipeline.dim_reduction

# Clustering (if enabled)
if pipeline.clustering is not None:
    pipeline.clustering
    pipeline.clustering.centroids  # Cluster centers
```

### Minimal Pipeline

```python
# Create pipeline with only VAE normalization
minimal_pipeline = create_single_cell_pipeline(
    n_genes=2000,
    enable_ambient_removal=False,
    enable_batch_correction=False,
    enable_dim_reduction=False,
    enable_clustering=False,
)
```

## Input Specifications

| Key | Shape | Description |
|-----|-------|-------------|
| `counts` | (n_cells, n_genes) | Raw count matrix |
| `ambient_profile` | (n_genes,) | Ambient expression profile (normalized) |
| `batch_labels` | (n_cells,) | Integer batch assignments |

## Output Specifications

| Key | Shape | Description |
|-----|-------|-------------|
| `counts` | (n_cells, n_genes) | Original count matrix |
| `ambient_profile` | (n_genes,) | Original ambient profile |
| `batch_labels` | (n_cells,) | Original batch labels |
| `decontaminated_counts` | (n_cells, n_genes) | Ambient-removed counts* |
| `normalized` | (n_cells, n_genes) | VAE-normalized expression |
| `latent` | (n_cells, latent_dim) | Latent space representation |
| `corrected_embeddings` | (n_cells, latent_dim) | Batch-corrected embeddings |
| `embeddings_2d` | (n_cells, umap_n_components) | 2D UMAP embeddings |
| `cluster_assignments` | (n_cells, n_clusters) | Soft cluster probabilities |

*Only present when `enable_ambient_removal=True`
