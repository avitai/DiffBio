# Single-Cell Analysis Pipeline

The `SingleCellPipeline` is an end-to-end differentiable pipeline for single-cell RNA-seq analysis, featuring scVI-style VAE normalization, Harmony batch correction, and soft k-means clustering.

## Overview

The pipeline processes single-cell RNA-seq data through five stages:

```mermaid
graph LR
    A[Counts] --> B[Ambient Removal]
    B --> C[VAE Normalization]
    C --> D[Batch Correction]
    D --> E[UMAP]
    E --> F[Clustering]
    F --> G[Results]

    style A fill:#0d9488,color:#fff
    style B fill:#6366f1,color:#fff
    style C fill:#0891b2,color:#fff
    style D fill:#7c3aed,color:#fff
    style E fill:#f59e0b,color:#fff
    style F fill:#ec4899,color:#fff
    style G fill:#059669,color:#fff
```

1. **Ambient RNA Removal** (optional): CellBender-style VAE decontamination
2. **VAE Normalization**: scVI-style variational autoencoder for count normalization
3. **Batch Correction** (optional): Harmony-style integration across batches
4. **Dimensionality Reduction** (optional): Parametric UMAP for visualization
5. **Soft Clustering** (optional): Differentiable soft k-means clustering

## Quick Start

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
n_batches = 3

data = {
    "counts": jax.random.poisson(jax.random.PRNGKey(0), lam=5.0, shape=(n_cells, n_genes)).astype(jnp.float32),
    "ambient_profile": jax.random.uniform(jax.random.PRNGKey(1), (n_genes,)),
    "batch_labels": jax.random.randint(jax.random.PRNGKey(2), (n_cells,), 0, n_batches),
}
data["ambient_profile"] = data["ambient_profile"] / data["ambient_profile"].sum()

# Run pipeline
result, _, _ = pipeline.apply(data, {}, None)

print(f"Normalized shape: {result['normalized'].shape}")        # (100, 2000)
print(f"Latent shape: {result['latent'].shape}")                # (100, 64)
print(f"Embeddings 2D shape: {result['embeddings_2d'].shape}")  # (100, 2)
print(f"Clusters shape: {result['cluster_assignments'].shape}") # (100, 10)
```

## Configuration

### SingleCellPipelineConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of genes in the expression matrix |
| `n_clusters` | int | 10 | Number of clusters for soft k-means |
| `latent_dim` | int | 64 | Dimension of the VAE latent space |
| `hidden_dims` | tuple[int, ...] | (128, 64) | Hidden layer dimensions for VAE |
| `umap_n_components` | int | 2 | Number of UMAP output dimensions |
| `batch_correction_clusters` | int | 100 | Number of clusters for Harmony |
| `batch_correction_iterations` | int | 10 | Number of Harmony iterations |
| `clustering_temperature` | float | 1.0 | Temperature for soft clustering |
| `enable_ambient_removal` | bool | True | Enable ambient RNA removal |
| `enable_batch_correction` | bool | True | Enable batch correction |
| `enable_dim_reduction` | bool | True | Enable UMAP dimensionality reduction |
| `enable_clustering` | bool | True | Enable soft clustering |

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
```

## Input Format

The pipeline expects a dictionary with three keys:

### counts

Raw count matrix with shape `(n_cells, n_genes)`:

```python
# From AnnData
counts = jnp.array(adata.X.toarray())  # (n_cells, n_genes)

# Or synthetic
counts = jax.random.poisson(key, lam=5.0, shape=(n_cells, n_genes))
```

### ambient_profile

Ambient expression profile with shape `(n_genes,)`:

```python
# Estimated ambient profile (e.g., from empty droplets)
ambient_profile = empty_droplets.mean(axis=0)
ambient_profile = ambient_profile / ambient_profile.sum()  # Normalize

# Or uniform prior
ambient_profile = jnp.ones(n_genes) / n_genes
```

### batch_labels

Batch assignments for each cell with shape `(n_cells,)`:

```python
# Integer batch labels
batch_labels = jnp.array([0, 0, 1, 1, 2, ...])  # (n_cells,)

# From metadata
batch_labels = jnp.array(adata.obs["batch"].cat.codes)
```

## Output Format

The pipeline returns a dictionary with:

| Key | Shape | Description |
|-----|-------|-------------|
| `counts` | (n_cells, n_genes) | Original count matrix |
| `ambient_profile` | (n_genes,) | Original ambient profile |
| `batch_labels` | (n_cells,) | Original batch labels |
| `decontaminated_counts` | (n_cells, n_genes) | Ambient-removed counts* |
| `normalized` | (n_cells, n_genes) | VAE-normalized expression |
| `latent` | (n_cells, latent_dim) | Latent space representation |
| `corrected_embeddings` | (n_cells, latent_dim) | Batch-corrected embeddings |
| `embeddings_2d` | (n_cells, umap_n_components) | 2D visualization embeddings |
| `cluster_assignments` | (n_cells, n_clusters) | Soft cluster assignments |

*Only present when `enable_ambient_removal=True`

## Pipeline Stages

### Stage 1: Ambient RNA Removal (Optional)

CellBender-style VAE decontamination:

```python
# Learns to separate cell signal from ambient contamination
# VAE models: P(x | z) = cell_signal(z) + ambient_fraction * ambient_profile
decontaminated = vae_decoder(z) * (1 - ambient_fraction)
```

Access the component:

```python
if pipeline.ambient_removal is not None:
    pipeline.ambient_removal  # DifferentiableAmbientRemoval
```

### Stage 2: VAE Normalization

scVI-style variational autoencoder:

```python
# Encode counts to latent space
z_mean, z_logvar = encoder(counts)
z = reparameterize(z_mean, z_logvar)

# Decode to normalized expression
normalized = decoder(z)
```

Key features:

- Handles count data with negative binomial likelihood
- Library size normalization
- Learns batch-invariant representations

Access the component:

```python
pipeline.vae_normalizer  # VAENormalizer
```

### Stage 3: Batch Correction (Optional)

Harmony-style integration:

```python
# Iteratively adjust embeddings to remove batch effects
# while preserving biological variation
for iteration in range(n_iterations):
    # Soft cluster assignment
    assignments = soft_kmeans(embeddings)
    # Compute correction factor per cluster per batch
    correction = compute_harmony_correction(assignments, batch_labels)
    # Apply correction
    embeddings = embeddings - correction
```

Access the component:

```python
if pipeline.batch_correction is not None:
    pipeline.batch_correction  # DifferentiableHarmony
```

### Stage 4: Dimensionality Reduction (Optional)

Parametric UMAP for visualization:

```python
# Neural network learns UMAP-like embedding
embeddings_2d = umap_network(corrected_embeddings)
```

Key features:

- Trainable parameters enable gradient flow
- Preserves local structure
- Produces consistent embeddings for new data

Access the component:

```python
if pipeline.dim_reduction is not None:
    pipeline.dim_reduction  # DifferentiableUMAP
```

### Stage 5: Soft Clustering (Optional)

Differentiable soft k-means:

```python
# Soft assignment based on distance to centroids
distances = compute_distances(embeddings, centroids)
assignments = softmax(-distances / temperature)
```

Key features:

- Temperature controls cluster sharpness
- Fully differentiable for end-to-end training
- Learnable cluster centroids

Access the component:

```python
if pipeline.clustering is not None:
    pipeline.clustering  # SoftKMeansClustering
    pipeline.clustering.centroids  # Cluster centroids
```

## Optional Components

Enable or disable pipeline stages with configuration flags:

```python
# Minimal pipeline (just VAE normalization)
minimal_pipeline = create_single_cell_pipeline(
    n_genes=2000,
    enable_ambient_removal=False,
    enable_batch_correction=False,
    enable_dim_reduction=False,
    enable_clustering=False,
)

# Full pipeline (all stages)
full_pipeline = create_single_cell_pipeline(
    n_genes=2000,
    enable_ambient_removal=True,
    enable_batch_correction=True,
    enable_dim_reduction=True,
    enable_clustering=True,
)
```

## Training

### Supervised Cell Type Classification

```python
from diffbio.pipelines import create_single_cell_pipeline
from diffbio.losses import ClusteringCompactnessLoss
import optax
from flax import nnx

pipeline = create_single_cell_pipeline(n_genes=2000, n_clusters=10)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(nnx.state(pipeline, nnx.Param))

def loss_fn(pipeline, data, labels):
    result, _, _ = pipeline.apply(data, {}, None)
    # Cross-entropy with true cell type labels
    log_probs = jnp.log(result["cluster_assignments"] + 1e-10)
    return -jnp.mean(jnp.sum(labels * log_probs, axis=-1))

@jax.jit
def train_step(pipeline, opt_state, data, labels):
    loss, grads = jax.value_and_grad(loss_fn)(pipeline, data, labels)
    params = nnx.state(pipeline, nnx.Param)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    nnx.update(pipeline, optax.apply_updates(params, updates))
    return loss, opt_state

# Training loop
pipeline.train_mode()
for epoch in range(100):
    loss, opt_state = train_step(pipeline, opt_state, train_data, train_labels)
pipeline.eval_mode()
```

### Unsupervised Clustering with VAE Loss

```python
from diffbio.losses import VAELoss, ClusteringCompactnessLoss

vae_loss = VAELoss(kl_weight=0.1)
cluster_loss = ClusteringCompactnessLoss()

def combined_loss(pipeline, data):
    result, _, _ = pipeline.apply(data, {}, None)

    # VAE reconstruction + KL divergence
    vae_l = vae_loss(result["normalized"], data["counts"], ...)

    # Clustering compactness
    cluster_l = cluster_loss(
        result["corrected_embeddings"],
        result["cluster_assignments"],
    )

    return vae_l + 0.1 * cluster_l
```

## Inference

### Single Sample

```python
pipeline.eval_mode()

result, _, _ = pipeline.apply(data, {}, None)

# Get hard cluster assignments
hard_assignments = jnp.argmax(result["cluster_assignments"], axis=-1)

# Get cell type probabilities
cell_type_probs = result["cluster_assignments"]
print(f"Cell 0 cluster probabilities: {cell_type_probs[0]}")
```

### Visualization

```python
import matplotlib.pyplot as plt

result, _, _ = pipeline.apply(data, {}, None)

# 2D UMAP visualization colored by cluster
embeddings = result["embeddings_2d"]
clusters = jnp.argmax(result["cluster_assignments"], axis=-1)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings[:, 0],
    embeddings[:, 1],
    c=clusters,
    cmap="tab10",
    s=10,
    alpha=0.7,
)
plt.colorbar(scatter, label="Cluster")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("Single-Cell Clustering")
plt.show()
```

### New Data Projection

```python
# Project new cells using trained pipeline
pipeline.eval_mode()
new_result, _, _ = pipeline.apply(new_data, {}, None)

# Clusters are assigned based on trained centroids
new_clusters = jnp.argmax(new_result["cluster_assignments"], axis=-1)
```

## Accessing Components

The pipeline exposes its sub-components for inspection:

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

## Integration with Scanpy/AnnData

```python
import scanpy as sc
import jax.numpy as jnp

# Load data
adata = sc.read_h5ad("data.h5ad")

# Prepare input
data = {
    "counts": jnp.array(adata.X.toarray()),
    "ambient_profile": jnp.ones(adata.n_vars) / adata.n_vars,
    "batch_labels": jnp.array(adata.obs["batch"].cat.codes),
}

# Run pipeline
result, _, _ = pipeline.apply(data, {}, None)

# Store results back in AnnData
adata.obsm["X_latent"] = result["latent"]
adata.obsm["X_umap_diffbio"] = result["embeddings_2d"]
adata.obs["cluster_diffbio"] = jnp.argmax(result["cluster_assignments"], axis=-1)
```

## References

1. Lopez, R. et al. (2018). "Deep generative modeling for single-cell transcriptomics." *Nature Methods*. - scVI methodology.

2. Fleming, S.J. et al. (2022). "Unsupervised removal of systematic background noise from droplet-based single-cell experiments using CellBender." *Nature Methods*.

3. Korsunsky, I. et al. (2019). "Fast, sensitive and accurate integration of single-cell data with Harmony." *Nature Methods*.

4. Sainburg, T. et al. (2021). "Parametric UMAP Embeddings for Representation and Semisupervised Learning." *Neural Computation*.
