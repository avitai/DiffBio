# Single-Cell Losses

DiffBio provides specialized loss functions for training single-cell analysis models.

## BatchMixingLoss

Encourages batch mixing in the latent space for batch correction.

### Overview

Batch mixing loss measures how well cells from different batches are mixed in the embedding space. Higher mixing indicates better batch correction.

### Usage

```python
from diffbio.losses.singlecell_losses import BatchMixingLoss

# Create loss function
batch_loss = BatchMixingLoss(
    n_neighbors=15,
    temperature=1.0,
)

# Compute loss
loss = batch_loss(
    embeddings=cell_embeddings,  # (n_cells, embedding_dim)
    batch_ids=batch_labels,      # (n_cells,) integer batch labels
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | int | 15 | Number of neighbors for mixing calculation |
| `temperature` | float | 1.0 | Softmax temperature |

### Algorithm

The batch mixing loss computes:

1. Find k-nearest neighbors for each cell
2. Calculate batch entropy in neighborhood
3. Maximize entropy (perfect mixing = uniform batch distribution)

$$L_{batch} = -\frac{1}{N}\sum_i H(batch | neighbors_i)$$

Where $H$ is the entropy of batch distribution among neighbors.

### Training Example

```python
from flax import nnx

def harmony_loss(model, features, batch_ids):
    """Train batch correction model."""
    data = {"features": features, "batch_ids": batch_ids}
    result, _, _ = model.apply(data, {}, None)

    # Negate to maximize mixing
    return -batch_loss(result["corrected_features"], batch_ids)

grads = nnx.grad(harmony_loss)(model, features, batch_ids)
```

## ClusteringCompactnessLoss

Encourages tight, well-separated clusters.

### Overview

This loss combines intra-cluster compactness with inter-cluster separation, similar to the silhouette score but differentiable.

### Usage

```python
from diffbio.losses.singlecell_losses import ClusteringCompactnessLoss

# Create loss function
cluster_loss = ClusteringCompactnessLoss(
    temperature=1.0,
    margin=1.0,
)

# Compute loss
loss = cluster_loss(
    embeddings=cell_embeddings,        # (n_cells, embedding_dim)
    cluster_assignments=assignments,   # (n_cells, n_clusters) soft assignments
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Softmax temperature |
| `margin` | float | 1.0 | Margin for separation |

### Algorithm

The compactness loss:

$$L_{compact} = L_{intra} - L_{inter}$$

Where:
- $L_{intra}$ = average distance to cluster centroid (minimize)
- $L_{inter}$ = average distance between cluster centroids (maximize)

### Training Example

```python
def clustering_loss(model, features):
    """Train clustering model."""
    data = {"features": features}
    result, _, _ = model.apply(data, {}, None)

    return cluster_loss(
        result["embeddings"],
        result["cluster_assignments"],
    )
```

## VelocityConsistencyLoss

Ensures RNA velocity vectors are consistent with expression dynamics.

### Overview

RNA velocity consistency loss encourages velocity predictions to be consistent with observed changes in gene expression along trajectories.

### Usage

```python
from diffbio.losses.singlecell_losses import VelocityConsistencyLoss

# Create loss function
velocity_loss = VelocityConsistencyLoss(
    n_neighbors=30,
    temperature=1.0,
)

# Compute loss
loss = velocity_loss(
    embeddings=cell_embeddings,  # (n_cells, embedding_dim)
    velocities=velocity_vectors, # (n_cells, n_genes)
    expression=expression,       # (n_cells, n_genes)
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | int | 30 | Neighbors for transition matrix |
| `temperature` | float | 1.0 | Transition probability temperature |

### Algorithm

The velocity consistency loss:

1. Compute transition probabilities from velocities
2. Compare predicted transitions with observed expression changes
3. Minimize discrepancy

$$L_{vel} = \sum_{ij} T_{ij} \cdot ||x_j - x_i - v_i||^2$$

Where $T_{ij}$ is the transition probability from cell $i$ to $j$.

### Training Example

```python
def velocity_model_loss(model, spliced, unspliced):
    """Train RNA velocity model."""
    data = {"spliced": spliced, "unspliced": unspliced}
    result, _, _ = model.apply(data, {}, None)

    return velocity_loss(
        result["embeddings"],
        result["velocity"],
        spliced,
    )
```

## Combining Losses

For comprehensive single-cell analysis:

```python
from diffbio.losses.singlecell_losses import (
    BatchMixingLoss,
    ClusteringCompactnessLoss,
)

batch_loss = BatchMixingLoss(n_neighbors=15)
cluster_loss = ClusteringCompactnessLoss()

def combined_scrnaseq_loss(model, features, batch_ids):
    """Combined loss for single-cell integration."""
    data = {"features": features, "batch_ids": batch_ids}
    result, _, _ = model.apply(data, {}, None)

    # Batch mixing (maximize)
    l_batch = -batch_loss(result["embeddings"], batch_ids)

    # Clustering (minimize)
    l_cluster = cluster_loss(
        result["embeddings"],
        result["assignments"],
    )

    # Reconstruction (if using VAE)
    l_recon = jnp.mean((result["reconstructed"] - features) ** 2)

    return l_batch + 0.5 * l_cluster + l_recon
```

## Loss Weighting Guidelines

| Scenario | Batch Weight | Cluster Weight | Notes |
|----------|--------------|----------------|-------|
| Strong batch effects | 1.0 | 0.1 | Prioritize batch correction |
| Clear cell types | 0.3 | 1.0 | Prioritize clustering |
| Balanced | 0.5 | 0.5 | Equal importance |

## Next Steps

- See [Single-Cell Operators](../operators/singlecell.md) for analysis operators
- Explore [Statistical Losses](statistical.md) for count-based losses
- Check [Training Overview](../training/overview.md) for training workflows
