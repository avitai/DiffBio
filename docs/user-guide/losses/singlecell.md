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
    n_batches=3,
    temperature=1.0,
)

# Compute loss (positional: embeddings, batch_labels)
loss = batch_loss(
    cell_embeddings,  # (n_cells, embedding_dim)
    batch_labels,     # (n_cells,) integer batch labels
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | int | 15 | Number of neighbors for mixing calculation |
| `n_batches` | int | 3 | Number of batches (static for JIT compatibility) |
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
    separation_weight=1.0,
    min_separation=1.0,
)

# Compute loss (positional: embeddings, soft assignments)
loss = cluster_loss(
    cell_embeddings,  # (n_cells, embedding_dim)
    assignments,      # (n_cells, n_clusters) soft assignments
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `separation_weight` | float | 1.0 | Weight for the inter-cluster separation term |
| `min_separation` | float | 1.0 | Minimum desired distance between cluster centroids |

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
    dt=0.1,
    cosine_weight=1.0,
    magnitude_weight=1.0,
)

# Compute loss (positional: expression, velocity, future_expression)
loss = velocity_loss(
    expression,         # (n_cells, n_genes) current expression
    velocity_vectors,   # (n_cells, n_genes) predicted velocity
    future_expression,  # (n_cells, n_genes) ground-truth or estimated future expression
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dt` | float | 0.1 | Time step for velocity extrapolation |
| `cosine_weight` | float | 1.0 | Weight for the directional (cosine) consistency term |
| `magnitude_weight` | float | 1.0 | Weight for the magnitude consistency term |

### Algorithm

The velocity consistency loss:

1. Compute transition probabilities from velocities
2. Compare predicted transitions with observed expression changes
3. Minimize discrepancy

$$L_{vel} = \sum_{ij} T_{ij} \cdot ||x_j - x_i - v_i||^2$$

Where $T_{ij}$ is the transition probability from cell $i$ to $j$.

### Training Example

```python
def velocity_model_loss(model, spliced, unspliced, future_spliced):
    """Train RNA velocity model."""
    data = {"spliced": spliced, "unspliced": unspliced}
    result, _, _ = model.apply(data, {}, None)

    return velocity_loss(
        spliced,
        result["velocity"],
        future_spliced,
    )
```

## Combining Losses

For full single-cell analysis:

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

## ShannonDiversityLoss

Measures assignment diversity using Shannon entropy of soft cluster assignments. Higher values indicate more uniform (diverse) cluster assignments, lower values indicate concentrated assignments. Delegates to `calibrax.metrics.functional.information.entropy` for per-cell computation.

### Usage

```python
from diffbio.losses.singlecell_losses import ShannonDiversityLoss

diversity_loss = ShannonDiversityLoss()

# Soft cluster probabilities: (n_cells, n_clusters)
assignments = jax.nn.softmax(logits, axis=-1)
diversity = diversity_loss(assignments)  # scalar, range [0, log(K)]
```

### Parameters

ShannonDiversityLoss has no configuration parameters.

### Algorithm

$$H = -\frac{1}{N}\sum_i \sum_k p_{ik} \log(p_{ik})$$

Where $p_{ik}$ is the soft assignment probability of cell $i$ to cluster $k$. Maximum entropy $\log(K)$ occurs with uniform assignments.

### Use Cases

- Regularize clustering to avoid degenerate solutions (all cells in one cluster)
- Encourage balanced cluster sizes during training
- Combine with compactness loss: `loss = compactness - lambda * diversity`

## SimpsonDiversityLoss

Mean Simpson concentration index of soft cluster assignments. Computes the sum of squared assignment probabilities per cell, averaged across all cells. Lower values indicate more diverse (uniform) assignments.

### Usage

```python
from diffbio.losses.singlecell_losses import SimpsonDiversityLoss

simpson_loss = SimpsonDiversityLoss()

assignments = jax.nn.softmax(logits, axis=-1)
concentration = simpson_loss(assignments)  # scalar, range [1/K, 1.0]
```

### Parameters

SimpsonDiversityLoss has no configuration parameters.

### Algorithm

$$D = \frac{1}{N}\sum_i \sum_k p_{ik}^2$$

- Uniform assignments over $K$ clusters yield $1/K$
- Fully concentrated (one-hot) assignments yield $1.0$

### Use Cases

- Alternative diversity regularizer to Shannon entropy
- More sensitive to dominant clusters than Shannon entropy
- Minimize Simpson index to encourage diverse cluster usage

## Next Steps

- See [Single-Cell Operators](../operators/singlecell.md) for analysis operators
- Explore [Statistical Losses](statistical.md) for count-based losses
- Check [Metric Losses](metric.md) for AUROC training surrogates
- Check [Training Overview](../training/overview.md) for training workflows
