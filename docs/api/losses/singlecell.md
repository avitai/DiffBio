# Single-Cell Losses API

Loss functions for training single-cell analysis models.

## BatchMixingLoss

::: diffbio.losses.singlecell_losses.BatchMixingLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## ClusteringCompactnessLoss

::: diffbio.losses.singlecell_losses.ClusteringCompactnessLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## VelocityConsistencyLoss

::: diffbio.losses.singlecell_losses.VelocityConsistencyLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## ShannonDiversityLoss

::: diffbio.losses.singlecell_losses.ShannonDiversityLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## SimpsonDiversityLoss

::: diffbio.losses.singlecell_losses.SimpsonDiversityLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - __call__

## Usage Examples

### Batch Mixing Loss

```python
from diffbio.losses import BatchMixingLoss

batch_loss = BatchMixingLoss(n_neighbors=15, temperature=1.0)

# Maximize batch mixing in latent space
loss = batch_loss(
    embeddings=latent_embeddings,  # (n_cells, latent_dim)
    batch_labels=batch_labels,     # (n_cells,)
)
```

### Clustering Compactness Loss

```python
from diffbio.losses import ClusteringCompactnessLoss

cluster_loss = ClusteringCompactnessLoss(separation_weight=1.0, min_separation=1.0)

# Encourage tight clusters
loss = cluster_loss(
    embeddings=cell_embeddings,
    assignments=soft_assignments,
)
```

### Combined Training

```python
def combined_loss(model, data):
    result, _, _ = model.apply(data, {}, None)

    # Batch mixing (negate to maximize)
    l_batch = -batch_loss(result["embeddings"], data["batch_ids"])

    # Clustering compactness
    l_cluster = cluster_loss(result["embeddings"], result["assignments"])

    return l_batch + 0.5 * l_cluster
```
