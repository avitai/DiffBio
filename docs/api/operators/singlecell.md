# Single-Cell Operators API

Differentiable operators for single-cell analysis including clustering, batch correction, and RNA velocity.

## SoftKMeansClustering

::: diffbio.operators.singlecell.soft_clustering.SoftKMeansClustering
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## SoftClusteringConfig

::: diffbio.operators.singlecell.soft_clustering.SoftClusteringConfig
    options:
      show_root_heading: true
      members: []

## DifferentiableHarmony

::: diffbio.operators.singlecell.batch_correction.DifferentiableHarmony
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## BatchCorrectionConfig

::: diffbio.operators.singlecell.batch_correction.BatchCorrectionConfig
    options:
      show_root_heading: true
      members: []

## DifferentiableVelocity

::: diffbio.operators.singlecell.velocity.DifferentiableVelocity
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## VelocityConfig

::: diffbio.operators.singlecell.velocity.VelocityConfig
    options:
      show_root_heading: true
      members: []

## DifferentiableAmbientRemoval

::: diffbio.operators.singlecell.ambient_removal.DifferentiableAmbientRemoval
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## AmbientRemovalConfig

::: diffbio.operators.singlecell.ambient_removal.AmbientRemovalConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Soft K-Means Clustering

```python
from flax import nnx
from diffbio.operators.singlecell import SoftKMeansClustering, SoftClusteringConfig

config = SoftClusteringConfig(n_clusters=10, n_embeddings=50)
clustering = SoftKMeansClustering(config, rngs=nnx.Rngs(42))

data = {"embeddings": embeddings}  # (n_cells, n_embeddings)
result, _, _ = clustering.apply(data, {}, None)
assignments = result["cluster_assignments"]
```

### Batch Correction

```python
from diffbio.operators.singlecell import DifferentiableHarmony, BatchCorrectionConfig

config = BatchCorrectionConfig(n_clusters=50, n_embeddings=50)
harmony = DifferentiableHarmony(config, rngs=nnx.Rngs(42))

data = {"embeddings": embeddings, "batch_ids": batch_labels}
result, _, _ = harmony.apply(data, {}, None)
corrected = result["corrected_embeddings"]
```

### RNA Velocity

```python
from diffbio.operators.singlecell import DifferentiableVelocity, VelocityConfig

config = VelocityConfig(n_genes=2000, hidden_dim=64)
velocity = DifferentiableVelocity(config, rngs=nnx.Rngs(42))

data = {"spliced": spliced, "unspliced": unspliced}
result, _, _ = velocity.apply(data, {}, None)
vel = result["velocity"]
```
