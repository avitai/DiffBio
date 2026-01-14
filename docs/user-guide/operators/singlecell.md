# Single-Cell Operators

DiffBio provides comprehensive differentiable operators for single-cell analysis, including clustering, batch correction, and RNA velocity.

<span class="operator-singlecell">Single-Cell</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Single-cell operators enable end-to-end optimization of:

- **SoftKMeansClustering**: Differentiable soft k-means with learnable centroids
- **DifferentiableHarmony**: Harmony-style batch correction
- **DifferentiableVelocity**: RNA velocity estimation via neural ODEs
- **DifferentiableAmbientRemoval**: VAE-based ambient RNA decontamination

## SoftKMeansClustering

Differentiable k-means clustering with soft assignments and learnable centroids.

### Quick Start

```python
from flax import nnx
from diffbio.operators.singlecell import SoftKMeansClustering, SoftClusteringConfig

# Configure clustering
config = SoftClusteringConfig(
    n_clusters=10,
    n_features=50,
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
clustering = SoftKMeansClustering(config, rngs=rngs)

# Apply to cell embeddings
data = {"embeddings": cell_embeddings}  # (n_cells, n_features)
result, state, metadata = clustering.apply(data, {}, None)

# Get results
assignments = result["cluster_assignments"]   # Soft assignments (n_cells, n_clusters)
centroids = result["centroids"]               # Learned centroids
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_clusters` | int | 10 | Number of clusters |
| `n_features` | int | 50 | Feature dimensionality |
| `temperature` | float | 1.0 | Softmax temperature |
| `learnable_centroids` | bool | True | Whether centroids are learnable |

### Soft K-Means Algorithm

Instead of hard cluster assignments:

$$p(c_k | x_i) = \frac{\exp(-d(x_i, \mu_k) / \tau)}{\sum_j \exp(-d(x_i, \mu_j) / \tau)}$$

Where $d$ is distance, $\mu_k$ are centroids, and $\tau$ is temperature.

## DifferentiableHarmony

Harmony-style batch correction for integrating multiple single-cell datasets.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableHarmony, BatchCorrectionConfig

# Configure Harmony
config = BatchCorrectionConfig(
    n_clusters=50,
    n_features=50,
    sigma=0.1,
    theta=2.0,
    n_iterations=10,
)

# Create operator
rngs = nnx.Rngs(42)
harmony = DifferentiableHarmony(config, rngs=rngs)

# Apply batch correction
data = {
    "features": cell_embeddings,  # (n_cells, n_features)
    "batch_ids": batch_labels,    # (n_cells,)
}
result, state, metadata = harmony.apply(data, {}, None)

# Get corrected embeddings
corrected = result["corrected_features"]
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_clusters` | int | 50 | Number of cluster centroids |
| `n_features` | int | 50 | Feature dimensionality |
| `sigma` | float | 0.1 | Bandwidth for soft clustering |
| `theta` | float | 2.0 | Diversity penalty strength |
| `n_iterations` | int | 10 | Number of correction iterations |

## DifferentiableVelocity

RNA velocity estimation using neural ODEs for modeling splicing dynamics.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableVelocity, VelocityConfig

# Configure velocity estimator
config = VelocityConfig(
    n_genes=2000,
    hidden_dim=64,
    n_layers=2,
    solver_steps=10,
)

# Create operator
rngs = nnx.Rngs(42)
velocity = DifferentiableVelocity(config, rngs=rngs)

# Apply to spliced/unspliced counts
data = {
    "spliced": spliced_counts,     # (n_cells, n_genes)
    "unspliced": unspliced_counts, # (n_cells, n_genes)
}
result, state, metadata = velocity.apply(data, {}, None)

# Get velocity vectors
velocities = result["velocity"]          # Gene velocity (n_cells, n_genes)
latent_time = result["latent_time"]      # Inferred pseudotime
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of genes |
| `hidden_dim` | int | 64 | ODE network hidden dimension |
| `n_layers` | int | 2 | Number of ODE network layers |
| `solver_steps` | int | 10 | ODE solver steps |

### RNA Velocity Model

Models the splicing dynamics:

$$\frac{du}{dt} = \alpha - \beta \cdot u$$
$$\frac{ds}{dt} = \beta \cdot u - \gamma \cdot s$$

Where $u$ is unspliced, $s$ is spliced, and $\alpha, \beta, \gamma$ are rate parameters.

## DifferentiableAmbientRemoval

VAE-based ambient RNA removal for cleaning droplet-based scRNA-seq data.

### Quick Start

```python
from diffbio.operators.singlecell import DifferentiableAmbientRemoval, AmbientRemovalConfig

# Configure ambient removal
config = AmbientRemovalConfig(
    n_genes=2000,
    latent_dim=20,
    hidden_dim=128,
)

# Create operator
rngs = nnx.Rngs(42)
ambient_removal = DifferentiableAmbientRemoval(config, rngs=rngs)

# Apply decontamination
data = {
    "counts": raw_counts,           # (n_cells, n_genes)
    "ambient_profile": ambient,     # (n_genes,) estimated from empty droplets
}
result, state, metadata = ambient_removal.apply(data, {}, None)

# Get decontaminated counts
clean_counts = result["decontaminated"]
contamination = result["contamination_fraction"]
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of genes |
| `latent_dim` | int | 20 | VAE latent dimension |
| `hidden_dim` | int | 128 | Encoder/decoder hidden dimension |

## Training Single-Cell Pipelines

### Combined Loss Example

```python
from diffbio.losses.singlecell_losses import (
    BatchMixingLoss,
    ClusteringCompactnessLoss,
)

batch_loss = BatchMixingLoss(n_neighbors=15, temperature=1.0)
cluster_loss = ClusteringCompactnessLoss(temperature=1.0)

def combined_loss(model, data):
    result, _, _ = model.apply(data, {}, None)

    # Batch mixing (maximize)
    l_batch = -batch_loss(result["corrected_features"], data["batch_ids"])

    # Cluster compactness (minimize)
    l_cluster = cluster_loss(
        result["corrected_features"],
        result["cluster_assignments"],
    )

    return l_batch + 0.1 * l_cluster
```

## Use Cases

| Application | Operator | Description |
|-------------|----------|-------------|
| Cell clustering | SoftKMeansClustering | Identify cell types |
| Dataset integration | DifferentiableHarmony | Merge multiple experiments |
| Trajectory inference | DifferentiableVelocity | Model differentiation |
| Data cleaning | DifferentiableAmbientRemoval | Remove ambient RNA |

## Next Steps

- See [Normalization Operators](normalization.md) for VAE-based normalization
- Explore [Single-Cell Losses](../losses/singlecell.md) for training objectives
- Check [Single-Cell Clustering Example](../../examples/basic/single-cell-clustering.md)
