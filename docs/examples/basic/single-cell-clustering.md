# Single-Cell Clustering Example

This example demonstrates differentiable single-cell clustering using DiffBio's soft k-means operator with end-to-end gradient optimization.

## Setup

```python
import jax
import jax.numpy as jnp
from flax import nnx
from diffbio.operators.singlecell import SoftKMeansClustering, SoftClusteringConfig
```

## Generate Synthetic Single-Cell Data

```python
def generate_synthetic_cells(n_cells=1000, n_features=50, n_clusters=5, seed=42):
    """Generate synthetic single-cell embedding data."""
    key = jax.random.key(seed)
    keys = jax.random.split(key, n_clusters + 1)

    # Generate cluster centers
    centers = jax.random.normal(keys[0], (n_clusters, n_features)) * 3

    # Assign cells to clusters
    cells_per_cluster = n_cells // n_clusters
    cells = []
    labels = []

    for i in range(n_clusters):
        noise = jax.random.normal(keys[i + 1], (cells_per_cluster, n_features)) * 0.5
        cluster_cells = centers[i] + noise
        cells.append(cluster_cells)
        labels.append(jnp.full(cells_per_cluster, i))

    return jnp.vstack(cells), jnp.concatenate(labels)

# Create dataset
cell_embeddings, true_labels = generate_synthetic_cells(
    n_cells=1000,
    n_features=50,
    n_clusters=5,
)

print(f"Cell embeddings shape: {cell_embeddings.shape}")  # (1000, 50)
print(f"True labels shape: {true_labels.shape}")          # (1000,)
```

## Create the Clustering Operator

```python
# Configure soft k-means clustering
config = SoftClusteringConfig(
    n_clusters=5,
    n_features=50,
    temperature=1.0,
)

# Create operator with random number generator
rngs = nnx.Rngs(42)
clustering = SoftKMeansClustering(config, rngs=rngs)
```

## Perform Clustering

```python
# Prepare data dictionary
data = {"embeddings": cell_embeddings}

# Apply clustering
result, state, metadata = clustering.apply(data, {}, None)

# Get results
soft_assignments = result["cluster_assignments"]   # (n_cells, n_clusters)
centroids = result["centroids"]                    # (n_clusters, n_features)

print(f"Soft assignments shape: {soft_assignments.shape}")
print(f"Centroids shape: {centroids.shape}")

# Hard assignments (most likely cluster)
hard_assignments = jnp.argmax(soft_assignments, axis=1)
```

## Differentiability: Computing Gradients

The key advantage of soft k-means is differentiability:

```python
from diffbio.losses.singlecell_losses import ClusteringCompactnessLoss

compactness_loss = ClusteringCompactnessLoss()

def loss_fn(clustering_op, embeddings):
    data = {"embeddings": embeddings}
    result, _, _ = clustering_op.apply(data, {}, None)
    return compactness_loss(
        embeddings=embeddings,
        assignments=result["cluster_assignments"],
    )

# Compute gradients
loss_value, grads = nnx.value_and_grad(loss_fn)(clustering, cell_embeddings)
print(f"Loss value: {loss_value:.4f}")
```

## Training with Gradient Descent

```python
import optax

optimizer = optax.adam(learning_rate=0.01)
params = nnx.state(clustering, nnx.Param)
opt_state = optimizer.init(params)

for step in range(50):
    loss_val, grads = nnx.value_and_grad(loss_fn)(clustering, cell_embeddings)

    params = nnx.state(clustering, nnx.Param)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    nnx.update(clustering, optax.apply_updates(params, updates))

    if step % 20 == 0:
        print(f"Step {step}: loss = {loss_val:.4f}")
```

## Temperature Effects

The temperature parameter controls assignment sharpness:

- **Low temperature** (0.1): Near-hard assignments, less smooth gradients
- **High temperature** (5.0): Very soft assignments, smoother gradients

## Next Steps

- See [Preprocessing Example](preprocessing.md) for read preprocessing
- Explore [Single-Cell Operators](../../user-guide/operators/singlecell.md) for more operators
