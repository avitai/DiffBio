# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Soft K-Means Clustering with Training
#
# **Duration:** 5-10 minutes | **Level:** Basic | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Generate structured synthetic data with known cluster membership
# 2. Train SoftKMeansClustering with an optax optimizer to learn cluster centroids
# 3. Evaluate clustering quality by comparing learned labels to ground truth
# 4. Verify gradient flow and JIT compilation
#
# ## Prerequisites
#
# - DiffBio installed (`uv pip install -e .`)
# - Basic understanding of k-means clustering
#
# ```bash
# source ./activate.sh
# uv run python examples/singlecell/clustering.py
# ```
#
# ---

# %% [markdown]
# ## Environment Setup

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx

print(f"JAX version: {jax.__version__}")
print(f"Backend: {jax.default_backend()}")

# %% [markdown]
# ## 1. Generate Synthetic 3-Cluster Data
#
# Create 3 well-separated Gaussian clusters in 2D, then embed them into a
# 20-dimensional space via a random projection. This gives the clustering
# operator a realistic task: find structure in a higher-dimensional space
# where the true clusters live on a 2D submanifold.

# %%
key = jax.random.key(0)
n_cells_per_cluster = 50
n_clusters_true = 3
n_features = 20

# 2D cluster centers, well-separated
centers_2d = jnp.array([
    [-3.0, -3.0],
    [3.0, -3.0],
    [0.0, 3.0],
])

# Generate 2D points around each center
keys = jax.random.split(key, n_clusters_true + 1)
clusters_2d = []
true_labels = []
for i in range(n_clusters_true):
    points = centers_2d[i] + jax.random.normal(keys[i], (n_cells_per_cluster, 2)) * 0.5
    clusters_2d.append(points)
    true_labels.append(jnp.full(n_cells_per_cluster, i))

points_2d = jnp.concatenate(clusters_2d, axis=0)
true_labels = jnp.concatenate(true_labels, axis=0)

# Random projection from 2D to 20D
projection_key = keys[-1]
projection = jax.random.normal(projection_key, (2, n_features)) / jnp.sqrt(2.0)
embeddings = points_2d @ projection

n_cells = embeddings.shape[0]
print(f"Data shape: {embeddings.shape} ({n_cells} cells, {n_features} features)")
print(f"True labels: {n_clusters_true} clusters, {n_cells_per_cluster} cells each")

# %% [markdown]
# ## 2. Configure the Operator
#
# Set `n_clusters=3` to match the true number of clusters. The temperature
# parameter controls the softness of assignments: lower temperatures produce
# sharper (more confident) cluster memberships.

# %%
from diffbio.operators.singlecell import (
    SoftClusteringConfig,
    SoftKMeansClustering,
)

config = SoftClusteringConfig(
    n_clusters=3,
    n_features=n_features,
    temperature=1.0,
    learnable_centroids=True,
)
operator = SoftKMeansClustering(config, rngs=nnx.Rngs(42))

# Initial cluster assignments before training
data = {"embeddings": embeddings}
result, _, _ = operator.apply(data, {}, None)
initial_labels = result["cluster_labels"]
print(f"Initial assignment distribution: {jnp.bincount(initial_labels, length=3).tolist()}")

# %% [markdown]
# ## 3. Training Loop
#
# Minimize the within-cluster dispersion: for each cell, compute the
# weighted sum of squared distances to centroids (weighted by soft
# assignments). This is the standard soft k-means objective.
#
# Because the operator is fully differentiable, `jax.grad` computes
# gradients of this loss with respect to the learnable centroids, and
# optax updates them.

# %%
# Use NNX functional API for JIT-compatible training: split the operator
# into a static graph definition and dynamic parameter/state partitions.
graphdef, params, other = nnx.split(operator, nnx.Param, ...)

optimizer = optax.adam(learning_rate=0.05)
opt_state = optimizer.init(params)

n_steps = 100


@jax.jit
def train_step(params, other, opt_state, input_data):
    """One gradient step minimizing within-cluster dispersion."""
    def loss_fn(params):
        model = nnx.merge(graphdef, params, other)
        result, _, _ = model.apply(input_data, {}, None)
        assignments = result["cluster_assignments"]  # (n_cells, n_clusters)
        centroids = result["centroids"]  # (n_clusters, n_features)

        # Squared distances from each cell to each centroid
        emb = input_data["embeddings"]
        diff = emb[:, None, :] - centroids[None, :, :]
        sq_dist = jnp.sum(diff ** 2, axis=-1)  # (n_cells, n_clusters)

        # Weighted dispersion: sum of (assignment * distance)
        return jnp.sum(assignments * sq_dist)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


# Training loop
losses = []
for step in range(n_steps):
    params, opt_state, loss = train_step(params, other, opt_state, data)
    losses.append(float(loss))

    if step % 20 == 0 or step == n_steps - 1:
        print(f"Step {step:3d}: loss = {float(loss):.4f}")

# Merge trained parameters back into the operator for evaluation
nnx.update(operator, params)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(losses)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/clustering_loss.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Evaluate Clustering Quality
#
# After training, compare the learned cluster labels against the true labels.
# Since cluster indices can be permuted, compute accuracy using the best
# label-to-cluster mapping.

# %%
result, _, _ = operator.apply(data, {}, None)
learned_labels = result["cluster_labels"]

# Compute best-match accuracy (handle label permutation)
from itertools import permutations

best_accuracy = 0.0
for perm in permutations(range(n_clusters_true)):
    remapped = jnp.zeros_like(learned_labels)
    for predicted, true_idx in enumerate(perm):
        remapped = jnp.where(learned_labels == predicted, true_idx, remapped)
    accuracy = float(jnp.mean(remapped == true_labels))
    best_accuracy = max(best_accuracy, accuracy)

print(f"Clustering accuracy (best permutation): {best_accuracy:.2%}")
print(f"Learned label distribution: {jnp.bincount(learned_labels, length=3).tolist()}")

# Verify assignments are confident (low entropy)
assignments = result["cluster_assignments"]
entropy = -jnp.sum(assignments * jnp.log(assignments + 1e-10), axis=-1)
print(f"Mean assignment entropy: {float(entropy.mean()):.4f} (lower = more confident)")

# %%
# Project back to 2D for visualization
emb_2d = embeddings @ jnp.linalg.pinv(projection)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

scatter0 = axes[0].scatter(
    emb_2d[:, 0], emb_2d[:, 1], c=true_labels, cmap="Set1", s=20, alpha=0.8
)
axes[0].set_title("True Labels")
axes[0].set_xlabel("Dim 1")
axes[0].set_ylabel("Dim 2")
fig.colorbar(scatter0, ax=axes[0], label="Cluster")

scatter1 = axes[1].scatter(
    emb_2d[:, 0], emb_2d[:, 1], c=learned_labels, cmap="Set1", s=20, alpha=0.8
)
axes[1].set_title("Learned Clusters vs True Labels")
axes[1].set_xlabel("Dim 1")
axes[1].set_ylabel("Dim 2")
fig.colorbar(scatter1, ax=axes[1], label="Cluster")

plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/clustering_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Verify Differentiability
#
# Confirm that gradients flow from the clustering loss back through the
# operator into the input embeddings.

# %%
def loss_fn(input_data):
    """Within-cluster dispersion loss from soft assignments and centroids."""
    result, _, _ = operator.apply(input_data, {}, None)
    assignments = result["cluster_assignments"]
    centroids = result["centroids"]
    emb = input_data["embeddings"]
    diff = emb[:, None, :] - centroids[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    return jnp.sum(assignments * sq_dist)


grad = jax.grad(loss_fn)(data)
grad_embeddings = grad["embeddings"]

print(f"Gradient shape: {grad_embeddings.shape}")
print(f"Gradient is non-zero: {bool(jnp.any(grad_embeddings != 0))}")
print(f"Gradient is finite: {bool(jnp.all(jnp.isfinite(grad_embeddings)))}")

# %% [markdown]
# ## 6. JIT Compilation
#
# Verify that the trained operator produces identical results under JIT.

# %%
jit_apply = jax.jit(lambda d: operator.apply(d, {}, None))
result_jit, _, _ = jit_apply(data)

assignments_match = jnp.allclose(
    result["cluster_assignments"],
    result_jit["cluster_assignments"],
    atol=1e-5,
)
print(f"Assignments match (eager vs JIT): {bool(assignments_match)}")

# %% [markdown]
# ## 7. Experiment: Temperature Effect
#
# Lower temperature makes assignments sharper (more one-hot), while higher
# temperature makes them softer (more uniform). This affects both the
# optimization landscape and the final clustering quality.

# %%
temperatures = [0.1, 0.5, 1.0, 5.0]
print("Temperature -> Mean max assignment probability:")
for temp in temperatures:
    temp_config = SoftClusteringConfig(
        n_clusters=3,
        n_features=n_features,
        temperature=temp,
    )
    temp_op = SoftKMeansClustering(temp_config, rngs=nnx.Rngs(42))
    temp_result, _, _ = temp_op.apply(data, {}, None)
    max_prob = temp_result["cluster_assignments"].max(axis=-1).mean()
    print(f"  T={temp:.1f}: {float(max_prob):.4f}")

# %% [markdown]
# ## Summary
#
# This example demonstrated:
# - Generating structured synthetic data with known cluster labels
# - Training SoftKMeansClustering with optax to learn cluster centroids
# - Evaluating clustering quality against ground truth
# - The effect of temperature on assignment sharpness
#
# ## Next Steps
#
# - [Imputation](imputation.py): Recover gene expression lost to dropout
# - [Trajectory inference](trajectory.py): Order cells along developmental pseudotime
