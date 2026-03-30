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
# # The DiffBio Operator Pattern
#
# **Duration:** 5-10 minutes | **Level:** Basic | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Understand the universal Config -> Construct -> Apply pattern used by all DiffBio operators
# 2. Know how to inspect operator outputs (shapes, keys, values)
# 3. Verify gradient flow through an operator (differentiability)
# 4. Confirm JIT compilation compatibility
# 5. Extract learnable parameters from an operator
#
# ## Prerequisites
#
# - DiffBio installed (`uv pip install -e .`)
# - Basic familiarity with JAX arrays
#
# ```bash
# source ./activate.sh
# uv run python examples/basics/operator_pattern.py
# ```
#
# ---

# %% [markdown]
# ## Environment Setup

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

print(f"JAX version: {jax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.device_count()}")

# %% [markdown]
# ## 1. Configuration
#
# Every DiffBio operator starts with a typed dataclass configuration. This
# separates hyperparameters from the operator logic and makes operators
# reproducible.
#
# SoftKMeansClustering performs differentiable soft k-means: instead of hard
# cluster assignments, cells receive soft assignment probabilities via
# softmax over negative squared distances to learnable centroids.

# %%
from diffbio.operators.singlecell import (
    SoftClusteringConfig,
    SoftKMeansClustering,
)

config = SoftClusteringConfig(
    n_clusters=5,
    n_features=20,
    temperature=1.0,
    learnable_centroids=True,
)
print(f"Config: n_clusters={config.n_clusters}, n_features={config.n_features}")

# Construct the operator with an RNG seed for reproducible parameter initialization
operator = SoftKMeansClustering(config, rngs=nnx.Rngs(42))
print(f"Operator: {type(operator).__name__}")

# %% [markdown]
# ## 2. Create Synthetic Data
#
# DiffBio operators expect dictionaries of JAX arrays. For clustering, the
# input key is `"embeddings"` with shape `(n_cells, n_features)`.

# %%
key = jax.random.key(0)
n_cells = 100
n_features = 20

embeddings = jax.random.normal(key, (n_cells, n_features))
data = {"embeddings": embeddings}

print(f"Input shape: {data['embeddings'].shape}")
print(f"Input dtype: {data['embeddings'].dtype}")

# %% [markdown]
# ## 3. Run the Operator
#
# All DiffBio operators follow the `apply()` contract:
#
# ```python
# result, state, metadata = operator.apply(data, state, metadata)
# ```
#
# - `data`: dictionary of JAX arrays (input and output)
# - `state`: operator state (empty dict `{}` for stateless operators)
# - `metadata`: optional metadata (use `None` for simple cases)

# %%
result, state, metadata = operator.apply(data, {}, None)

# Inspect all output keys and shapes
print("Output keys and shapes:")
for key_name, value in result.items():
    if hasattr(value, "shape"):
        print(f"  {key_name}: shape={value.shape}, dtype={value.dtype}")

# %% [markdown]
# The operator returns:
# - `"embeddings"`: the original input (passed through)
# - `"cluster_assignments"`: soft assignment probabilities (n_cells, n_clusters)
# - `"cluster_labels"`: hard labels from argmax (n_cells,)
# - `"centroids"`: learned centroid positions (n_clusters, n_features)

# %%
# Verify soft assignments sum to 1 per cell (valid probability distributions)
assignment_sums = result["cluster_assignments"].sum(axis=-1)
print(
    f"Assignment row sums (should be ~1.0):"
    f" min={assignment_sums.min():.6f}, max={assignment_sums.max():.6f}"
)

# Check cluster label distribution
labels = result["cluster_labels"]
unique_labels, counts = jnp.unique(labels, return_counts=True, size=config.n_clusters)
print(f"Cluster label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")

# %%
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(result["cluster_assignments"], aspect="auto", cmap="viridis")
ax.set_xlabel("Cluster")
ax.set_ylabel("Cell")
ax.set_title("Soft Cluster Assignments")
fig.colorbar(im, ax=ax, label="Assignment Probability")
plt.tight_layout()
plt.savefig(
    "docs/assets/examples/basic/operator_pattern_assignments.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## 4. Verify Differentiability
#
# DiffBio's core value proposition: bioinformatics operators are end-to-end
# differentiable. Gradients flow from a scalar loss back through the operator
# into the input data.

# %%
def loss_fn(input_data):
    """Compute a scalar loss from soft cluster assignments."""
    result, _, _ = operator.apply(input_data, {}, None)
    return result["cluster_assignments"].sum()


grad = jax.grad(loss_fn)(data)

# Verify gradient properties
grad_embeddings = grad["embeddings"]
print(f"Gradient shape: {grad_embeddings.shape}")
print(f"Gradient is non-zero: {bool(jnp.any(grad_embeddings != 0))}")
print(f"Gradient is finite: {bool(jnp.all(jnp.isfinite(grad_embeddings)))}")
print(f"Gradient mean magnitude: {jnp.abs(grad_embeddings).mean():.6f}")

# %% [markdown]
# ## 5. JIT Compilation
#
# All DiffBio operators work with `jax.jit` for accelerated execution.
# The JIT-compiled version produces identical results.

# %%
jit_apply = jax.jit(lambda d: operator.apply(d, {}, None))
result_jit, _, _ = jit_apply(data)

# Verify JIT output matches eager output
assignments_match = jnp.allclose(
    result["cluster_assignments"],
    result_jit["cluster_assignments"],
    atol=1e-5,
)
labels_match = jnp.array_equal(result["cluster_labels"], result_jit["cluster_labels"])
print(f"Assignments match (eager vs JIT): {bool(assignments_match)}")
print(f"Labels match (eager vs JIT): {bool(labels_match)}")

# %% [markdown]
# ## 6. Parameter Extraction
#
# DiffBio operators built on Flax NNX store learnable parameters as
# `nnx.Param` variables. Use `nnx.state()` to inspect the full parameter
# tree.

# %%
param_state = nnx.state(operator, nnx.Param)

# Display parameter names and shapes
print("Learnable parameters:")
for path_tuple, param in param_state.flat_state():
    path_str = ".".join(path_tuple)
    arr = param[...]
    print(f"  {path_str}: shape={arr.shape}, dtype={arr.dtype}")

# Direct access to centroids
centroids = operator.centroids[...]
print(f"\nCentroid matrix shape: {centroids.shape}")
print(f"Centroid value range: [{centroids.min():.4f}, {centroids.max():.4f}]")

# %% [markdown]
# ## Summary
#
# This example demonstrated the universal DiffBio operator pattern:
#
# 1. **Configure**: Create a typed config dataclass with operator parameters
# 2. **Construct**: Instantiate the operator with config and RNG seed
# 3. **Apply**: Call `operator.apply(data, state, metadata)` with dictionary input
# 4. **Differentiate**: Use `jax.grad` to verify gradient flow
# 5. **Accelerate**: Use `jax.jit` for compiled execution
# 6. **Inspect**: Use `nnx.state()` to extract learnable parameters
#
# Every DiffBio operator -- from alignment to variant calling to single-cell
# analysis -- follows this same pattern.
#
# ## Next Steps
#
# - [Clustering with training](../singlecell/clustering.py): Train soft k-means on structured data
# - [Imputation](../singlecell/imputation.py): MAGIC-style diffusion imputation
# - [Trajectory inference](../singlecell/trajectory.py): Pseudotime and fate probabilities
