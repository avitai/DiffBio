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
# # [Example Title]
#
# **Duration:** [X minutes] | **Level:** [Basic/Intermediate/Advanced]
# | **Device:** [CPU-compatible/GPU-optional/GPU-required]
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Understand [bioinformatics concept]
# 2. Be able to [use DiffBio operator]
# 3. Know how to [verify differentiability]
#
# ## Prerequisites
#
# - DiffBio installed (see setup instructions)
# - Basic understanding of [domain concept]
#
# ```bash
# source ./activate.sh
# uv run python examples/path/to/example.py
# ```
#
# ---

# %%
# Environment setup
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
# [Explain what the operator does and what parameters control.]

# %%
# Configure the operator
from diffbio.operators.singlecell import SoftKMeansClustering, SoftClusteringConfig

config = SoftClusteringConfig(
    n_clusters=5,
    n_features=20,
    temperature=1.0,
)
operator = SoftKMeansClustering(config, rngs=nnx.Rngs(42))

print(f"Operator created: {type(operator).__name__}")

# %% [markdown]
# ## 2. Create Data
#
# [Explain the data format and what it represents biologically.]

# %%
# Generate synthetic data
key = jax.random.key(0)
n_cells, n_features = 100, 20
embeddings = jax.random.normal(key, (n_cells, n_features))

data = {"embeddings": embeddings}
print(f"Input shape: {data['embeddings'].shape}")
print(f"Input dtype: {data['embeddings'].dtype}")

# %% [markdown]
# ## 3. Run the Operator
#
# [Explain what the operator computes and what outputs to expect.]

# %%
# Apply the operator
result, state, metadata = operator.apply(data, {}, None)

# Inspect outputs -- print shapes and key values for every result tensor
for key_name in sorted(result.keys()):
    value = result[key_name]
    if hasattr(value, "shape"):
        print(f"  {key_name}: shape={value.shape}, dtype={value.dtype}")

# %%
# --- REQUIRED: Visualize the operator output ---
#
# Every example must produce at least one figure. Place plt.savefig()
# before plt.show() to write the figure to docs/assets/examples/.
# The path mirrors the example location:
#   examples/singlecell/clustering.py
#   -> docs/assets/examples/singlecell/clustering_assignments.png

ASSET_DIR = "docs/assets/examples/domain"  # <-- change to match your example

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(
    result["cluster_assignments"][:30],
    aspect="auto",
    cmap="viridis",
)
ax.set_xlabel("Cluster")
ax.set_ylabel("Cell")
ax.set_title("Soft Cluster Assignments (first 30 cells)")
plt.colorbar(im, ax=ax, label="Assignment Probability")
plt.tight_layout()
# Uncomment after creating the asset directory:
# plt.savefig(f"{ASSET_DIR}/operator_output.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Verify Differentiability
#
# DiffBio operators are end-to-end differentiable. This is the core
# value proposition: gradients flow through bioinformatics computations.

# %%
# Gradient flow test


def loss_fn(input_data):
    """Compute a scalar loss from the operator output."""
    res, _, _ = operator.apply(input_data, {}, None)
    return res["cluster_assignments"].sum()


grad = jax.grad(loss_fn)(data)
print(f"Gradient shape: {grad['embeddings'].shape}")
print(f"Gradient is non-zero: {bool(jnp.any(grad['embeddings'] != 0))}")
print(f"Gradient is finite: {bool(jnp.all(jnp.isfinite(grad['embeddings'])))}")

# %% [markdown]
# ## 5. JIT Compilation
#
# All DiffBio operators are compatible with JAX's JIT compiler for
# accelerated execution on GPU/TPU.

# %%
# JIT-compiled forward pass
jit_apply = jax.jit(lambda d: operator.apply(d, {}, None))
result_jit, _, _ = jit_apply(data)
print(
    "JIT output matches: "
    f"{bool(jnp.allclose(result['cluster_assignments'], result_jit['cluster_assignments']))}"
)

# %% [markdown]
# ## 6. Experiments
#
# [Sweep a key parameter and visualize how it affects the output.
# Every experiment section should produce a figure.]

# %%
# Experiment: vary temperature and observe assignment sharpness
temperatures = [0.1, 0.5, 1.0, 5.0]
mean_max_probs = []

for temp in temperatures:
    cfg = SoftClusteringConfig(n_clusters=5, n_features=20, temperature=temp)
    op = SoftKMeansClustering(cfg, rngs=nnx.Rngs(42))
    res, _, _ = op.apply(data, {}, None)
    mean_max_probs.append(float(res["cluster_assignments"].max(axis=1).mean()))
    print(f"  T={temp}: mean max probability = {mean_max_probs[-1]:.4f}")

# %%
# --- REQUIRED: Visualize experiment results ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(temperatures, mean_max_probs, "o-", linewidth=2, markersize=7, color="tab:blue")
ax.set_xlabel("Temperature")
ax.set_ylabel("Mean Max Assignment Probability")
ax.set_title("Assignment Sharpness vs Temperature")
plt.tight_layout()
# Uncomment after creating the asset directory:
# plt.savefig(f"{ASSET_DIR}/operator_sweep.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# In this example, you learned:
# - How to configure and run [operator name]
# - That gradients flow through the computation (differentiability)
# - That JIT compilation works for accelerated execution
# - How [parameter] affects [behavior] (experiment)
#
# ## Next Steps
#
# - [Link to related example]
# - [Link to API reference]
# - [Link to more advanced pipeline example]
