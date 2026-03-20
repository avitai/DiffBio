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
# # End-to-End Single-Cell Pipeline
#
# **Duration:** 30 minutes | **Level:** Advanced
# | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Chain five DiffBio operators into a full single-cell analysis pipeline
# 2. Trace how data dictionaries flow between operators (key mapping)
# 3. Verify end-to-end gradient flow through a multi-operator chain
# 4. JIT-compile a pipeline sub-chain for accelerated execution
#
# ## Prerequisites
#
# - DiffBio installed (see setup instructions)
# - Familiarity with single-cell RNA-seq concepts (count matrices, clustering)
# - Understanding of the `apply()` contract from DiffBio basics examples
#
# ```bash
# source ./activate.sh
# uv run python examples/pipelines/singlecell_pipeline.py
# ```
#
# ---

# %% [markdown]
# ## Pipeline Overview
#
# This example chains five operators in sequence, with each operator
# reading from and writing to a shared data dictionary:
#
# ```
# DifferentiableSimulator
#   -> produces "counts", "group_labels", "batch_labels"
#
# DifferentiableAmbientRemoval
#   -> reads "counts", "ambient_profile"
#   -> produces "decontaminated_counts"
#
# DifferentiableDiffusionImputer
#   -> reads "counts"
#   -> produces "imputed_counts"
#
# SoftKMeansClustering
#   -> reads "embeddings" (wired from imputed counts)
#   -> produces "cluster_assignments", "cluster_labels"
#
# DifferentiablePseudotime
#   -> reads "embeddings" (wired from imputed counts)
#   -> produces "pseudotime", "transition_matrix"
# ```

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
# ## 1. Operator Construction
#
# Each operator is configured independently. The pipeline chains them
# via data dictionary key passing -- the output dict of one operator
# provides the input keys for the next.

# %%
# Step 1 operator: Simulate realistic scRNA-seq counts
from diffbio.operators.singlecell import (
    DifferentiableSimulator,
    SimulationConfig,
)

N_CELLS = 30
N_GENES = 40
N_GROUPS = 3
N_CLUSTERS = 3

sim_config = SimulationConfig(
    n_cells=N_CELLS,
    n_genes=N_GENES,
    n_groups=N_GROUPS,
    stochastic=True,
    stream_name="sample",
)
simulator = DifferentiableSimulator(sim_config, rngs=nnx.Rngs(0, sample=1))
print(f"Simulator: {N_CELLS} cells x {N_GENES} genes, {N_GROUPS} groups")

# %%
# Step 2 operator: Remove ambient RNA contamination
from diffbio.operators.singlecell import (
    AmbientRemovalConfig,
    DifferentiableAmbientRemoval,
)

ambient_config = AmbientRemovalConfig(
    n_genes=N_GENES,
    latent_dim=8,
    hidden_dims=[32, 16],
    ambient_prior=0.01,
)
ambient_remover = DifferentiableAmbientRemoval(ambient_config, rngs=nnx.Rngs(2, sample=3))
print("Ambient remover: latent_dim=8, hidden=[32, 16]")

# %%
# Step 3 operator: Diffusion imputation for dropout recovery
from diffbio.operators.singlecell import (
    DifferentiableDiffusionImputer,
    DiffusionImputerConfig,
)

impute_config = DiffusionImputerConfig(
    n_neighbors=5,
    diffusion_t=2,
)
imputer = DifferentiableDiffusionImputer(impute_config, rngs=nnx.Rngs(4))
print("Imputer: n_neighbors=5, diffusion_t=2")

# %%
# Step 4 operator: Soft k-means clustering
from diffbio.operators.singlecell import (
    SoftClusteringConfig,
    SoftKMeansClustering,
)

cluster_config = SoftClusteringConfig(
    n_clusters=N_CLUSTERS,
    n_features=N_GENES,
    temperature=1.0,
)
clusterer = SoftKMeansClustering(cluster_config, rngs=nnx.Rngs(5))
print(f"Clusterer: {N_CLUSTERS} clusters, temperature=1.0")

# %%
# Step 5 operator: Pseudotime ordering via diffusion maps
from diffbio.operators.singlecell import (
    DifferentiablePseudotime,
    PseudotimeConfig,
)

pseudo_config = PseudotimeConfig(
    n_neighbors=5,
    n_diffusion_components=3,
    root_cell_index=0,
)
pseudotime_op = DifferentiablePseudotime(pseudo_config, rngs=nnx.Rngs(6))
print("Pseudotime: n_neighbors=5, n_components=3, root=cell 0")

# %% [markdown]
# ## 2. Step 1 -- Simulate Counts
#
# The simulator produces a count matrix with known group structure,
# batch effects, and expression-dependent dropout.

# %%
rp = simulator.generate_random_params(jax.random.key(42), {})
sim_result, sim_state, sim_meta = simulator.apply({}, {}, None, random_params=rp)

print("=== Step 1: Simulation ===")
print(f"  Output keys: {sorted(sim_result.keys())}")
print(f"  counts: shape={sim_result['counts'].shape}, dtype={sim_result['counts'].dtype}")
print(f"  group_labels: shape={sim_result['group_labels'].shape}")
print(f"  Mean expression: {sim_result['counts'].mean():.2f}")
print(f"  Fraction zeros: {float((sim_result['counts'] == 0).mean()):.3f}")

# %% [markdown]
# ## 3. Step 2 -- Ambient Removal
#
# The ambient remover expects "counts" and "ambient_profile".
# The ambient profile is typically estimated as the mean expression
# across empty droplets; here we approximate it from the data.

# %%
counts = sim_result["counts"]
ambient_profile = counts.mean(axis=0)

ambient_data = {
    "counts": counts,
    "ambient_profile": ambient_profile,
}
ambient_result, _, _ = ambient_remover.apply(ambient_data, {}, None)

print("=== Step 2: Ambient Removal ===")
print(f"  Output keys: {sorted(ambient_result.keys())}")
print(f"  decontaminated_counts: shape={ambient_result['decontaminated_counts'].shape}")
print(f"  contamination_fraction: shape={ambient_result['contamination_fraction'].shape}")
print(f"  Mean contamination: {ambient_result['contamination_fraction'].mean():.4f}")
print(f"  Decontaminated mean: {ambient_result['decontaminated_counts'].mean():.2f}")

# %% [markdown]
# ## 4. Step 3 -- Diffusion Imputation
#
# The imputer constructs a cell-cell affinity graph and diffuses
# expression values to recover dropout events. It reads "counts"
# and produces "imputed_counts".
#
# Here the original simulated counts (which retain signal variance)
# are fed directly to the imputer.

# %%
impute_data = {"counts": counts}
impute_result, _, _ = imputer.apply(impute_data, {}, None)

print("=== Step 3: Diffusion Imputation ===")
print(f"  Output keys: {sorted(impute_result.keys())}")
print(f"  imputed_counts: shape={impute_result['imputed_counts'].shape}")
print(f"  diffusion_operator: shape={impute_result['diffusion_operator'].shape}")
print(f"  Imputed mean: {impute_result['imputed_counts'].mean():.2f}")
print(f"  Input variance:   {float(counts.var()):.2f}")
print(f"  Imputed variance: {float(impute_result['imputed_counts'].var()):.2f}")

# %% [markdown]
# ## 5. Step 4 -- Soft K-Means Clustering
#
# The clusterer reads "embeddings". Wire imputed counts as the
# embedding space (in practice, a PCA or latent representation
# would be used; for demonstration, raw imputed counts suffice).

# %%
cluster_data = {"embeddings": impute_result["imputed_counts"]}
cluster_result, _, _ = clusterer.apply(cluster_data, {}, None)

print("=== Step 4: Clustering ===")
print(f"  Output keys: {sorted(cluster_result.keys())}")
print(f"  cluster_assignments: shape={cluster_result['cluster_assignments'].shape}")
print(f"  cluster_labels: shape={cluster_result['cluster_labels'].shape}")
print(f"  centroids: shape={cluster_result['centroids'].shape}")

# Show cluster distribution
labels = cluster_result["cluster_labels"]
for k in range(N_CLUSTERS):
    cell_count = int(jnp.sum(labels == k))
    print(f"    Cluster {k}: {cell_count} cells")

# %% [markdown]
# ## 6. Step 5 -- Pseudotime Ordering
#
# Pseudotime also reads "embeddings". The operator constructs a
# diffusion map from the cell-cell graph and measures distance
# from the root cell.

# %%
pseudo_data = {"embeddings": impute_result["imputed_counts"]}
pseudo_result, _, _ = pseudotime_op.apply(pseudo_data, {}, None)

print("=== Step 5: Pseudotime ===")
print(f"  Output keys: {sorted(pseudo_result.keys())}")
print(f"  pseudotime: shape={pseudo_result['pseudotime'].shape}")
print(f"  transition_matrix: shape={pseudo_result['transition_matrix'].shape}")
print(f"  diffusion_components: shape={pseudo_result['diffusion_components'].shape}")
print(
    f"  Pseudotime range: [{float(pseudo_result['pseudotime'].min()):.4f}, "
    f"{float(pseudo_result['pseudotime'].max()):.4f}]"
)
print(f"  Root cell pseudotime: {float(pseudo_result['pseudotime'][0]):.6f}")

# %%
# Figure 1: Pipeline overview -- input counts, cluster assignments, pseudotime
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: Input counts heatmap (cells x genes)
im0 = axes[0].imshow(sim_result["counts"][:, :20], cmap="viridis", aspect="auto")
axes[0].set_xlabel("Gene index")
axes[0].set_ylabel("Cell index")
axes[0].set_title("Input Counts (first 20 genes)")
plt.colorbar(im0, ax=axes[0], label="Count")

# Panel 2: Cluster assignments heatmap
im1 = axes[1].imshow(
    cluster_result["cluster_assignments"], cmap="coolwarm", aspect="auto"
)
axes[1].set_xlabel("Cluster")
axes[1].set_ylabel("Cell index")
axes[1].set_title("Cluster Assignments")
plt.colorbar(im1, ax=axes[1], label="Probability")

# Panel 3: Pseudotime values
pseudotime_vals = pseudo_result["pseudotime"]
sorted_idx = jnp.argsort(pseudotime_vals)
axes[2].plot(pseudotime_vals[sorted_idx], color="tab:purple", linewidth=1.5)
axes[2].set_xlabel("Cell (sorted)")
axes[2].set_ylabel("Pseudotime")
axes[2].set_title("Pseudotime Values")

plt.tight_layout()
plt.savefig("docs/assets/examples/pipelines/pipeline_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Deterministic Sub-Pipeline
#
# The deterministic operators (imputer, clusterer, pseudotime) can be
# wrapped into a single function for JIT compilation and gradient flow.
# The stochastic operators (simulator, ambient remover) use internal
# RNG state and run eagerly before the deterministic chain.


# %%
# Wrap the deterministic chain as a single function
def deterministic_pipeline(
    input_counts: jax.Array,
) -> dict[str, jax.Array]:
    """Chain imputation -> clustering -> pseudotime on input counts."""
    imp_out, _, _ = imputer.apply({"counts": input_counts}, {}, None)
    clust_out, _, _ = clusterer.apply({"embeddings": imp_out["imputed_counts"]}, {}, None)
    pt_out, _, _ = pseudotime_op.apply({"embeddings": imp_out["imputed_counts"]}, {}, None)
    return {
        "imputed_counts": imp_out["imputed_counts"],
        "cluster_assignments": clust_out["cluster_assignments"],
        "cluster_labels": clust_out["cluster_labels"],
        "pseudotime": pt_out["pseudotime"],
    }


pipeline_result = deterministic_pipeline(counts)
print("Deterministic pipeline output keys:")
for key_name, value in sorted(pipeline_result.items()):
    if hasattr(value, "shape"):
        print(f"  {key_name}: {value.shape}")

# %% [markdown]
# ## 8. Verify Differentiability
#
# Gradients flow through the deterministic chain: imputation,
# clustering, and pseudotime are all end-to-end differentiable.
# This enables gradient-based optimization of upstream parameters.


# %%
# Gradient through deterministic chain: imputer -> clusterer -> pseudotime
def deterministic_loss_fn(input_counts: jax.Array) -> jax.Array:
    """Scalar loss flowing through impute -> cluster -> pseudotime."""
    imp_out, _, _ = imputer.apply({"counts": input_counts}, {}, None)
    clust_out, _, _ = clusterer.apply({"embeddings": imp_out["imputed_counts"]}, {}, None)
    pt_out, _, _ = pseudotime_op.apply({"embeddings": imp_out["imputed_counts"]}, {}, None)
    return clust_out["cluster_assignments"].sum() + pt_out["pseudotime"].sum()


grad = jax.grad(deterministic_loss_fn)(counts)

print("=== End-to-End Gradient Verification (Impute -> Cluster -> Pseudotime) ===")
print(f"  Gradient shape: {grad.shape}")
print(f"  Gradient is non-zero: {bool(jnp.any(grad != 0))}")
print(f"  Gradient is finite:   {bool(jnp.all(jnp.isfinite(grad)))}")
print(f"  Gradient abs mean:    {float(jnp.abs(grad).mean()):.6f}")


# %%
# Verify the clusterer's learnable centroids receive gradients
def centroid_loss_fn(cluster_model: SoftKMeansClustering) -> jax.Array:
    """Loss on cluster assignments for centroid gradient check."""
    result, _, _ = cluster_model.apply({"embeddings": impute_result["imputed_counts"]}, {}, None)
    return result["cluster_assignments"].sum()


centroid_grad_fn = nnx.grad(centroid_loss_fn, argnums=nnx.DiffState(0, nnx.Param))
centroid_grads = centroid_grad_fn(clusterer)
c_grad = centroid_grads.centroids[...]
print(f"\n  Centroid gradient shape: {c_grad.shape}")
print(f"  Centroid gradient non-zero: {bool(jnp.any(c_grad != 0))}")
print(f"  Centroid gradient finite:   {bool(jnp.all(jnp.isfinite(c_grad)))}")

# %% [markdown]
# ## 9. JIT Compilation
#
# The deterministic sub-pipeline can be JIT-compiled as a single
# computation graph. JAX traces through imputation, clustering,
# and pseudotime, fusing them into an optimized kernel.

# %%
jit_pipeline = jax.jit(deterministic_pipeline)
jit_result = jit_pipeline(counts)

print("=== JIT Compilation Verification ===")
for key_name in sorted(jit_result.keys()):
    eager_val = pipeline_result[key_name]
    jit_val = jit_result[key_name]
    if hasattr(eager_val, "shape"):
        match = bool(jnp.allclose(eager_val, jit_val, atol=1e-5))
        print(f"  {key_name}: shape={jit_val.shape}, JIT matches eager: {match}")

# %% [markdown]
# ## 10. Experiments
#
# Explore how pipeline parameters affect the downstream results.

# %%
# Experiment 1: Vary diffusion time and observe imputation smoothness
print("=== Experiment: Diffusion Time ===")
for t_val in [1, 2, 4]:
    exp_config = DiffusionImputerConfig(n_neighbors=5, diffusion_t=t_val)
    exp_imputer = DifferentiableDiffusionImputer(exp_config, rngs=nnx.Rngs(4))
    exp_result, _, _ = exp_imputer.apply({"counts": counts}, {}, None)
    variance = float(exp_result["imputed_counts"].var())
    print(f"  t={t_val}: imputed variance={variance:.4f}")

# %%
# Experiment 2: Vary clustering temperature and observe assignment sharpness
print("\n=== Experiment: Clustering Temperature ===")
for temp in [0.1, 1.0, 10.0]:
    exp_config = SoftClusteringConfig(n_clusters=N_CLUSTERS, n_features=N_GENES, temperature=temp)
    exp_clusterer = SoftKMeansClustering(exp_config, rngs=nnx.Rngs(5))
    exp_data = {"embeddings": impute_result["imputed_counts"]}
    exp_result, _, _ = exp_clusterer.apply(exp_data, {}, None)
    assignments = exp_result["cluster_assignments"]
    mean_entropy = float(-jnp.sum(assignments * jnp.log(assignments + 1e-8), axis=-1).mean())
    print(f"  T={temp}: mean assignment entropy={mean_entropy:.4f}")

# %% [markdown]
# ## Summary
#
# This example demonstrated:
# - Chaining five DiffBio operators via data dictionary key passing
# - How each operator reads specific keys and produces new ones
# - End-to-end gradient flow through the deterministic sub-chain
# - JIT compilation of the impute -> cluster -> pseudotime pipeline
# - Parameter sensitivity experiments for diffusion time and temperature
#
# ## Next Steps
#
# - [scVI Benchmark](../ecosystem/scvi_benchmark.py) -- VAE normalization with
#   calibrax evaluation metrics
# - [Calibrax Metrics](../ecosystem/calibrax_metrics.py) -- training vs evaluation
#   metric comparison
