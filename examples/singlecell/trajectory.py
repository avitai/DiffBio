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
# # Trajectory Inference: Pseudotime and Fate Probability
#
# **Duration:** 10-20 minutes | **Level:** Intermediate | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Construct synthetic branching trajectory data with known ordering
# 2. Compute differentiable pseudotime via diffusion maps
# 3. Estimate fate probabilities for branching lineages
# 4. Chain pseudotime with SwitchDE to detect gene regulation events
# 5. Verify gradient flow and JIT compilation for all operators
#
# ## Prerequisites
#
# - DiffBio installed (`uv pip install -e .`)
# - Understanding of pseudotime ordering in single-cell biology
#
# ```bash
# source ./activate.sh
# uv run python examples/singlecell/trajectory.py
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

# %% [markdown]
# ## 1. Generate Branching Trajectory Data
#
# Create a Y-shaped trajectory: a stem from the root that branches into two
# terminal fates. Cells along the stem and branches are ordered by a known
# ground-truth pseudotime, providing a benchmark for the inferred ordering.
#
# The trajectory lives in 2D, then gets embedded into 20D via random
# projection (mimicking a PCA embedding of real scRNA-seq data).

# %%
key = jax.random.key(0)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)

n_stem = 30
n_branch = 20
n_features = 20

# Stem: linear from (0, 0) to (3, 0)
t_stem = jnp.linspace(0.0, 1.0, n_stem)
stem_x = t_stem * 3.0
stem_y = jnp.zeros(n_stem)

# Branch A: from (3, 0) curving up to (6, 3)
t_branch_a = jnp.linspace(0.0, 1.0, n_branch)
branch_a_x = 3.0 + t_branch_a * 3.0
branch_a_y = t_branch_a * 3.0

# Branch B: from (3, 0) curving down to (6, -3)
branch_b_x = 3.0 + t_branch_a * 3.0
branch_b_y = -t_branch_a * 3.0

# Combine 2D positions
positions_2d = jnp.stack([
    jnp.concatenate([stem_x, branch_a_x, branch_b_x]),
    jnp.concatenate([stem_y, branch_a_y, branch_b_y]),
], axis=-1)

# Add noise
noise = jax.random.normal(k1, positions_2d.shape) * 0.15
positions_2d = positions_2d + noise

# Ground truth pseudotime: stem [0, 0.5), branches [0.5, 1.0]
true_pseudotime = jnp.concatenate([
    t_stem * 0.5,
    0.5 + t_branch_a * 0.5,
    0.5 + t_branch_a * 0.5,
])

# Ground truth fate labels: stem=neither, branch_a=0, branch_b=1
# For evaluation, stem cells closer to branch A get label 0, others get 1
true_fate = jnp.concatenate([
    jnp.zeros(n_stem, dtype=jnp.int32),  # stem -> assigned post-hoc
    jnp.zeros(n_branch, dtype=jnp.int32),  # branch A
    jnp.ones(n_branch, dtype=jnp.int32),  # branch B
])

n_cells = positions_2d.shape[0]

# Project to higher dimensions
projection = jax.random.normal(k2, (2, n_features)) / jnp.sqrt(2.0)
embeddings = positions_2d @ projection

print(f"Total cells: {n_cells} (stem={n_stem}, branch_a={n_branch}, branch_b={n_branch})")
print(f"Embedding shape: {embeddings.shape}")

# %% [markdown]
# ## 2. Compute Pseudotime
#
# `DifferentiablePseudotime` builds a k-NN affinity graph, constructs a
# Markov transition matrix, eigendecomposes it for diffusion map components,
# and computes pseudotime as the L2 distance from the root cell in diffusion
# space.

# %%
from diffbio.operators.singlecell import (
    DifferentiablePseudotime,
    PseudotimeConfig,
)

pt_config = PseudotimeConfig(
    n_neighbors=10,
    n_diffusion_components=5,
    root_cell_index=0,  # First stem cell is the root
    metric="euclidean",
)
pseudotime_op = DifferentiablePseudotime(pt_config, rngs=nnx.Rngs(0))

data = {"embeddings": embeddings}
pt_result, pt_state, pt_metadata = pseudotime_op.apply(data, {}, None)

pseudotime = pt_result["pseudotime"]
print(f"Pseudotime shape: {pseudotime.shape}")
print(f"Pseudotime range: [{float(pseudotime.min()):.4f}, {float(pseudotime.max()):.4f}]")
print(f"Diffusion components shape: {pt_result['diffusion_components'].shape}")
print(f"Transition matrix shape: {pt_result['transition_matrix'].shape}")

# %% [markdown]
# ## 3. Evaluate Pseudotime Quality
#
# A good pseudotime should correlate strongly with the known ground-truth
# ordering. Check Spearman rank correlation.

# %%
# Rank correlation between inferred and true pseudotime
inferred_ranks = jnp.argsort(jnp.argsort(pseudotime)).astype(jnp.float32)
true_ranks = jnp.argsort(jnp.argsort(true_pseudotime)).astype(jnp.float32)
rank_corr = jnp.corrcoef(inferred_ranks, true_ranks)[0, 1]
print(f"Rank correlation (inferred vs true pseudotime): {float(rank_corr):.4f}")

# Root cell should have pseudotime = 0
print(f"Root cell pseudotime: {float(pseudotime[0]):.6f} (should be 0)")

# Stem cells should have lower pseudotime than branch cells
stem_mean = float(pseudotime[:n_stem].mean())
branch_mean = float(pseudotime[n_stem:].mean())
print(f"Mean pseudotime - stem: {stem_mean:.4f}, branches: {branch_mean:.4f}")

# %%
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(
    positions_2d[:, 0], positions_2d[:, 1], c=pseudotime, cmap="plasma", s=20, alpha=0.8
)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("Cells Colored by Pseudotime")
fig.colorbar(sc, ax=ax, label="Pseudotime")
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/trajectory_pseudotime.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Compute Fate Probabilities
#
# `DifferentiableFateProbability` uses the Markov transition matrix to compute
# absorption probabilities: the probability that each transient cell eventually
# reaches each terminal state. This requires specifying which cells are
# terminal (absorbing states).

# %%
from diffbio.operators.singlecell import (
    DifferentiableFateProbability,
    FateProbabilityConfig,
)

# Terminal states: last cell in branch A and last cell in branch B
terminal_a_idx = n_stem + n_branch - 1  # Last cell in branch A
terminal_b_idx = n_stem + 2 * n_branch - 1  # Last cell in branch B
terminal_states = jnp.array([terminal_a_idx, terminal_b_idx])

fate_config = FateProbabilityConfig(n_macrostates=2)
fate_op = DifferentiableFateProbability(fate_config, rngs=nnx.Rngs(0))

# Fate probability needs the transition matrix from pseudotime
fate_data = {
    "transition_matrix": pt_result["transition_matrix"],
    "terminal_states": terminal_states,
}
fate_result, _, _ = fate_op.apply(fate_data, {}, None)

fate_probs = fate_result["fate_probabilities"]
macrostates = fate_result["macrostates"]

print(f"Fate probabilities shape: {fate_probs.shape}")
print(f"Macrostates shape: {macrostates.shape}")

# %% [markdown]
# ## 5. Evaluate Fate Probabilities
#
# Verify that fate probabilities sum to approximately 1 and that cells on
# each branch are assigned to the correct terminal fate.

# %%
# Row sums should be ~1
fate_sums = fate_probs.sum(axis=-1)
print(f"Fate probability row sums: min={float(fate_sums.min()):.4f}, max={float(fate_sums.max()):.4f}")

# Branch A cells should favor terminal A (column 0)
branch_a_fate = fate_probs[n_stem:n_stem + n_branch, 0]
branch_b_fate = fate_probs[n_stem + n_branch:, 1]
print(f"Branch A cells -> fate A probability: mean={float(branch_a_fate.mean()):.4f}")
print(f"Branch B cells -> fate B probability: mean={float(branch_b_fate.mean()):.4f}")

# Terminal cells should have probability 1 for their own fate
print(f"Terminal A fate probs: {fate_probs[terminal_a_idx].tolist()}")
print(f"Terminal B fate probs: {fate_probs[terminal_b_idx].tolist()}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: scatter colored by fate A probability
sc0 = axes[0].scatter(
    positions_2d[:, 0], positions_2d[:, 1],
    c=fate_probs[:, 0], cmap="RdYlBu_r", s=20, alpha=0.8, vmin=0, vmax=1,
)
axes[0].set_title("Fate A Probability")
axes[0].set_xlabel("PC 1")
axes[0].set_ylabel("PC 2")
fig.colorbar(sc0, ax=axes[0], label="P(Fate A)")

# Right: stacked bar per cell group (stem, branch A, branch B)
group_labels = ["Stem", "Branch A", "Branch B"]
group_slices = [slice(0, n_stem), slice(n_stem, n_stem + n_branch), slice(n_stem + n_branch, None)]
mean_fate_a = [float(fate_probs[s, 0].mean()) for s in group_slices]
mean_fate_b = [float(fate_probs[s, 1].mean()) for s in group_slices]

x_pos = range(len(group_labels))
axes[1].bar(x_pos, mean_fate_a, label="Fate A", color="tab:blue")
axes[1].bar(x_pos, mean_fate_b, bottom=mean_fate_a, label="Fate B", color="tab:orange")
axes[1].set_xticks(list(x_pos))
axes[1].set_xticklabels(group_labels)
axes[1].set_ylabel("Mean Fate Probability")
axes[1].set_title("Fate Probabilities by Cell Group")
axes[1].legend()

plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/trajectory_fate.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Chain with SwitchDE
#
# `DifferentiableSwitchDE` models gene expression as a sigmoidal function of
# pseudotime. By chaining pseudotime output into SwitchDE, the full pipeline
# from embeddings to switch-gene detection becomes differentiable.

# %%
from diffbio.operators.singlecell import (
    DifferentiableSwitchDE,
    SwitchDEConfig,
)

# Create synthetic gene expression that switches at different pseudotime points
key_genes = jax.random.key(99)
n_switch_genes = 10

# Normalize pseudotime to [0, 1] for SwitchDE
pt_normalized = pseudotime / (pseudotime.max() + 1e-8)

# Generate synthetic counts: sigmoidal expression patterns
true_switch_times = jax.random.uniform(key_genes, (n_switch_genes,), minval=0.2, maxval=0.8)
counts_synthetic = jnp.zeros((n_cells, n_switch_genes))
for g_idx in range(n_switch_genes):
    sigmoid_val = jax.nn.sigmoid((pt_normalized - true_switch_times[g_idx]) * 10.0)
    counts_synthetic = counts_synthetic.at[:, g_idx].set(sigmoid_val * 5.0)

# Add noise
counts_synthetic = counts_synthetic + jax.random.normal(
    jax.random.key(100), counts_synthetic.shape
) * 0.3

switch_config = SwitchDEConfig(n_genes=n_switch_genes, temperature=0.5)
switch_op = DifferentiableSwitchDE(switch_config, rngs=nnx.Rngs(42))

switch_data = {
    "counts": counts_synthetic,
    "pseudotime": pt_normalized,
}
switch_result, _, _ = switch_op.apply(switch_data, {}, None)

print(f"Switch times (learned init): {switch_result['switch_times'][:5].tolist()}")
print(f"Switch scores: {switch_result['switch_scores'][:5].tolist()}")
print(f"Predicted expression shape: {switch_result['predicted_expression'].shape}")

# %% [markdown]
# ## 7. Verify Differentiability
#
# Verify gradient flow for both operators independently, confirming that
# DiffBio's trajectory operators are end-to-end differentiable.

# %%
# Pseudotime gradient flow
def pt_loss_fn(input_data):
    """Scalar loss from pseudotime values."""
    result, _, _ = pseudotime_op.apply(input_data, {}, None)
    return result["pseudotime"].sum()


pt_grad = jax.grad(pt_loss_fn)({"embeddings": embeddings})
pt_grad_emb = pt_grad["embeddings"]
print("Pseudotime operator gradients:")
print(f"  Shape: {pt_grad_emb.shape}")
print(f"  Non-zero: {bool(jnp.any(pt_grad_emb != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(pt_grad_emb)))}")

# Fate probability gradient flow (through transition matrix).
# Only differentiate with respect to the float-valued transition_matrix,
# since terminal_states is integer-valued and not differentiable.
def fate_loss_fn(transition_matrix):
    """Scalar loss from fate probabilities."""
    input_data = {"transition_matrix": transition_matrix, "terminal_states": terminal_states}
    result, _, _ = fate_op.apply(input_data, {}, None)
    return result["fate_probabilities"].sum()


fate_grad_tm = jax.grad(fate_loss_fn)(pt_result["transition_matrix"])
print("\nFate probability operator gradients:")
print(f"  Shape: {fate_grad_tm.shape}")
print(f"  Non-zero: {bool(jnp.any(fate_grad_tm != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(fate_grad_tm)))}")

# %% [markdown]
# ## 8. JIT Compilation
#
# Verify that both operators produce correct results under JIT compilation.

# %%
# Pseudotime JIT
jit_pt = jax.jit(lambda d: pseudotime_op.apply(d, {}, None))
pt_result_jit, _, _ = jit_pt({"embeddings": embeddings})
pt_match = jnp.allclose(pseudotime, pt_result_jit["pseudotime"], atol=1e-4)
print(f"Pseudotime matches (eager vs JIT): {bool(pt_match)}")

# Fate probability JIT
jit_fate = jax.jit(lambda d: fate_op.apply(d, {}, None))
fate_result_jit, _, _ = jit_fate(fate_data)
fate_match = jnp.allclose(
    fate_probs,
    fate_result_jit["fate_probabilities"],
    atol=1e-4,
)
print(f"Fate probabilities match (eager vs JIT): {bool(fate_match)}")

# %% [markdown]
# ## 9. Experiment: Number of Diffusion Components
#
# The `n_diffusion_components` parameter controls how many eigenvectors of
# the transition matrix are used to compute pseudotime. More components
# capture finer structure but are noisier; fewer components give smoother
# ordering.

# %%
component_counts = [2, 5, 10, 15]
sweep_rank_corrs = []
print("n_components -> Rank correlation with true pseudotime")
print("-" * 55)
for n_comp in component_counts:
    comp_config = PseudotimeConfig(
        n_neighbors=10,
        n_diffusion_components=n_comp,
        root_cell_index=0,
    )
    comp_op = DifferentiablePseudotime(comp_config, rngs=nnx.Rngs(0))
    comp_result, _, _ = comp_op.apply({"embeddings": embeddings}, {}, None)
    comp_pt = comp_result["pseudotime"]

    inferred_r = jnp.argsort(jnp.argsort(comp_pt)).astype(jnp.float32)
    corr = float(jnp.corrcoef(inferred_r, true_ranks)[0, 1])
    sweep_rank_corrs.append(corr)
    print(f"  n_components={n_comp:2d}: rank_corr={corr:.4f}")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(component_counts, sweep_rank_corrs, "o-", color="tab:green")
ax.set_xlabel("n_diffusion_components")
ax.set_ylabel("Rank Correlation")
ax.set_title("Pseudotime Quality vs Diffusion Components")
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/trajectory_sweep.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# This example demonstrated:
# - Pseudotime inference via diffusion maps on a synthetic branching trajectory
# - Fate probability estimation using absorption probabilities on the Markov chain
# - Chaining pseudotime output into SwitchDE for gene regulation analysis
# - End-to-end differentiability of the full trajectory inference pipeline
#
# ## Next Steps
#
# - [Clustering](clustering.py): Discover cell populations with soft k-means
# - [Imputation](imputation.py): Recover gene expression lost to dropout
