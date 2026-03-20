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
# # Gene Regulatory Network Inference
#
# **Duration:** 25 minutes | **Level:** Advanced
# | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Understand GRN inference using GATv2 attention on a TF-gene bipartite graph
# 2. Generate synthetic expression data from a known regulatory network
# 3. Apply DifferentiableGRN to extract an inferred GRN matrix
# 4. Compare the inferred network with ground truth regulatory edges
# 5. Verify differentiability and JIT compatibility
#
# ## Prerequisites
#
# - DiffBio installed (see setup instructions)
# - Understanding of gene regulatory networks, transcription factors, and
#   regulatory target genes
# - Familiarity with graph attention networks (conceptual)
#
# ```bash
# source ./activate.sh
# uv run python examples/singlecell/grn_inference.py
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
# ## 1. Define a Ground Truth Regulatory Network
#
# Create a synthetic GRN with 5 transcription factors (TFs) and 20 target
# genes. Each TF regulates a subset of genes with known regulatory strengths.
# This serves as the ground truth for evaluating the inferred network.

# %%
# Network parameters
n_tfs = 5
n_genes = 20
n_cells = 100

# TF indices: first 5 genes are TFs
tf_indices = jnp.arange(n_tfs)

# Ground truth regulatory matrix: (n_tfs, n_genes)
# Each TF regulates a specific set of target genes
key = jax.random.key(42)
k1, k2, k3, k4 = jax.random.split(key, 4)

# Sparse ground truth: each TF regulates ~4 genes
grn_truth = jnp.zeros((n_tfs, n_genes))

# TF 0 -> genes 5, 6, 7, 8 (positive regulation)
grn_truth = grn_truth.at[0, 5:9].set(1.0)
# TF 1 -> genes 9, 10, 11 (positive regulation)
grn_truth = grn_truth.at[1, 9:12].set(1.0)
# TF 2 -> genes 12, 13, 14, 15 (positive regulation)
grn_truth = grn_truth.at[2, 12:16].set(1.0)
# TF 3 -> genes 16, 17 (positive regulation)
grn_truth = grn_truth.at[3, 16:18].set(1.0)
# TF 4 -> genes 18, 19 (positive regulation)
grn_truth = grn_truth.at[4, 18:20].set(1.0)

# TFs also have autoregulatory loops
grn_truth = grn_truth.at[0, 0].set(0.5)
grn_truth = grn_truth.at[1, 1].set(0.5)
grn_truth = grn_truth.at[2, 2].set(0.5)
grn_truth = grn_truth.at[3, 3].set(0.5)
grn_truth = grn_truth.at[4, 4].set(0.5)

n_true_edges = int((grn_truth > 0).sum())
print(f"Ground truth GRN: {n_tfs} TFs x {n_genes} genes")
print(f"True regulatory edges: {n_true_edges}")
print(f"Network density: {n_true_edges / (n_tfs * n_genes):.2%}")

# %% [markdown]
# ## 2. Simulate Expression from the Network
#
# Generate synthetic single-cell expression data that reflects the
# regulatory structure. TF expression drives target gene expression
# through the ground truth regulatory weights.

# %%
# Simulate TF expression: each TF has a base level plus cell-specific noise
tf_base = jnp.array([10.0, 8.0, 12.0, 6.0, 9.0])
tf_expression = (
    tf_base[None, :] + jax.random.normal(k1, (n_cells, n_tfs)) * 2.0
)
tf_expression = jnp.maximum(tf_expression, 0.1)  # non-negative

# Generate target gene expression as a function of TF activity:
# gene_expr = sum(TF_expr * regulatory_weight) + noise
# For each gene, sum over all TFs weighted by the GRN
regulated_signal = tf_expression @ grn_truth  # (n_cells, n_genes)

# Add baseline expression and noise
baseline = jax.random.poisson(k2, jnp.ones((n_cells, n_genes)) * 2.0)
noise = jax.random.normal(k3, (n_cells, n_genes)) * 0.5

gene_expression = jnp.maximum(regulated_signal + baseline.astype(jnp.float32) + noise, 0.0)

# Assemble full count matrix: TFs (genes 0-4) + targets (genes 5-19)
counts = jnp.concatenate([tf_expression, gene_expression[:, n_tfs:]], axis=1)
# Overwrite TF columns in gene_expression with actual TF expression
counts = counts.at[:, :n_tfs].set(tf_expression)

print(f"Expression matrix shape: {counts.shape}")
print(f"Mean expression per TF: {counts[:, :n_tfs].mean(axis=0)}")
print(f"Mean expression (targets): {counts[:, n_tfs:].mean():.2f}")

# Verify that strongly regulated genes have higher expression than weakly regulated
strong_mask = grn_truth.sum(axis=0) >= 1.0  # genes with direct TF regulation
weak_mask = grn_truth.sum(axis=0) < 1.0     # genes with weak or no TF regulation
print(f"Mean expr (strongly regulated): {counts[:, strong_mask].mean():.2f}")
print(f"Mean expr (weakly regulated): {counts[:, weak_mask].mean():.2f}")

# %% [markdown]
# ## 3. Apply DifferentiableGRN
#
# The operator builds a dense TF-gene bipartite graph, computes edge features
# from expression, applies GATv2 attention to learn regulatory strengths,
# then extracts the GRN matrix with soft L1 sparsity.

# %%
from diffbio.operators.singlecell import (
    DifferentiableGRN,
    GRNInferenceConfig,
)

config_grn = GRNInferenceConfig(
    n_tfs=n_tfs,
    n_genes=n_genes,
    hidden_dim=16,
    num_heads=4,
    sparsity_temperature=0.1,
    sparsity_lambda=0.01,
)
grn_op = DifferentiableGRN(config_grn, rngs=nnx.Rngs(0))
print(f"GRN operator created: {type(grn_op).__name__}")
print(f"  hidden_dim={config_grn.hidden_dim}, "
      f"num_heads={config_grn.num_heads}, "
      f"sparsity_temp={config_grn.sparsity_temperature}")

# %%
# Run GRN inference
data_grn = {
    "counts": counts,
    "tf_indices": tf_indices,
}
result_grn, state_grn, meta_grn = grn_op.apply(data_grn, {}, None)

grn_matrix = result_grn["grn_matrix"]
tf_activity = result_grn["tf_activity"]

print(f"Inferred GRN matrix shape: {grn_matrix.shape}")
print(f"TF activity shape: {tf_activity.shape}")
print(f"GRN value range: [{float(grn_matrix.min()):.4f}, {float(grn_matrix.max()):.4f}]")
print(f"GRN sparsity (fraction near zero, |w| < 0.01): "
      f"{float((jnp.abs(grn_matrix) < 0.01).mean()):.4f}")

# %%
# Figure 1: Ground truth vs inferred GRN
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(grn_truth, cmap="Blues", aspect="auto")
ax1.set_xlabel("Gene index")
ax1.set_ylabel("TF index")
ax1.set_title("Ground Truth GRN")
plt.colorbar(im1, ax=ax1, label="Regulatory weight")

im2 = ax2.imshow(jnp.abs(grn_matrix), cmap="Reds", aspect="auto")
ax2.set_xlabel("Gene index")
ax2.set_ylabel("TF index")
ax2.set_title("Inferred GRN (|weights|)")
plt.colorbar(im2, ax=ax2, label="|Inferred weight|")

plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/grn_heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Compare Inferred Network with Ground Truth
#
# Evaluate the inferred GRN against the known regulatory edges.
# The comparison uses the absolute GRN values as scores, with
# higher scores indicating stronger predicted regulation.

# %%
# Binarize truth and predictions at a threshold
grn_abs = jnp.abs(grn_matrix)
truth_binary = (grn_truth > 0).astype(jnp.float32)

# Compute overlap statistics at different thresholds
print("=== GRN Comparison with Ground Truth ===\n")
print(f"{'Threshold':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} "
      f"{'Recall':>10}")
print("-" * 50)

thresholds = [0.0, 0.01, 0.05, 0.1, 0.5]
recall_values = []

for threshold in thresholds:
    pred_binary = (grn_abs > threshold).astype(jnp.float32)
    tp = float((pred_binary * truth_binary).sum())
    fp = float((pred_binary * (1 - truth_binary)).sum())
    fn = float(((1 - pred_binary) * truth_binary).sum())
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    recall_values.append(recall)
    print(f"{threshold:>10.2f} {int(tp):>5} {int(fp):>5} {int(fn):>5} "
          f"{precision:>10.4f} {recall:>10.4f}")

# %%
# Figure 2: Recall at different thresholds
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    [str(t) for t in thresholds],
    recall_values,
    color="tab:green",
    edgecolor="k",
    linewidth=0.5,
)
ax.set_xlabel("Threshold")
ax.set_ylabel("Recall")
ax.set_title("Recall @ Threshold")
ax.set_ylim(0, 1.05)
for bar, val in zip(bars, recall_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/grn_threshold_recall.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Per-TF analysis: which TF's targets are best recovered
print("\n=== Per-TF Regulatory Strength ===\n")
print(f"{'TF':>4} {'True Targets':>14} {'Mean |inferred|':>16} "
      f"{'Max |inferred|':>16}")
print("-" * 54)

for tf_idx in range(n_tfs):
    true_targets = int(grn_truth[tf_idx].sum())
    target_mask = grn_truth[tf_idx] > 0
    mean_strength = float(jnp.where(target_mask, grn_abs[tf_idx], 0.0).sum() / true_targets)
    max_strength = float(grn_abs[tf_idx].max())
    print(f"{tf_idx:>4} {true_targets:>14} {mean_strength:>16.4f} "
          f"{max_strength:>16.4f}")

# %%
# Compare inferred TF activity with ground truth TF activity
true_activity = counts[:, :n_tfs] @ grn_truth  # (n_cells, n_genes)
# Summarize per-TF activity correlation
print("\n=== TF Activity Correlation (per TF) ===\n")

for tf_idx in range(n_tfs):
    inferred = tf_activity[:, tf_idx]
    true = true_activity[:, tf_idx]  # Aggregate TF activity across target genes
    # Pearson correlation
    inferred_centered = inferred - inferred.mean()
    true_centered = true - true.mean()
    corr = float(
        (inferred_centered * true_centered).sum()
        / (jnp.sqrt((inferred_centered ** 2).sum() * (true_centered ** 2).sum()) + 1e-10)
    )
    print(f"  TF {tf_idx}: correlation = {corr:.4f}")

# %% [markdown]
# ## 5. Verify Differentiability
#
# The GRN inference operator is fully differentiable. Gradients flow through
# the GATv2 attention, bipartite graph construction, and soft sparsity gating.
# This enables gradient-based optimization of regulatory network models.

# %%
print("=== Gradient Flow Verification ===\n")


def loss_fn_grn(input_counts):
    """Scalar loss from inferred GRN matrix (counts only)."""
    d = {"counts": input_counts, "tf_indices": tf_indices}
    res, _, _ = grn_op.apply(d, {}, None)
    return res["grn_matrix"].sum()


grad_grn = jax.grad(loss_fn_grn)(counts)
print("GRN Inference:")
print(f"  Gradient shape: {grad_grn.shape}")
print(f"  Non-zero: {bool(jnp.any(grad_grn != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(grad_grn)))}")

# %%
# Also verify gradient flows through TF activity output


def loss_fn_activity(input_counts):
    """Scalar loss from TF activity (counts only)."""
    d = {"counts": input_counts, "tf_indices": tf_indices}
    res, _, _ = grn_op.apply(d, {}, None)
    return res["tf_activity"].sum()


grad_act = jax.grad(loss_fn_activity)(counts)
print("TF Activity:")
print(f"  Gradient shape: {grad_act.shape}")
print(f"  Non-zero: {bool(jnp.any(grad_act != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(grad_act)))}")

# %% [markdown]
# ## 6. JIT Compilation
#
# The GRN inference operator is JIT-compatible. The bipartite graph
# construction and GATv2 forward pass are fully traceable.

# %%
print("=== JIT Compilation ===\n")

jit_grn = jax.jit(lambda d: grn_op.apply(d, {}, None))
result_jit_grn, _, _ = jit_grn(data_grn)
match_grn = jnp.allclose(
    result_grn["grn_matrix"],
    result_jit_grn["grn_matrix"],
    atol=1e-5,
)
print(f"GRN JIT matches eager: {bool(match_grn)}")

match_act = jnp.allclose(
    result_grn["tf_activity"],
    result_jit_grn["tf_activity"],
    atol=1e-5,
)
print(f"TF activity JIT matches eager: {bool(match_act)}")

# %% [markdown]
# ## 7. Experiments
#
# ### Vary the sparsity temperature
#
# The sparsity temperature controls the soft L1 gating: lower values
# produce sharper thresholding (more zeros), while higher values preserve
# more regulatory weights.

# %%
print("=== Experiment: Sparsity Temperature Effect ===\n")

for temp in [0.01, 0.05, 0.1, 0.5, 1.0]:
    cfg = GRNInferenceConfig(
        n_tfs=n_tfs,
        n_genes=n_genes,
        hidden_dim=16,
        num_heads=4,
        sparsity_temperature=temp,
    )
    op = DifferentiableGRN(cfg, rngs=nnx.Rngs(0))
    res, _, _ = op.apply(data_grn, {}, None)
    grn = res["grn_matrix"]
    sparsity = float((jnp.abs(grn) < 0.01).mean())
    value_range = float(jnp.abs(grn).max())
    print(f"  temp={temp:.2f} -> sparsity: {sparsity:.4f}, "
          f"max |weight|: {value_range:.4f}")

# %% [markdown]
# ### Vary the number of attention heads
#
# More attention heads allow the GATv2 layer to capture different
# types of regulatory relationships simultaneously.

# %%
print("\n=== Experiment: Number of Attention Heads ===\n")

for n_heads in [1, 2, 4, 8]:
    # hidden_dim must be divisible by num_heads
    hdim = max(16, n_heads * 4)
    cfg = GRNInferenceConfig(
        n_tfs=n_tfs,
        n_genes=n_genes,
        hidden_dim=hdim,
        num_heads=n_heads,
        sparsity_temperature=0.1,
    )
    op = DifferentiableGRN(cfg, rngs=nnx.Rngs(0))
    res, _, _ = op.apply(data_grn, {}, None)
    grn = res["grn_matrix"]
    grn_a = jnp.abs(grn)

    # Compute recall of true edges at a loose threshold
    pred_bin = (grn_a > 0.01).astype(jnp.float32)
    tp = float((pred_bin * truth_binary).sum())
    recall = tp / (float(truth_binary.sum()) + 1e-10)
    print(f"  heads={n_heads}, hidden={hdim} -> recall@0.01: {recall:.4f}, "
          f"max |w|: {float(grn_a.max()):.4f}")

# %% [markdown]
# ### Vary the number of cells
#
# More cells provide better estimation of mean expression patterns,
# which affects the quality of edge features for the GATv2 layer.

# %%
print("\n=== Experiment: Number of Cells ===\n")

for n_c in [20, 50, 100, 200]:
    # Generate data for varying cell counts
    k_sub = jax.random.fold_in(k4, n_c)
    k_sub1, k_sub2, k_sub3 = jax.random.split(k_sub, 3)

    tf_expr = tf_base[None, :] + jax.random.normal(k_sub1, (n_c, n_tfs)) * 2.0
    tf_expr = jnp.maximum(tf_expr, 0.1)
    regulated = tf_expr @ grn_truth
    bl = jax.random.poisson(k_sub2, jnp.ones((n_c, n_genes)) * 2.0).astype(jnp.float32)
    ns = jax.random.normal(k_sub3, (n_c, n_genes)) * 0.5
    ge = jnp.maximum(regulated + bl + ns, 0.0)
    cts = jnp.concatenate([tf_expr, ge[:, n_tfs:]], axis=1)
    cts = cts.at[:, :n_tfs].set(tf_expr)

    d = {"counts": cts, "tf_indices": tf_indices}
    res, _, _ = grn_op.apply(d, {}, None)
    grn = res["grn_matrix"]
    grn_a = jnp.abs(grn)
    pred_bin = (grn_a > 0.01).astype(jnp.float32)
    tp = float((pred_bin * truth_binary).sum())
    recall = tp / (float(truth_binary.sum()) + 1e-10)
    print(f"  n_cells={n_c:>3} -> recall@0.01: {recall:.4f}, "
          f"max |w|: {float(grn_a.max()):.4f}")

# %% [markdown]
# ## Summary
#
# Differentiable GRN inference was demonstrated using GATv2 attention on a
# TF-gene bipartite graph:
#
# - A known regulatory network with 5 TFs and 20 genes was defined
# - Expression data was simulated from the ground truth network
# - DifferentiableGRN extracted regulatory weights via graph attention
# - The inferred GRN was compared with ground truth at multiple thresholds
# - Sparsity gating produces biologically realistic sparse networks
#
# The operator is fully differentiable and JIT-compatible, enabling
# gradient-based optimization of GRN inference models within end-to-end
# pipelines.
#
# ## Next Steps
#
# - Train the GRN operator by backpropagating a supervised regulatory loss
# - Combine GRN inference with trajectory inference for dynamic GRN analysis
# - Explore regulon discovery by thresholding the inferred GRN matrix
# - Compare with GENIE3/SCENIC on real scRNA-seq data
