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
# # MAGIC-Style Diffusion Imputation
#
# **Duration:** 10-20 minutes | **Level:** Intermediate | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Understand MAGIC-style diffusion imputation for scRNA-seq dropout recovery
# 2. Configure and run DifferentiableDiffusionImputer on synthetic count data
# 3. Compare imputed values against known ground truth
# 4. Explore how diffusion time (`diffusion_t`) controls smoothing strength
# 5. Verify gradient flow and JIT compilation
#
# ## Prerequisites
#
# - DiffBio installed (`uv pip install -e .`)
# - Basic understanding of scRNA-seq count matrices and dropout events
#
# ```bash
# source ./activate.sh
# uv run python examples/singlecell/imputation.py
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
# ## 1. Generate Synthetic Data with Dropout
#
# Single-cell RNA sequencing suffers from dropout events: genes that are
# expressed but fail to be captured, appearing as zeros in the count matrix.
#
# To evaluate imputation, generate a ground truth expression matrix from
# Poisson-distributed counts, then introduce artificial dropout by zeroing
# entries with probability proportional to the inverse of expression level
# (low-expression genes drop out more).

# %%
key = jax.random.key(0)
n_cells = 60
n_genes = 50

# Ground truth: Poisson counts from a structured expression profile.
# Create 3 cell groups with distinct gene programs.
key, k1, k2, k3, k4 = jax.random.split(key, 5)

# Base expression rates per cell group (20 cells each)
rates_group1 = jnp.concatenate([jnp.full(25, 8.0), jnp.full(25, 2.0)])
rates_group2 = jnp.concatenate([jnp.full(25, 2.0), jnp.full(25, 8.0)])
rates_group3 = jnp.full(50, 5.0)

rates = jnp.stack([
    jnp.broadcast_to(rates_group1, (20, n_genes)),
    jnp.broadcast_to(rates_group2, (20, n_genes)),
    jnp.broadcast_to(rates_group3, (20, n_genes)),
]).reshape(n_cells, n_genes)

ground_truth = jax.random.poisson(k1, rates).astype(jnp.float32)

# Introduce dropout: zero-mask with probability inversely related to expression.
# P(dropout) = exp(-expression / scale), so low-expression genes drop out more.
dropout_scale = 3.0
dropout_probs = jnp.exp(-ground_truth / dropout_scale)
dropout_mask = jax.random.bernoulli(k2, dropout_probs)  # True = dropout
observed = jnp.where(dropout_mask, 0.0, ground_truth)

# Statistics
total_entries = n_cells * n_genes
n_dropouts = int(dropout_mask.sum())
n_true_zeros = int((ground_truth == 0).sum())
print(f"Data shape: ({n_cells}, {n_genes})")
print(f"Ground truth zero fraction: {n_true_zeros / total_entries:.2%}")
print(f"Dropout events introduced: {n_dropouts}")
print(f"Observed zero fraction: {float((observed == 0).sum()) / total_entries:.2%}")

# %% [markdown]
# ## 2. Configure the Diffusion Imputer
#
# `DifferentiableDiffusionImputer` implements the MAGIC algorithm:
#
# 1. Compute pairwise distances between cells
# 2. Build an alpha-decaying kernel for cell-cell affinity
# 3. Symmetrize the affinity graph
# 4. Eigendecompose to construct the Markov diffusion operator M^t
# 5. Impute: `imputed = M^t @ counts`
#
# Key parameters:
# - `n_neighbors`: controls local bandwidth for the affinity kernel
# - `diffusion_t`: number of diffusion steps (higher = more smoothing)
# - `decay`: exponent of the alpha-decaying kernel

# %%
from diffbio.operators.singlecell import (
    DifferentiableDiffusionImputer,
    DiffusionImputerConfig,
)

config = DiffusionImputerConfig(
    n_neighbors=5,
    diffusion_t=3,
    decay=1.0,
    metric="euclidean",
)

imputer = DifferentiableDiffusionImputer(config, rngs=nnx.Rngs(0))
print(f"Operator: {type(imputer).__name__}")
print(f"  n_neighbors={config.n_neighbors}, diffusion_t={config.diffusion_t}")

# %% [markdown]
# ## 3. Run Imputation

# %%
data = {"counts": observed}
result, state, metadata = imputer.apply(data, {}, None)

imputed = result["imputed_counts"]
print(f"Imputed shape: {imputed.shape}")
print(f"Diffusion operator shape: {result['diffusion_operator'].shape}")

# %%
n_show = 20
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

vmax = float(ground_truth[:n_show, :n_show].max())
axes[0].imshow(ground_truth[:n_show, :n_show], aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
axes[0].set_title("Original Counts")
axes[0].set_xlabel("Gene")
axes[0].set_ylabel("Cell")

axes[1].imshow(observed[:n_show, :n_show], aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
axes[1].set_title("Dropout Counts")
axes[1].set_xlabel("Gene")

im = axes[2].imshow(imputed[:n_show, :n_show], aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
axes[2].set_title("Imputed Counts")
axes[2].set_xlabel("Gene")

fig.colorbar(im, ax=axes, shrink=0.8, label="Expression")
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/imputation_heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Evaluate Imputation Quality
#
# Compare the imputed values to the ground truth. Focus on the dropout
# positions where the observed data was zeroed but the ground truth had
# nonzero expression.

# %%
# Identify dropout positions (observed=0, ground_truth>0)
is_dropout = (dropout_mask) & (ground_truth > 0)
n_dropout_nonzero = int(is_dropout.sum())

# Correlation between imputed and ground truth at dropout positions
imputed_at_dropout = imputed[is_dropout]
truth_at_dropout = ground_truth[is_dropout]

# Per-gene correlation across all cells
gene_correlations = []
for g in range(n_genes):
    corr = jnp.corrcoef(imputed[:, g], ground_truth[:, g])[0, 1]
    gene_correlations.append(float(corr))
gene_correlations = jnp.array(gene_correlations)

# Overall MSE
mse_observed = float(jnp.mean((observed - ground_truth) ** 2))
mse_imputed = float(jnp.mean((imputed - ground_truth) ** 2))

print(f"Dropout positions evaluated: {n_dropout_nonzero}")
print(f"MSE (observed vs truth): {mse_observed:.4f}")
print(f"MSE (imputed vs truth):  {mse_imputed:.4f}")
print(f"Mean per-gene correlation (imputed vs truth): {float(gene_correlations.mean()):.4f}")
print(f"Median per-gene correlation: {float(jnp.median(gene_correlations)):.4f}")

# %%
# Per-gene correlation scatter for gene 0
gene_idx = 0
corr_val = float(jnp.corrcoef(imputed[:, gene_idx], ground_truth[:, gene_idx])[0, 1])

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(imputed[:, gene_idx], ground_truth[:, gene_idx], s=15, alpha=0.7)
ax.set_xlabel("Imputed")
ax.set_ylabel("Original")
ax.set_title(f"Gene {gene_idx}: Imputed vs Original (r={corr_val:.3f})")

# Reference line
lo = min(float(imputed[:, gene_idx].min()), float(ground_truth[:, gene_idx].min()))
hi = max(float(imputed[:, gene_idx].max()), float(ground_truth[:, gene_idx].max()))
ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/imputation_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Effect of Diffusion Time
#
# The `diffusion_t` parameter controls how many diffusion steps are applied.
# Fewer steps preserve local structure but recover less signal. More steps
# provide stronger smoothing but can over-smooth and blur distinct cell
# populations.

# %%
diffusion_times = [1, 2, 3, 5, 8]
sweep_mses = []
sweep_corrs = []

print("diffusion_t -> MSE vs truth | Mean gene correlation")
print("-" * 55)
for t in diffusion_times:
    t_config = DiffusionImputerConfig(
        n_neighbors=5,
        diffusion_t=t,
        decay=1.0,
    )
    t_imputer = DifferentiableDiffusionImputer(t_config, rngs=nnx.Rngs(0))
    t_result, _, _ = t_imputer.apply(data, {}, None)
    t_imputed = t_result["imputed_counts"]

    mse = float(jnp.mean((t_imputed - ground_truth) ** 2))
    correlations = []
    for g in range(n_genes):
        corr = jnp.corrcoef(t_imputed[:, g], ground_truth[:, g])[0, 1]
        correlations.append(float(corr))
    mean_corr = float(jnp.mean(jnp.array(correlations)))

    sweep_mses.append(mse)
    sweep_corrs.append(mean_corr)
    print(f"  t={t}: MSE={mse:8.4f} | corr={mean_corr:.4f}")

# %%
fig, ax1 = plt.subplots(figsize=(7, 4))

color_mse = "tab:blue"
ax1.plot(diffusion_times, sweep_mses, "o-", color=color_mse, label="MSE")
ax1.set_xlabel("diffusion_t")
ax1.set_ylabel("MSE", color=color_mse)
ax1.tick_params(axis="y", labelcolor=color_mse)

ax2 = ax1.twinx()
color_corr = "tab:orange"
ax2.plot(diffusion_times, sweep_corrs, "s-", color=color_corr, label="Correlation")
ax2.set_ylabel("Mean Gene Correlation", color=color_corr)
ax2.tick_params(axis="y", labelcolor=color_corr)

fig.suptitle("Imputation Quality vs Diffusion Time")
fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/imputation_sweep.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Verify Differentiability
#
# The diffusion imputer is fully differentiable: gradients flow from the
# imputed counts back through the eigendecomposition and affinity
# construction into the input counts.

# %%
def loss_fn(input_data):
    """Scalar loss from imputed counts."""
    result, _, _ = imputer.apply(input_data, {}, None)
    return result["imputed_counts"].sum()


grad = jax.grad(loss_fn)(data)
grad_counts = grad["counts"]

print(f"Gradient shape: {grad_counts.shape}")
print(f"Gradient is non-zero: {bool(jnp.any(grad_counts != 0))}")
print(f"Gradient is finite: {bool(jnp.all(jnp.isfinite(grad_counts)))}")
print(f"Gradient mean magnitude: {float(jnp.abs(grad_counts).mean()):.6f}")

# %% [markdown]
# ## 7. JIT Compilation
#
# Verify that JIT-compiled imputation produces identical results.

# %%
jit_apply = jax.jit(lambda d: imputer.apply(d, {}, None))
result_jit, _, _ = jit_apply(data)

imputed_match = jnp.allclose(
    result["imputed_counts"],
    result_jit["imputed_counts"],
    atol=1e-4,
)
print(f"Imputed counts match (eager vs JIT): {bool(imputed_match)}")

# %% [markdown]
# ## 8. Alternative: Transformer Denoiser
#
# DiffBio also provides `DifferentiableTransformerDenoiser`, which treats
# genes as tokens and uses a transformer encoder to predict masked gene
# expression from unmasked context. This is a learned denoiser that requires
# training, in contrast to the graph-diffusion approach of MAGIC which is
# purely analytical.
#
# Key differences:
# - **DiffusionImputer**: No learnable parameters, uses cell-cell graph structure
# - **TransformerDenoiser**: Learnable transformer weights, uses gene-gene context
#
# The transformer approach can capture complex gene-gene dependencies but
# requires a training phase, making it better suited for large datasets where
# the learned representations generalize across cells.

# %% [markdown]
# ## Summary
#
# This example demonstrated:
# - MAGIC-style diffusion imputation for recovering dropout events in scRNA-seq
# - The effect of `diffusion_t` on smoothing strength and imputation quality
# - End-to-end differentiability through eigendecomposition and graph diffusion
# - JIT compatibility for accelerated execution
#
# ## Next Steps
#
# - [Trajectory inference](trajectory.py): Order cells along developmental pseudotime
# - [Clustering](clustering.py): Discover cell populations with soft k-means
