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
# # Batch Correction with Harmony, MMD, and WGAN
#
# **Duration:** 20 minutes | **Level:** Intermediate
# | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Understand batch effects in multi-sample single-cell experiments
# 2. Apply three differentiable batch correction strategies:
#    DifferentiableHarmony, DifferentiableMMDBatchCorrection,
#    and DifferentiableWGANBatchCorrection
# 3. Evaluate correction quality with a simple batch mixing metric
# 4. Verify differentiability and JIT compatibility for all three
#
# ## Prerequisites
#
# - DiffBio installed (see setup instructions)
# - Basic understanding of batch effects in scRNA-seq data
#
# ```bash
# source ./activate.sh
# uv run python examples/singlecell/batch_correction.py
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
# ## 1. Create Synthetic Batched Data
#
# Generate data with 2 batches, each containing 2 cell types. The batch effect
# is a shift in the mean expression, simulating technical variation between
# experimental runs. Both batches contain the same biological types.

# %%
# Parameters
n_cells_per_batch = 50
n_batches = 2
n_types = 2
n_features = 30  # embedding or gene dimension
n_cells = n_cells_per_batch * n_batches

key = jax.random.key(42)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)

# Cell type centers (shared biology)
type_centers = jnp.array([
    jnp.concatenate([jnp.ones(15) * 5.0, jnp.zeros(15)]),  # Type 0: high first half
    jnp.concatenate([jnp.zeros(15), jnp.ones(15) * 5.0]),  # Type 1: high second half
])

# Batch effect: constant shift per batch
batch_shift = jnp.array([
    jnp.zeros(n_features),                    # Batch 0: no shift (reference)
    jnp.ones(n_features) * 3.0,               # Batch 1: +3.0 shift
])

# Generate cell profiles
embeddings_list = []
batch_labels_list = []
type_labels_list = []

for batch_idx in range(n_batches):
    for type_idx in range(n_types):
        n = n_cells_per_batch // n_types
        subkey = jax.random.fold_in(k1, batch_idx * n_types + type_idx)
        noise = jax.random.normal(subkey, (n, n_features)) * 0.5
        cells = type_centers[type_idx] + batch_shift[batch_idx] + noise
        embeddings_list.append(cells)
        batch_labels_list.append(jnp.full(n, batch_idx))
        type_labels_list.append(jnp.full(n, type_idx))

embeddings = jnp.concatenate(embeddings_list, axis=0)
batch_labels = jnp.concatenate(batch_labels_list, axis=0).astype(jnp.int32)
type_labels = jnp.concatenate(type_labels_list, axis=0).astype(jnp.int32)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Batch distribution: {jnp.bincount(batch_labels, length=n_batches)}")
print(f"Type distribution: {jnp.bincount(type_labels, length=n_types)}")


# %% [markdown]
# ## 2. Batch Mixing Metric
#
# Define a simple batch mixing metric: for each cell, look at its nearest
# neighbors and compute the entropy of batch labels. Higher entropy means
# better mixing (batches are interleaved rather than segregated).

# %%
def compute_batch_mixing_score(
    emb: jax.Array,
    b_labels: jax.Array,
    n_neighbors: int = 10,
    n_batch: int = 2,
) -> float:
    """Compute batch mixing entropy from k-nearest neighbors.

    For each cell, the batch label distribution among its k nearest
    neighbors is computed, and the mean Shannon entropy across all cells
    is returned. Higher values indicate better batch mixing.

    Args:
        emb: Embedding matrix (n_cells, n_features).
        b_labels: Batch labels (n_cells,).
        n_neighbors: Number of neighbors to consider.
        n_batch: Number of batches.

    Returns:
        Mean entropy (higher = better mixing).
    """
    # Pairwise squared distances
    sq = jnp.sum(emb ** 2, axis=1, keepdims=True)
    dists = sq + sq.T - 2.0 * emb @ emb.T
    dists = dists + jnp.eye(emb.shape[0]) * 1e10  # mask self

    # Find k nearest neighbors via argsort
    nn_idx = jnp.argsort(dists, axis=1)[:, :n_neighbors]
    nn_batches = b_labels[nn_idx]  # (n_cells, k)

    # Batch proportion per cell neighborhood
    batch_onehot = jax.nn.one_hot(nn_batches, n_batch)  # (n_cells, k, n_batch)
    batch_props = batch_onehot.mean(axis=1)  # (n_cells, n_batch)

    # Shannon entropy -- clamp proportions away from 0 for stable log
    batch_props_safe = jnp.clip(batch_props, 1e-8, 1.0)
    entropy = -jnp.sum(batch_props_safe * jnp.log(batch_props_safe), axis=1)
    return float(entropy.mean())


# Baseline mixing score (before correction)
baseline_score = compute_batch_mixing_score(embeddings, batch_labels)
print(f"Baseline batch mixing score: {baseline_score:.4f}")
print(f"Perfect mixing (2 batches): {float(-jnp.log(jnp.array(0.5))):.4f}")

# %% [markdown]
# ## 3. DifferentiableHarmony
#
# Harmony uses soft clustering with batch-aware centroid updates. Iterative
# correction moves cell embeddings toward corrected centroids, reducing
# batch effects while preserving biological variation.

# %%
from diffbio.operators.singlecell import (
    BatchCorrectionConfig,
    DifferentiableHarmony,
)

config_harmony = BatchCorrectionConfig(
    n_clusters=20,
    n_features=n_features,
    n_batches=n_batches,
    n_iterations=10,
    theta=2.0,
    sigma=0.1,
    temperature=1.0,
)
harmony = DifferentiableHarmony(config_harmony, rngs=nnx.Rngs(0))
print(f"Harmony operator created: {type(harmony).__name__}")

# %%
# Run Harmony correction
data_harmony = {
    "embeddings": embeddings,
    "batch_labels": batch_labels,
}
result_harmony, state_h, meta_h = harmony.apply(data_harmony, {}, None)

corrected_harmony = result_harmony["corrected_embeddings"]
assignments_h = result_harmony["cluster_assignments"]

print(f"Corrected shape: {corrected_harmony.shape}")
print(f"Cluster assignments shape: {assignments_h.shape}")

score_harmony = compute_batch_mixing_score(corrected_harmony, batch_labels)
print(f"Harmony batch mixing score: {score_harmony:.4f} (baseline: {baseline_score:.4f})")

# %% [markdown]
# ## 4. DifferentiableMMDBatchCorrection
#
# The MMD-based corrector uses an autoencoder with Maximum Mean Discrepancy
# regularization. The MMD loss penalizes distributional differences between
# batches in latent space, pushing the encoder toward batch-invariant
# representations.

# %%
from diffbio.operators.singlecell import (
    DifferentiableMMDBatchCorrection,
    MMDBatchCorrectionConfig,
)

config_mmd = MMDBatchCorrectionConfig(
    n_genes=n_features,
    hidden_dim=64,
    latent_dim=16,
    kernel_bandwidth=1.0,
)
mmd_corrector = DifferentiableMMDBatchCorrection(config_mmd, rngs=nnx.Rngs(1))
print(f"MMD corrector created: {type(mmd_corrector).__name__}")

# %%
# Run MMD correction (uses "expression" key instead of "embeddings")
data_mmd = {
    "expression": embeddings,
    "batch_labels": batch_labels,
}
result_mmd, state_m, meta_m = mmd_corrector.apply(data_mmd, {}, None)

corrected_mmd = result_mmd["corrected_expression"]
latent_mmd = result_mmd["latent"]
mmd_loss = result_mmd["mmd_loss"]
recon_loss = result_mmd["reconstruction_loss"]

print(f"Corrected shape: {corrected_mmd.shape}")
print(f"Latent shape: {latent_mmd.shape}")
print(f"MMD loss: {float(mmd_loss):.4f}")
print(f"Reconstruction loss: {float(recon_loss):.4f}")

score_mmd = compute_batch_mixing_score(corrected_mmd, batch_labels)
print(f"MMD batch mixing score: {score_mmd:.4f} (baseline: {baseline_score:.4f})")

# %% [markdown]
# ## 5. DifferentiableWGANBatchCorrection
#
# The WGAN-based corrector uses an adversarial autoencoder with a Wasserstein
# discriminator. Gradient reversal ensures the encoder learns to fool the
# discriminator, producing batch-invariant latent representations.

# %%
from diffbio.operators.singlecell import (
    DifferentiableWGANBatchCorrection,
    WGANBatchCorrectionConfig,
)

config_wgan = WGANBatchCorrectionConfig(
    n_genes=n_features,
    hidden_dim=64,
    latent_dim=16,
    discriminator_hidden_dim=32,
)
wgan_corrector = DifferentiableWGANBatchCorrection(config_wgan, rngs=nnx.Rngs(2))
print(f"WGAN corrector created: {type(wgan_corrector).__name__}")

# %%
# Run WGAN correction
data_wgan = {
    "expression": embeddings,
    "batch_labels": batch_labels,
}
result_wgan, state_w, meta_w = wgan_corrector.apply(data_wgan, {}, None)

corrected_wgan = result_wgan["corrected_expression"]
latent_wgan = result_wgan["latent"]
gen_loss = result_wgan["generator_loss"]
disc_loss = result_wgan["discriminator_loss"]

print(f"Corrected shape: {corrected_wgan.shape}")
print(f"Generator loss: {float(gen_loss):.4f}")
print(f"Discriminator loss: {float(disc_loss):.4f}")

score_wgan = compute_batch_mixing_score(corrected_wgan, batch_labels)
print(f"WGAN batch mixing score: {score_wgan:.4f} (baseline: {baseline_score:.4f})")

# %% [markdown]
# ## 6. Compare All Methods
#
# Compare the batch mixing scores from all three correction methods.
# Since these are untrained models, the scores reflect initial behavior.
# With training, all methods should converge to higher mixing scores.

# %%
print("=== Batch Correction Comparison ===\n")
print(f"{'Method':<12} {'Mixing Score':>14} {'vs Baseline':>14}")
print("-" * 42)
print(f"{'Uncorrected':<12} {baseline_score:>14.4f} {'---':>14}")
print(f"{'Harmony':<12} {score_harmony:>14.4f} "
      f"{score_harmony - baseline_score:>+14.4f}")
print(f"{'MMD':<12} {score_mmd:>14.4f} "
      f"{score_mmd - baseline_score:>+14.4f}")
print(f"{'WGAN':<12} {score_wgan:>14.4f} "
      f"{score_wgan - baseline_score:>+14.4f}")

# Mean shift between batches (lower = better correction)
print(f"\n{'Method':<12} {'Batch 0 Mean':>14} {'Batch 1 Mean':>14} {'Shift':>14}")
print("-" * 56)
for name, emb in [
    ("Uncorrected", embeddings),
    ("Harmony", corrected_harmony),
    ("MMD", corrected_mmd),
    ("WGAN", corrected_wgan),
]:
    b0_mask = batch_labels == 0
    b1_mask = batch_labels == 1
    m0 = float(jnp.where(b0_mask[:, None], emb, 0.0).sum() / b0_mask.sum() / n_features)
    m1 = float(jnp.where(b1_mask[:, None], emb, 0.0).sum() / b1_mask.sum() / n_features)
    print(f"{name:<12} {m0:>14.4f} {m1:>14.4f} {abs(m1 - m0):>14.4f}")

# %%
# Figure 1: PCA scatter plots before and after correction


def pca_2d(data: jax.Array) -> jax.Array:
    """Project data to 2D using SVD-based PCA."""
    centered = data - data.mean(axis=0, keepdims=True)
    # Clamp to avoid extreme values from untrained autoencoders
    centered = jnp.clip(centered, -100.0, 100.0)
    _, _, vt = jnp.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    return jnp.clip(proj, -50.0, 50.0)


panels = [
    ("Uncorrected", embeddings),
    ("Harmony", corrected_harmony),
    ("MMD", corrected_mmd),
    ("WGAN", corrected_wgan),
]

fig, axes = plt.subplots(2, 2, figsize=(9, 8))
batch_colors = ["#4C72B0", "#DD8452"]

for ax, (title, emb) in zip(axes.flat, panels):
    pc = pca_2d(emb)
    for b in range(n_batches):
        mask = batch_labels == b
        ax.scatter(
            pc[mask, 0], pc[mask, 1],
            s=15, alpha=0.7, color=batch_colors[b], label=f"Batch {b}",
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.legend(fontsize=8, loc="best")

fig.suptitle("PCA Projections Colored by Batch", fontsize=13)
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/batch_pca_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Verify Differentiability
#
# All three batch correction methods are fully differentiable, enabling
# end-to-end gradient-based optimization in single-cell pipelines.

# %%
print("=== Gradient Flow Verification ===\n")

# Harmony gradient: differentiate only w.r.t. float embeddings


def loss_fn_harmony(input_emb):
    """Scalar loss from Harmony correction (embeddings only)."""
    d = {"embeddings": input_emb, "batch_labels": batch_labels}
    res, _, _ = harmony.apply(d, {}, None)
    return res["corrected_embeddings"].sum()


grad_harmony = jax.grad(loss_fn_harmony)(embeddings)
print("Harmony:")
print(f"  Gradient shape: {grad_harmony.shape}")
print(f"  Non-zero: {bool(jnp.any(grad_harmony != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(grad_harmony)))}")

# %%
# MMD gradient: differentiate only w.r.t. float expression


def loss_fn_mmd(input_expr):
    """Scalar loss from MMD correction (expression only)."""
    d = {"expression": input_expr, "batch_labels": batch_labels}
    res, _, _ = mmd_corrector.apply(d, {}, None)
    return res["corrected_expression"].sum()


grad_mmd = jax.grad(loss_fn_mmd)(embeddings)
print("MMD:")
print(f"  Gradient shape: {grad_mmd.shape}")
print(f"  Non-zero: {bool(jnp.any(grad_mmd != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(grad_mmd)))}")

# %%
# WGAN gradient: differentiate only w.r.t. float expression


def loss_fn_wgan(input_expr):
    """Scalar loss from WGAN correction (expression only)."""
    d = {"expression": input_expr, "batch_labels": batch_labels}
    res, _, _ = wgan_corrector.apply(d, {}, None)
    return res["corrected_expression"].sum()


grad_wgan = jax.grad(loss_fn_wgan)(embeddings)
print("WGAN:")
print(f"  Gradient shape: {grad_wgan.shape}")
print(f"  Non-zero: {bool(jnp.any(grad_wgan != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(grad_wgan)))}")

# %% [markdown]
# ## 8. JIT Compilation
#
# All three operators support JIT compilation for efficient execution.

# %%
print("=== JIT Compilation ===\n")

# Harmony JIT
jit_harmony = jax.jit(lambda d: harmony.apply(d, {}, None))
result_jit_h, _, _ = jit_harmony(data_harmony)
match_h = jnp.allclose(
    result_harmony["corrected_embeddings"],
    result_jit_h["corrected_embeddings"],
    atol=1e-5,
)
print(f"Harmony JIT matches eager: {bool(match_h)}")

# %%
# MMD JIT
jit_mmd = jax.jit(lambda d: mmd_corrector.apply(d, {}, None))
result_jit_m, _, _ = jit_mmd(data_mmd)
match_m = jnp.allclose(
    result_mmd["corrected_expression"],
    result_jit_m["corrected_expression"],
    atol=1e-5,
)
print(f"MMD JIT matches eager: {bool(match_m)}")

# %%
# WGAN JIT
jit_wgan = jax.jit(lambda d: wgan_corrector.apply(d, {}, None))
result_jit_w, _, _ = jit_wgan(data_wgan)
match_w = jnp.allclose(
    result_wgan["corrected_expression"],
    result_jit_w["corrected_expression"],
    atol=1e-5,
)
print(f"WGAN JIT matches eager: {bool(match_w)}")

# %% [markdown]
# ## 9. Experiments
#
# ### Vary the number of Harmony iterations
#
# More iterations allow Harmony to converge further toward
# batch-corrected embeddings.

# %%
print("=== Experiment: Harmony Iteration Count ===\n")

for n_iter in [1, 5, 10, 20]:
    cfg = BatchCorrectionConfig(
        n_clusters=20,
        n_features=n_features,
        n_batches=n_batches,
        n_iterations=n_iter,
        theta=2.0,
        sigma=0.1,
        temperature=1.0,
    )
    h = DifferentiableHarmony(cfg, rngs=nnx.Rngs(0))
    res, _, _ = h.apply(data_harmony, {}, None)
    score = compute_batch_mixing_score(res["corrected_embeddings"], batch_labels)
    print(f"  n_iterations={n_iter:>2} -> mixing score: {score:.4f}")

# %% [markdown]
# ### Vary MMD kernel bandwidth
#
# The kernel bandwidth controls the sensitivity of the MMD loss
# to distributional differences between batches.

# %%
print("\n=== Experiment: MMD Kernel Bandwidth ===\n")

for bw in [0.1, 0.5, 1.0, 5.0]:
    cfg = MMDBatchCorrectionConfig(
        n_genes=n_features,
        hidden_dim=64,
        latent_dim=16,
        kernel_bandwidth=bw,
    )
    m = DifferentiableMMDBatchCorrection(cfg, rngs=nnx.Rngs(1))
    res, _, _ = m.apply(data_mmd, {}, None)
    score = compute_batch_mixing_score(res["corrected_expression"], batch_labels)
    loss_val = float(res["mmd_loss"])
    print(f"  bandwidth={bw:.1f} -> mixing: {score:.4f}, MMD loss: {loss_val:.4f}")

# %%
# Figure 2: Batch mixing scores comparison
method_names = ["Uncorrected", "Harmony", "MMD", "WGAN"]
mixing_scores = [
    float(jnp.clip(jnp.array(v), -10.0, 10.0))
    for v in [baseline_score, score_harmony, score_mmd, score_wgan]
]
colors = ["#999999", "#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(method_names, mixing_scores, color=colors)
ax.set_ylabel("Batch Mixing Score (Entropy)")
ax.set_title("Batch Mixing Score by Correction Method")
y_top = max(abs(v) for v in mixing_scores) * 1.3 if any(v != 0 for v in mixing_scores) else 1.0
ax.set_ylim(min(0, min(mixing_scores) * 1.2), y_top)
for bar, val in zip(bars, mixing_scores):
    y_pos = min(float(bar.get_height()) + 0.005, y_top * 0.95)
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y_pos,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/batch_mixing_scores.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# Three batch correction strategies were compared on synthetic batched data:
#
# - **Harmony**: Iterative soft-clustering with batch-aware centroid updates --
#   fast, no training needed, works directly on embeddings
# - **MMD**: Autoencoder with Maximum Mean Discrepancy regularization --
#   penalizes batch distribution differences in latent space
# - **WGAN**: Adversarial autoencoder with Wasserstein discriminator --
#   gradient reversal learns batch-invariant representations
#
# All three methods are fully differentiable and JIT-compatible, enabling
# integration into end-to-end optimizable bioinformatics pipelines.
#
# ## Next Steps
#
# - Train MMD/WGAN correctors to improve batch mixing while preserving biology
# - Chain batch correction with clustering or trajectory inference
# - Explore GradNormBalancer from opifex for multi-loss balancing
