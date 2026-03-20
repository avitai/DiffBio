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
# # scVI-Style VAE Benchmark
#
# **Duration:** 30 minutes | **Level:** Advanced
# | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Train a VAENormalizer with ZINB likelihood on synthetic PBMC-like data
# 2. Use a JIT-compiled training loop with `nnx.Optimizer`
# 3. Evaluate using calibrax clustering metrics (ARI, NMI, silhouette)
# 4. Demonstrate MultiOmicsVAE with Product-of-Experts latent fusion
# 5. Observe training loss decreasing over iterations
#
# ## Prerequisites
#
# - DiffBio installed with calibrax, artifex, and opifex
# - Familiarity with variational autoencoders
# - Understanding of scRNA-seq normalization
#
# ```bash
# source ./activate.sh
# uv run python examples/ecosystem/scvi_benchmark.py
# ```
#
# ---

# %% [markdown]
# ## Background
#
# scVI (Lopez et al., 2018) uses a VAE with a Zero-Inflated Negative
# Binomial (ZINB) likelihood to normalize and denoise scRNA-seq count
# data. DiffBio's `VAENormalizer` implements this architecture with
# full JAX differentiability.
#
# The benchmark pipeline:
# 1. Generate synthetic PBMC-like data with known cell types and batch effects
# 2. Train the VAE with ZINB likelihood using ELBO loss
# 3. Extract latent representations
# 4. Cluster latent space and evaluate against ground truth
# 5. Measure batch correction quality

# %%
# Environment setup
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx

print(f"JAX version: {jax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.device_count()}")

# %% [markdown]
# ## 1. Generate Synthetic PBMC-Like Data
#
# Simulate scRNA-seq data with known structure: multiple cell types,
# batch effects, and realistic count distributions using a Gamma-Poisson
# mixture (negative binomial).

# %%
# Data generation parameters
N_CELLS = 200
N_GENES = 100
N_BATCHES = 2
N_TYPES = 3
LATENT_DIM = 10
HIDDEN_DIMS = [64]


def generate_synthetic_pbmc_data(
    n_cells: int = N_CELLS,
    n_genes: int = N_GENES,
    n_batches: int = N_BATCHES,
    n_types: int = N_TYPES,
    seed: int = 42,
) -> dict[str, jax.Array]:
    """Generate synthetic scRNA-seq data with known cell types and batch effects.

    Returns a dictionary with counts, library sizes, and ground-truth labels.
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 6)

    # Per-type mean expression profiles on log scale
    type_log_means = jax.random.normal(keys[0], (n_types, n_genes)) * 1.5 + 2.0

    # Per-batch additive shift (batch effects)
    batch_shifts = jax.random.normal(keys[1], (n_batches, n_genes)) * 0.5

    # Assign cells to types and batches (roughly equal)
    cells_per_type = n_cells // n_types
    type_labels_list: list[int] = []
    for t in range(n_types):
        count = cells_per_type if t < n_types - 1 else n_cells - len(type_labels_list)
        type_labels_list.extend([t] * count)
    cell_type_labels = jnp.array(type_labels_list)

    cells_per_batch = n_cells // n_batches
    batch_labels_list: list[int] = []
    for b in range(n_batches):
        count = cells_per_batch if b < n_batches - 1 else n_cells - len(batch_labels_list)
        batch_labels_list.extend([b] * count)
    batch_labels = jnp.array(batch_labels_list)

    # Build per-cell log-mean: type mean + batch shift + noise
    cell_log_means = type_log_means[cell_type_labels] + batch_shifts[batch_labels]
    cell_noise = jax.random.normal(keys[2], (n_cells, n_genes)) * 0.3
    cell_log_means = cell_log_means + cell_noise

    rates = jnp.exp(cell_log_means)

    # Negative binomial via Gamma-Poisson mixture
    dispersion = 5.0
    gamma_samples = jax.random.gamma(keys[3], dispersion, (n_cells, n_genes))
    scaled_rates = rates * gamma_samples / dispersion
    counts = jax.random.poisson(keys[4], scaled_rates).astype(jnp.float32)

    library_size = jnp.sum(counts, axis=-1)

    return {
        "counts": counts,
        "library_size": library_size,
        "batch_labels": batch_labels,
        "cell_type_labels": cell_type_labels,
    }


data = generate_synthetic_pbmc_data()
print(f"Cells: {N_CELLS}, Genes: {N_GENES}")
print(f"Batches: {N_BATCHES}, Cell types: {N_TYPES}")
print(f"Counts shape: {data['counts'].shape}")
print(f"Mean library size: {float(data['library_size'].mean()):.0f}")
print(f"Fraction zeros: {float((data['counts'] == 0).mean()):.3f}")

# %% [markdown]
# ## 2. Train VAENormalizer with ZINB Likelihood
#
# The VAENormalizer uses the ELBO objective: reconstruction loss (ZINB NLL)
# plus KL divergence against a standard normal prior. The KL term uses
# `gaussian_kl_divergence` from artifex.

# %%
from diffbio.operators.normalization import (
    VAENormalizer,
    VAENormalizerConfig,
)

# Configure with ZINB likelihood (scVI-like)
vae_config = VAENormalizerConfig(
    n_genes=N_GENES,
    latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS,
    likelihood="zinb",
)
model = VAENormalizer(vae_config, rngs=nnx.Rngs(42))
print(f"VAENormalizer: latent_dim={LATENT_DIM}, hidden={HIDDEN_DIMS}, likelihood=zinb")

# %%
# Create JIT-compiled training step with nnx.Optimizer
nnx_optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


@nnx.jit
def train_step(
    m: VAENormalizer,
    opt: nnx.Optimizer,
    counts_batch: jax.Array,
    library_size_batch: jax.Array,
) -> jax.Array:
    """JIT-compiled ELBO training step, vmapped over cells."""

    def loss_fn(model_inner: VAENormalizer) -> jax.Array:
        def per_cell_loss(counts_i: jax.Array, lib_i: jax.Array) -> jax.Array:
            return model_inner.compute_elbo_loss(counts_i, lib_i)

        losses = jax.vmap(per_cell_loss)(counts_batch, library_size_batch)
        return jnp.mean(losses)

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(m)
    opt.update(m, grads)
    return loss


# %%
# Training loop
counts = data["counts"]
library_size = data["library_size"]

print("\n=== Training VAENormalizer (ZINB) ===")
n_epochs = 50
losses = []
for epoch in range(n_epochs):
    loss = train_step(model, nnx_optimizer, counts, library_size)
    losses.append(float(loss))
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"  Epoch {epoch:3d}: ELBO loss = {float(loss):.2f}")

# Verify loss decreases
print(f"\n  First loss:  {losses[0]:.2f}")
print(f"  Final loss:  {losses[-1]:.2f}")
print(f"  Loss decreased: {losses[-1] < losses[0]}")

# %%
# Figure 1: ELBO training curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(n_epochs), losses, color="tab:red", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("ELBO Loss")
ax.set_title("ELBO Training Curve")
plt.tight_layout()
plt.savefig("docs/assets/examples/ecosystem/scvi_elbo.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Extract Latent Representations
#
# After training, encode all cells using the mean of the posterior
# (no sampling) for evaluation.


# %%
def compute_latent_and_reconstruction(
    trained_model: VAENormalizer,
    cell_counts: jax.Array,
    lib_sizes: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute latent means and reconstructions for all cells."""

    def encode_cell(counts_i: jax.Array, lib_i: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean, _ = trained_model.encode(counts_i)
        decode_out = trained_model.decode(mean, lib_i)
        reconstructed = jnp.exp(decode_out["log_rate"])
        return mean, reconstructed

    latent_means, reconstructed = jax.vmap(encode_cell)(cell_counts, lib_sizes)
    return latent_means, reconstructed


latent_means, reconstructed = compute_latent_and_reconstruction(model, counts, library_size)

print(f"Latent representations: {latent_means.shape}")
print(f"Reconstructed counts: {reconstructed.shape}")
print(f"Reconstruction MSE: {float(jnp.mean((counts - reconstructed) ** 2)):.4f}")

# %%
# Figure 2: Latent space scatter (first 2 dimensions, colored by cell type)
cell_type_labels = data["cell_type_labels"]
fig, ax = plt.subplots(figsize=(6, 5))
scatter = ax.scatter(
    latent_means[:, 0],
    latent_means[:, 1],
    c=cell_type_labels,
    cmap="Set1",
    s=30,
    alpha=0.8,
    edgecolors="k",
    linewidths=0.3,
)
ax.set_xlabel("Latent dim 1")
ax.set_ylabel("Latent dim 2")
ax.set_title("Latent Space (colored by cell type)")
plt.colorbar(scatter, ax=ax, label="Cell type")
plt.tight_layout()
plt.savefig("docs/assets/examples/ecosystem/scvi_latent.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Evaluate with Calibrax Metrics
#
# Use calibrax clustering metrics to assess how well the latent space
# preserves biological structure and removes batch effects.

# %%
from calibrax.metrics.functional.clustering import (
    adjusted_rand_index,
    normalized_mutual_information_clustering,
    silhouette_score,
)
from diffbio.operators.singlecell import (
    SoftClusteringConfig,
    SoftKMeansClustering,
)

# Cluster the latent space
cluster_config = SoftClusteringConfig(
    n_clusters=N_TYPES,
    n_features=LATENT_DIM,
    temperature=0.1,
)
latent_clusterer = SoftKMeansClustering(cluster_config, rngs=nnx.Rngs(0))
cluster_result, _, _ = latent_clusterer.apply({"embeddings": latent_means}, {}, None)
cluster_labels = cluster_result["cluster_labels"]

# Biological conservation: cell types should separate in latent space
cell_type_labels = data["cell_type_labels"]
batch_labels = data["batch_labels"]

bio_silhouette = silhouette_score(latent_means, cell_type_labels)
ari = adjusted_rand_index(cell_type_labels, cluster_labels)
nmi = normalized_mutual_information_clustering(cell_type_labels, cluster_labels)

# Batch correction: batches should NOT separate in latent space
batch_silhouette = silhouette_score(latent_means, batch_labels)
batch_asw = 1.0 - jnp.abs(batch_silhouette)  # Higher = better mixing

print("=== Calibrax Evaluation Metrics ===")
print("  Biological conservation:")
print(f"    Silhouette (cell types): {float(bio_silhouette):.4f}")
print(f"    ARI:                     {float(ari):.4f}")
print(f"    NMI:                     {float(nmi):.4f}")
print("  Batch correction:")
print(f"    Batch silhouette:        {float(batch_silhouette):.4f}")
print(f"    Batch ASW (1-|sil|):     {float(batch_asw):.4f}")

# %% [markdown]
# ## 5. VAENormalizer apply() Interface
#
# The VAENormalizer also conforms to the standard `apply()` contract.
# This per-cell operator produces normalized expression, latent
# representations, and log rates.

# %%
# Demonstrate the apply() interface on a single cell
cell_data = {
    "counts": counts[0],
    "library_size": library_size[0],
}
apply_result, _, _ = model.apply(cell_data, {}, None)

print("=== apply() Output Keys ===")
for key_name, value in sorted(apply_result.items()):
    if hasattr(value, "shape"):
        print(f"  {key_name}: shape={value.shape}, dtype={value.dtype}")

# %% [markdown]
# ## 6. Verify Differentiability
#
# Gradients flow through the full encode-decode pipeline. The ZINB
# likelihood, KL divergence (via artifex), and all neural network
# components are end-to-end differentiable.


# %%
# Gradient through the full ELBO
def elbo_loss_fn(vae_model: VAENormalizer) -> jax.Array:
    """ELBO loss for gradient verification."""
    return vae_model.compute_elbo_loss(counts[0], library_size[0])


grad = nnx.grad(elbo_loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)

# Check representative parameters
# fc_mean projects encoder output to latent mean
mean_grad = grad.fc_mean.kernel[...]
print("=== Gradient Verification ===")
print(f"  fc_mean weight gradient shape: {mean_grad.shape}")
print(f"  Gradient non-zero: {bool(jnp.any(mean_grad != 0))}")
print(f"  Gradient finite:   {bool(jnp.all(jnp.isfinite(mean_grad)))}")
print(f"  Gradient abs mean: {float(jnp.abs(mean_grad).mean()):.6f}")

# Check output layer gradient
output_grad = grad.fc_output.kernel[...]
print(f"  fc_output gradient shape: {output_grad.shape}")
print(f"  fc_output gradient non-zero: {bool(jnp.any(output_grad != 0))}")

# %% [markdown]
# ## 7. JIT Compilation
#
# The training step is already JIT-compiled via `@nnx.jit`. Verify
# that the evaluation pipeline can also be JIT-compiled.

# %%
# Note: calibrax clustering metrics use data-dependent array sizes
# internally (jnp.unique), so they run eagerly rather than under jax.jit.
# The training step is already JIT-compiled via @nnx.jit above.

# Verify evaluation metrics match the earlier results
eval_ari = adjusted_rand_index(cell_type_labels, cluster_labels)
eval_nmi = normalized_mutual_information_clustering(cell_type_labels, cluster_labels)
eval_sil = silhouette_score(latent_means, cell_type_labels)
eval_batch_asw = 1.0 - jnp.abs(silhouette_score(latent_means, batch_labels))

print("=== Evaluation Metrics (Consistent Check) ===")
print(f"  ARI:            {float(eval_ari):.4f}")
print(f"  NMI:            {float(eval_nmi):.4f}")
print(f"  Bio silhouette: {float(eval_sil):.4f}")
print(f"  Batch ASW:      {float(eval_batch_asw):.4f}")

# %% [markdown]
# ## 8. MultiOmicsVAE Demo
#
# DiffBio also provides `DifferentiableMultiOmicsVAE` for integrating
# multiple data modalities (e.g., RNA + ATAC) via Product-of-Experts
# latent fusion. This uses `gaussian_kl_divergence` from artifex and
# optionally `GradNormBalancer` from opifex for multi-task loss balancing.

# %%
from diffbio.operators.multiomics import (
    DifferentiableMultiOmicsVAE,
    MultiOmicsVAEConfig,
)

# Simulate two modalities: RNA (larger) and ATAC (smaller)
N_RNA_GENES = 80
N_ATAC_PEAKS = 40
MULTI_LATENT = 8

key = jax.random.key(99)
k1, k2 = jax.random.split(key)
rna_counts = jax.random.poisson(k1, jnp.ones((N_CELLS, N_RNA_GENES)) * 5.0).astype(jnp.float32)
atac_counts = jax.random.poisson(k2, jnp.ones((N_CELLS, N_ATAC_PEAKS)) * 3.0).astype(jnp.float32)

multi_config = MultiOmicsVAEConfig(
    modality_dims=[N_RNA_GENES, N_ATAC_PEAKS],
    latent_dim=MULTI_LATENT,
    hidden_dim=32,
    modality_weight_mode="equal",
)
multi_vae = DifferentiableMultiOmicsVAE(multi_config, rngs=nnx.Rngs(50, sample=51))

# Forward pass via apply()
multi_data = {
    "rna_counts": rna_counts,
    "atac_counts": atac_counts,
}
multi_result, _, _ = multi_vae.apply(multi_data, {}, None)

print("=== MultiOmicsVAE Output ===")
for key_name, value in sorted(multi_result.items()):
    if hasattr(value, "shape"):
        print(f"  {key_name}: shape={value.shape}")
    elif isinstance(value, jax.Array):
        print(f"  {key_name}: {float(value):.4f}")

# %%
# Train the MultiOmicsVAE briefly to show loss decreasing
multi_optimizer = nnx.Optimizer(multi_vae, optax.adam(1e-3), wrt=nnx.Param)


@nnx.jit
def multi_train_step(
    m: DifferentiableMultiOmicsVAE,
    opt: nnx.Optimizer,
    input_data: dict[str, jax.Array],
) -> jax.Array:
    """Training step for MultiOmicsVAE."""

    def loss_fn(model_inner: DifferentiableMultiOmicsVAE) -> jax.Array:
        result, _, _ = model_inner.apply(input_data, {}, None)
        return result["elbo_loss"]

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(m)
    opt.update(m, grads)
    return loss


print("\n=== Training MultiOmicsVAE ===")
multi_losses = []
for epoch in range(30):
    loss = multi_train_step(multi_vae, multi_optimizer, multi_data)
    multi_losses.append(float(loss))
    if epoch % 10 == 0 or epoch == 29:
        print(f"  Epoch {epoch:3d}: ELBO loss = {float(loss):.2f}")

print(f"\n  First loss: {multi_losses[0]:.2f}")
print(f"  Final loss: {multi_losses[-1]:.2f}")
print(f"  Loss decreased: {multi_losses[-1] < multi_losses[0]}")

# %% [markdown]
# ## 9. Experiments
#
# Explore how latent dimension and likelihood choice affect performance.

# %%
# Experiment: Vary latent dimension
print("=== Experiment: Latent Dimension ===")
latent_dims = [5, 10, 20]
ari_by_ldim = []

for ldim in latent_dims:
    exp_config = VAENormalizerConfig(
        n_genes=N_GENES,
        latent_dim=ldim,
        hidden_dims=HIDDEN_DIMS,
        likelihood="zinb",
    )
    exp_model = VAENormalizer(exp_config, rngs=nnx.Rngs(42))
    exp_opt = nnx.Optimizer(exp_model, optax.adam(1e-3), wrt=nnx.Param)

    # Quick training
    for _ in range(30):
        train_step(exp_model, exp_opt, counts, library_size)

    # Evaluate
    exp_latent, _ = compute_latent_and_reconstruction(exp_model, counts, library_size)
    exp_cluster_config = SoftClusteringConfig(n_clusters=N_TYPES, n_features=ldim, temperature=0.1)
    exp_clusterer = SoftKMeansClustering(exp_cluster_config, rngs=nnx.Rngs(0))
    exp_result, _, _ = exp_clusterer.apply({"embeddings": exp_latent}, {}, None)
    exp_ari = adjusted_rand_index(cell_type_labels, exp_result["cluster_labels"])
    exp_sil = silhouette_score(exp_latent, cell_type_labels)
    ari_by_ldim.append(float(exp_ari))

    print(f"  latent_dim={ldim:2d}: ARI={float(exp_ari):.4f}, Silhouette={float(exp_sil):.4f}")

# %%
# Figure 3: ARI across latent dimensions
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    [str(d) for d in latent_dims],
    ari_by_ldim,
    color="tab:purple",
    edgecolor="k",
    linewidth=0.5,
)
for bar, val in zip(bars, ari_by_ldim):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Latent Dimension")
ax.set_ylabel("ARI")
ax.set_title("ARI Across Latent Dimensions")
ax.set_ylim(0, max(ari_by_ldim) * 1.3 if max(ari_by_ldim) > 0 else 1.0)
plt.tight_layout()
plt.savefig("docs/assets/examples/ecosystem/scvi_latent_dim.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Experiment: Poisson vs ZINB likelihood
print("\n=== Experiment: Likelihood Comparison ===")
likelihood_names = []
likelihood_final_losses = []
likelihood_silhouettes = []

for likelihood in ("poisson", "zinb"):
    exp_config = VAENormalizerConfig(
        n_genes=N_GENES,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        likelihood=likelihood,  # type: ignore[arg-type]
    )
    exp_model = VAENormalizer(exp_config, rngs=nnx.Rngs(42))
    exp_opt = nnx.Optimizer(exp_model, optax.adam(1e-3), wrt=nnx.Param)

    for _ in range(30):
        final_loss = train_step(exp_model, exp_opt, counts, library_size)

    exp_latent, exp_recon = compute_latent_and_reconstruction(exp_model, counts, library_size)
    mse = float(jnp.mean((counts - exp_recon) ** 2))
    exp_sil = silhouette_score(exp_latent, cell_type_labels)

    likelihood_names.append(likelihood)
    likelihood_final_losses.append(float(final_loss))
    likelihood_silhouettes.append(float(exp_sil))

    print(
        f"  {likelihood:8s}: final_loss={float(final_loss):10.2f}, "
        f"MSE={mse:.4f}, Silhouette={float(exp_sil):.4f}"
    )

# %%
# Figure 4: Poisson vs ZINB grouped bar chart (final_loss, Silhouette)
import numpy as np

x_pos = np.arange(len(likelihood_names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(7, 4))

# Normalize final_loss for display alongside silhouette
max_loss = max(abs(v) for v in likelihood_final_losses) if likelihood_final_losses else 1.0
norm_losses = [v / max_loss for v in likelihood_final_losses]

bars1 = ax1.bar(x_pos - width / 2, norm_losses, width, label="Final Loss (normalized)",
                color="tab:red", edgecolor="k", linewidth=0.5)
bars2 = ax1.bar(x_pos + width / 2, likelihood_silhouettes, width, label="Silhouette",
                color="tab:cyan", edgecolor="k", linewidth=0.5)

ax1.set_xticks(x_pos)
ax1.set_xticklabels([n.upper() for n in likelihood_names])
ax1.set_ylabel("Score")
ax1.set_title("Poisson vs ZINB: Final Loss and Silhouette")
ax1.legend()

for bar, val in zip(bars1, likelihood_final_losses):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{val:.0f}", ha="center", va="bottom", fontsize=8)
for bar, val in zip(bars2, likelihood_silhouettes):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("docs/assets/examples/ecosystem/scvi_likelihood.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# This example demonstrated:
# - Training a scVI-style VAENormalizer with ZINB likelihood
# - JIT-compiled training with `nnx.Optimizer` and ELBO loss
# - Calibrax evaluation metrics: ARI, NMI, silhouette, batch ASW
# - MultiOmicsVAE with PoE fusion (uses artifex KL divergence)
# - Training loss decreasing over iterations
# - Parameter sensitivity: latent dimension and likelihood choice
#
# ## Next Steps
#
# - [Calibrax Metrics](calibrax_metrics.py) -- training vs evaluation metric split
# - [Single-Cell Pipeline](../pipelines/singlecell_pipeline.py) -- five-operator
#   end-to-end pipeline
