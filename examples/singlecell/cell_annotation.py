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
# # Cell Type Annotation with Three Modes
#
# **Duration:** 15 minutes | **Level:** Intermediate
# | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Understand three cell type annotation strategies (celltypist, cellassign, scanvi)
# 2. Apply DifferentiableCellAnnotator in each mode on synthetic multi-type data
# 3. Compare predictions and verify differentiability across all modes
#
# ## Prerequisites
#
# - DiffBio installed (see setup instructions)
# - Basic understanding of cell type classification in single-cell analysis
#
# ```bash
# source ./activate.sh
# uv run python examples/singlecell/cell_annotation.py
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
# ## 1. Create Synthetic Multi-Type Data
#
# Generate expression profiles for 3 cell types (50 cells each, 20 genes).
# Each type has a distinct expression signature: different genes are upregulated
# per type, mimicking real marker-gene patterns.

# %%
# Synthetic data: 3 cell types, 50 cells each, 20 genes
n_types = 3
n_cells_per_type = 50
n_cells = n_types * n_cells_per_type
n_genes = 20

key = jax.random.key(42)
k1, k2, k3, k4 = jax.random.split(key, 4)

# Base expression: low-level noise for all genes
base = jax.random.poisson(k1, jnp.ones((n_cells, n_genes)) * 1.0)

# Per-type marker upregulation: each type upregulates a distinct gene block
# Type 0: genes 0-6 upregulated, Type 1: genes 7-13, Type 2: genes 14-19
type_labels_true = jnp.concatenate([
    jnp.full(n_cells_per_type, 0),
    jnp.full(n_cells_per_type, 1),
    jnp.full(n_cells_per_type, 2),
])

marker_signal = jnp.zeros((n_cells, n_genes))
marker_signal = marker_signal.at[:50, :7].set(10.0)
marker_signal = marker_signal.at[50:100, 7:14].set(10.0)
marker_signal = marker_signal.at[100:, 14:].set(10.0)

counts = (base + marker_signal).astype(jnp.float32)

# Add small noise to prevent degeneracies
counts = counts + jax.random.uniform(k2, counts.shape) * 0.1

print(f"Expression matrix shape: {counts.shape}")
print(f"True label distribution: {jnp.bincount(type_labels_true, length=n_types)}")
print(f"Mean expression per type (first 3 genes):")
for t in range(n_types):
    mask = type_labels_true == t
    mean_expr = jnp.where(mask[:, None], counts, 0.0).sum(axis=0) / mask.sum()
    print(f"  Type {t}: {mean_expr[:3]}")

# %% [markdown]
# ## 2. Mode 1 -- Celltypist (MLP Classifier)
#
# The celltypist mode encodes counts through a VAE encoder, then applies
# a linear classifier head on the latent space. This is the simplest mode,
# suitable when labelled training data is available.

# %%
from diffbio.operators.singlecell import (
    CellAnnotatorConfig,
    DifferentiableCellAnnotator,
)

# Configure celltypist mode
config_ct = CellAnnotatorConfig(
    annotation_mode="celltypist",
    n_cell_types=n_types,
    n_genes=n_genes,
    latent_dim=8,
    hidden_dims=[32, 16],
    stochastic=True,
    stream_name="sample",
)
annotator_ct = DifferentiableCellAnnotator(config_ct, rngs=nnx.Rngs(0))
print(f"Celltypist annotator created: {type(annotator_ct).__name__}")

# %%
# Run celltypist annotation
data_ct = {"counts": counts}
result_ct, state_ct, meta_ct = annotator_ct.apply(data_ct, {}, None)

probs_ct = result_ct["cell_type_probabilities"]
labels_ct = result_ct["cell_type_labels"]
latent_ct = result_ct["latent"]

print(f"Probabilities shape: {probs_ct.shape}")
print(f"Predicted labels shape: {labels_ct.shape}")
print(f"Latent shape: {latent_ct.shape}")
print(f"Predicted label counts: {jnp.bincount(labels_ct, length=n_types)}")

# %% [markdown]
# ## 3. Mode 2 -- Cellassign (Marker Matrix Guided)
#
# The cellassign mode uses a binary marker matrix that specifies which genes
# are markers for each cell type. It computes per-type Poisson log-likelihoods
# using only marker genes, then applies softmax to get type probabilities.

# %%
# Build binary marker matrix: (n_types, n_genes)
# Type 0 markers: genes 0-6, Type 1: genes 7-13, Type 2: genes 14-19
marker_matrix = jnp.zeros((n_types, n_genes))
marker_matrix = marker_matrix.at[0, :7].set(1.0)
marker_matrix = marker_matrix.at[1, 7:14].set(1.0)
marker_matrix = marker_matrix.at[2, 14:].set(1.0)

print(f"Marker matrix shape: {marker_matrix.shape}")
print(f"Markers per type: {marker_matrix.sum(axis=1)}")

# %%
# Configure cellassign mode
config_ca = CellAnnotatorConfig(
    annotation_mode="cellassign",
    n_cell_types=n_types,
    n_genes=n_genes,
    latent_dim=8,
    hidden_dims=[32, 16],
    marker_matrix_shape=(n_types, n_genes),
    stochastic=True,
    stream_name="sample",
)
annotator_ca = DifferentiableCellAnnotator(config_ca, rngs=nnx.Rngs(1))

# Run cellassign annotation
data_ca = {"counts": counts, "marker_matrix": marker_matrix}
result_ca, state_ca, meta_ca = annotator_ca.apply(data_ca, {}, None)

probs_ca = result_ca["cell_type_probabilities"]
labels_ca = result_ca["cell_type_labels"]

print(f"Cellassign predicted labels: {jnp.bincount(labels_ca, length=n_types)}")
print(f"Max probability per cell (mean): {probs_ca.max(axis=1).mean():.4f}")

# %% [markdown]
# ## 4. Mode 3 -- Scanvi (Semi-supervised VAE)
#
# The scanvi mode uses a semi-supervised VAE with type-conditioned priors
# in latent space. Each cell type has learned prior parameters mu_y and
# logvar_y, so that different types occupy distinct latent regions.
# For labelled cells, the known labels guide the prior; for unlabelled
# cells, the classifier predictions are used.

# %%
# Configure scanvi mode with partial labels
config_sv = CellAnnotatorConfig(
    annotation_mode="scanvi",
    n_cell_types=n_types,
    n_genes=n_genes,
    latent_dim=8,
    hidden_dims=[32, 16],
    gene_likelihood="poisson",
    stochastic=True,
    stream_name="sample",
)
annotator_sv = DifferentiableCellAnnotator(config_sv, rngs=nnx.Rngs(2))

# Provide labels for 20% of cells (semi-supervised)
n_labeled = 30
label_indices = jnp.arange(0, n_labeled)
known_labels = type_labels_true[:n_labeled]

data_sv = {
    "counts": counts,
    "known_labels": known_labels,
    "label_indices": label_indices,
}
result_sv, state_sv, meta_sv = annotator_sv.apply(data_sv, {}, None)

probs_sv = result_sv["cell_type_probabilities"]
labels_sv = result_sv["cell_type_labels"]

print(f"Scanvi predicted labels: {jnp.bincount(labels_sv, length=n_types)}")
print(f"Labelled cell predictions match known: "
      f"{bool(jnp.all(labels_sv[:n_labeled] == known_labels))}")

# %% [markdown]
# ## 5. Compare Predictions Across Modes
#
# Each mode starts from the same synthetic data. Since these are untrained
# models, the predictions reflect initial weights rather than learned patterns.
# With training, all three modes converge to correct annotations.

# %%
# Compare predictions across modes
print("=== Prediction Comparison ===")
print(f"{'Mode':<12} {'Type 0':>8} {'Type 1':>8} {'Type 2':>8} {'Mean Confidence':>16}")
print("-" * 54)
for name, probs, labels in [
    ("Celltypist", probs_ct, labels_ct),
    ("Cellassign", probs_ca, labels_ca),
    ("Scanvi", probs_sv, labels_sv),
]:
    counts_per_type = jnp.bincount(labels, length=n_types)
    mean_conf = probs.max(axis=1).mean()
    print(f"{name:<12} {int(counts_per_type[0]):>8} {int(counts_per_type[1]):>8} "
          f"{int(counts_per_type[2]):>8} {float(mean_conf):>16.4f}")

# Check cellassign accuracy (should be best with explicit markers)
accuracy_ca = jnp.mean(labels_ca == type_labels_true)
print(f"\nCellassign accuracy with known markers: {float(accuracy_ca):.4f}")

# %%
# Figure 1: Annotation confidence by mode
mode_names = ["Celltypist", "Cellassign", "Scanvi"]
mean_confidences = [
    float(probs_ct.max(axis=1).mean()),
    float(probs_ca.max(axis=1).mean()),
    float(probs_sv.max(axis=1).mean()),
]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(mode_names, mean_confidences, color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_ylabel("Mean Max Probability")
ax.set_ylim(0, 1.0)
ax.set_title("Annotation Confidence by Mode")
for bar, val in zip(bars, mean_confidences):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/annotation_confidence.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Verify Differentiability
#
# Gradients must flow through each annotation mode. This is the core DiffBio
# value proposition: cell type annotation becomes part of differentiable
# end-to-end pipelines.

# %%
# Gradient check for celltypist mode
print("=== Gradient Flow Verification ===\n")


def loss_fn_ct(input_data):
    """Scalar loss from celltypist annotation."""
    res, _, _ = annotator_ct.apply(input_data, {}, None)
    return res["cell_type_probabilities"].sum()


grad_ct = jax.grad(loss_fn_ct)(data_ct)
print("Celltypist:")
print(f"  Gradient shape: {grad_ct['counts'].shape}")
print(f"  Non-zero: {bool(jnp.any(grad_ct['counts'] != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(grad_ct['counts'])))}")

# %%
# Gradient check for cellassign mode
# Cellassign uses Poisson log-likelihoods; differentiate a weighted
# probability to show non-trivial gradient flow through marker scoring.


def loss_fn_ca(input_counts):
    """Scalar loss from cellassign annotation (counts only)."""
    d = {"counts": input_counts, "marker_matrix": marker_matrix}
    res, _, _ = annotator_ca.apply(d, {}, None)
    # Weight by type index to break softmax symmetry
    weights = jnp.arange(n_types, dtype=jnp.float32)
    return (res["cell_type_probabilities"] * weights[None, :]).sum()


grad_ca = jax.grad(loss_fn_ca)(counts)
print("Cellassign:")
print(f"  Gradient shape: {grad_ca.shape}")
print(f"  Non-zero: {bool(jnp.any(grad_ca != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(grad_ca)))}")

# %%
# Gradient check for scanvi mode
# Scanvi data includes integer arrays (known_labels, label_indices) that
# jax.grad cannot differentiate. Differentiate only w.r.t. counts.


def loss_fn_sv(input_counts):
    """Scalar loss from scanvi annotation (counts only)."""
    d = {
        "counts": input_counts,
        "known_labels": known_labels,
        "label_indices": label_indices,
    }
    res, _, _ = annotator_sv.apply(d, {}, None)
    return res["cell_type_probabilities"].sum()


grad_sv = jax.grad(loss_fn_sv)(counts)
print("Scanvi:")
print(f"  Gradient shape: {grad_sv.shape}")
print(f"  Non-zero: {bool(jnp.any(grad_sv != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(grad_sv)))}")

# %% [markdown]
# ## 7. JIT Compilation
#
# All three modes are compatible with JAX JIT for accelerated execution.
# JIT compilation traces the function once and caches the compiled version.

# %%
# JIT celltypist
print("=== JIT Compilation ===\n")

jit_ct = jax.jit(lambda d: annotator_ct.apply(d, {}, None))
result_jit_ct, _, _ = jit_ct(data_ct)
match_ct = jnp.allclose(
    result_ct["cell_type_probabilities"],
    result_jit_ct["cell_type_probabilities"],
    atol=1e-5,
)
print(f"Celltypist JIT matches eager: {bool(match_ct)}")

# %%
# JIT cellassign
jit_ca = jax.jit(lambda d: annotator_ca.apply(d, {}, None))
result_jit_ca, _, _ = jit_ca(data_ca)
match_ca = jnp.allclose(
    result_ca["cell_type_probabilities"],
    result_jit_ca["cell_type_probabilities"],
    atol=1e-5,
)
print(f"Cellassign JIT matches eager: {bool(match_ca)}")

# %%
# JIT scanvi
jit_sv = jax.jit(lambda d: annotator_sv.apply(d, {}, None))
result_jit_sv, _, _ = jit_sv(data_sv)
match_sv = jnp.allclose(
    result_sv["cell_type_probabilities"],
    result_jit_sv["cell_type_probabilities"],
    atol=1e-5,
)
print(f"Scanvi JIT matches eager: {bool(match_sv)}")

# %% [markdown]
# ## 8. Experiments
#
# ### Vary the number of labelled cells for scanvi
#
# The scanvi mode benefits from partial labels. Increasing the labelled
# fraction should improve annotation confidence for the remaining cells.

# %%
# Experiment: vary labelled fraction in scanvi
print("=== Experiment: Labelled Fraction Effect on Scanvi ===\n")

labelled_counts = [5, 15, 30, 50]
scanvi_confidences = []

for n_lab in labelled_counts:
    lab_idx = jnp.arange(0, n_lab)
    lab_labels = type_labels_true[:n_lab]
    d = {
        "counts": counts,
        "known_labels": lab_labels,
        "label_indices": lab_idx,
    }
    res, _, _ = annotator_sv.apply(d, {}, None)
    p = res["cell_type_probabilities"]
    mean_conf = float(p.max(axis=1).mean())
    scanvi_confidences.append(mean_conf)
    print(f"  {n_lab:>3} labelled cells -> mean confidence: {mean_conf:.4f}")

# %%
# Figure 2: Scanvi confidence vs number of labelled cells
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(labelled_counts, scanvi_confidences, "o-", color="#4C72B0", linewidth=2, markersize=7)
ax.set_xlabel("Number of Labelled Cells")
ax.set_ylabel("Mean Max Probability")
ax.set_title("Scanvi Confidence vs Labelled Cells")
ax.set_xticks(labelled_counts)
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/annotation_labelled_fraction.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Compare gene likelihoods for scanvi
#
# The scanvi mode supports both Poisson and ZINB gene likelihoods.

# %%
# Experiment: Poisson vs ZINB likelihood in scanvi
config_zinb = CellAnnotatorConfig(
    annotation_mode="scanvi",
    n_cell_types=n_types,
    n_genes=n_genes,
    latent_dim=8,
    hidden_dims=[32, 16],
    gene_likelihood="zinb",
    stochastic=True,
    stream_name="sample",
)
annotator_zinb = DifferentiableCellAnnotator(config_zinb, rngs=nnx.Rngs(3))

result_zinb, _, _ = annotator_zinb.apply(data_sv, {}, None)
labels_zinb = result_zinb["cell_type_labels"]

print("Scanvi gene likelihood comparison:")
print(f"  Poisson predictions: {jnp.bincount(labels_sv, length=n_types)}")
print(f"  ZINB predictions:    {jnp.bincount(labels_zinb, length=n_types)}")

# %% [markdown]
# ## Summary
#
# In this example, three cell type annotation modes were demonstrated:
#
# - **Celltypist**: MLP classifier on VAE latent space -- fast, requires training labels
# - **Cellassign**: Marker-gene guided Poisson likelihood -- leverages prior marker knowledge
# - **Scanvi**: Semi-supervised VAE with type-conditioned priors -- uses partial labels
#
# All modes produce differentiable outputs, allowing gradient-based optimization,
# and are JIT-compatible for efficient execution.
#
# ## Next Steps
#
# - Train the annotators using `compute_elbo_loss()` for end-to-end optimization
# - Chain annotation with batch correction for multi-sample workflows
# - Explore the API reference for CellAnnotatorConfig options
