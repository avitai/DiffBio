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
# # Calibrax Metrics for Evaluation
#
# **Duration:** 25 minutes | **Level:** Advanced
# | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Distinguish training losses (differentiable surrogates) from evaluation
#    metrics (exact, non-differentiable)
# 2. Use DifferentiableAUROC for training and ExactAUROC for evaluation
# 3. Use calibrax clustering metrics (ARI, NMI, silhouette) for evaluation
# 4. Apply ShannonDiversityLoss as a training regularizer via calibrax entropy
#
# ## Prerequisites
#
# - DiffBio installed with calibrax
# - Familiarity with single-cell clustering concepts
# - Understanding of the `apply()` contract
#
# ```bash
# source ./activate.sh
# uv run python examples/ecosystem/calibrax_metrics.py
# ```
#
# ---

# %% [markdown]
# ## Training vs Evaluation Metrics
#
# Evaluation metrics such as AUROC, ARI, NMI, and silhouette score involve
# sorting, argmax, or other non-differentiable operations. They cannot be
# used directly as training objectives because gradients are zero almost
# everywhere.
#
# DiffBio provides differentiable surrogates for training and exact
# implementations (backed by calibrax) for evaluation:
#
# | Purpose    | AUROC                   | Clustering Quality      |
# |------------|-------------------------|-------------------------|
# | Training   | `DifferentiableAUROC`   | `ClusteringCompactnessLoss` |
# | Evaluation | `ExactAUROC` (calibrax) | `silhouette_score` etc. |

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
# ## 1. DifferentiableAUROC vs ExactAUROC
#
# `DifferentiableAUROC` approximates the Wilcoxon-Mann-Whitney statistic
# using a sigmoid, making it differentiable. `ExactAUROC` delegates to
# calibrax's trapezoidal-rule implementation for exact evaluation.

# %%
from diffbio.losses import DifferentiableAUROC, ExactAUROC

# Create both metric variants
diff_auroc = DifferentiableAUROC(temperature=1.0)
exact_auroc = ExactAUROC()

# Generate binary classification data
key = jax.random.key(42)
k1, k2 = jax.random.split(key)
n_samples = 100

# Predictions: well-separated positive and negative scores
pos_scores = 0.7 + 0.2 * jax.random.normal(k1, (n_samples // 2,))
neg_scores = 0.3 + 0.2 * jax.random.normal(k2, (n_samples // 2,))
predictions = jnp.concatenate([pos_scores, neg_scores])
labels = jnp.concatenate([jnp.ones(n_samples // 2), jnp.zeros(n_samples // 2)])

# Compare training surrogate vs exact metric
diff_value = diff_auroc(predictions, labels)
exact_value = exact_auroc(predictions, labels)

print("=== AUROC Comparison ===")
print(f"  DifferentiableAUROC (training):  {float(diff_value):.4f}")
print(f"  ExactAUROC (evaluation):         {float(exact_value):.4f}")
print(f"  Difference:                      {abs(float(diff_value) - float(exact_value)):.4f}")


# %%
# Verify DifferentiableAUROC has gradients, ExactAUROC does not
def diff_loss(preds: jax.Array) -> jax.Array:
    """Differentiable AUROC loss for gradient check."""
    return diff_auroc(preds, labels)


grad_diff = jax.grad(diff_loss)(predictions)
print(f"\nDifferentiableAUROC gradient shape: {grad_diff.shape}")
print(f"DifferentiableAUROC gradient non-zero: {bool(jnp.any(grad_diff != 0))}")
print(f"DifferentiableAUROC gradient finite: {bool(jnp.all(jnp.isfinite(grad_diff)))}")

# %% [markdown]
# ## 2. Train SoftKMeansClustering on Synthetic Data
#
# Train cluster centroids using `ClusteringCompactnessLoss` as the
# differentiable training objective, then evaluate the result using
# calibrax's exact clustering metrics.

# %%
from diffbio.operators.singlecell import (
    SoftClusteringConfig,
    SoftKMeansClustering,
)

# Generate structured synthetic data: 3 groups with distinct expression
N_CELLS = 60
N_GENES = 20
N_CLUSTERS = 3

key = jax.random.key(10)
keys = jax.random.split(key, 4)

# Create 3 well-separated clusters in gene space
group_centers = jax.random.normal(keys[0], (N_CLUSTERS, N_GENES)) * 3.0
true_labels = jnp.repeat(jnp.arange(N_CLUSTERS), N_CELLS // N_CLUSTERS)
noise = jax.random.normal(keys[1], (N_CELLS, N_GENES)) * 0.5
embeddings = group_centers[true_labels] + noise

print(f"Synthetic data: {N_CELLS} cells, {N_GENES} genes, {N_CLUSTERS} true clusters")
print(f"Embeddings shape: {embeddings.shape}")
print(f"True labels distribution: {[int(jnp.sum(true_labels == k)) for k in range(N_CLUSTERS)]}")

# %%
# Configure and train the clusterer
from diffbio.losses import ClusteringCompactnessLoss

cluster_config = SoftClusteringConfig(
    n_clusters=N_CLUSTERS,
    n_features=N_GENES,
    temperature=1.0,
)
clusterer = SoftKMeansClustering(cluster_config, rngs=nnx.Rngs(20))
compactness_loss = ClusteringCompactnessLoss(separation_weight=1.0, min_separation=2.0)

# Training loop using nnx.Optimizer
optimizer = nnx.Optimizer(clusterer, optax.adam(0.05), wrt=nnx.Param)

data = {"embeddings": embeddings}


@nnx.jit
def train_step(
    model: SoftKMeansClustering,
    opt: nnx.Optimizer,
    input_data: dict[str, jax.Array],
) -> jax.Array:
    """Single training step for clustering."""

    def loss_fn(m: SoftKMeansClustering) -> jax.Array:
        result, _, _ = m.apply(input_data, {}, None)
        return compactness_loss(
            result["embeddings"],
            result["cluster_assignments"],
            result["centroids"],
        )

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
    opt.update(model, grads)
    return loss


print("\n=== Training Clustering ===")
n_epochs = 50
training_losses = []
for epoch in range(n_epochs):
    loss = train_step(clusterer, optimizer, data)
    training_losses.append(float(loss))
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"  Epoch {epoch:3d}: loss={float(loss):.4f}")

# %%
# Figure 1: Training loss curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(n_epochs), training_losses, color="tab:blue", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Curve")
plt.tight_layout()
plt.savefig("docs/assets/examples/ecosystem/calibrax_loss.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Evaluate with Calibrax Clustering Metrics
#
# After training, evaluate using exact metrics from calibrax:
# - **Adjusted Rand Index (ARI)**: measures agreement between predicted
#   and true labels, adjusted for chance
# - **Normalized Mutual Information (NMI)**: measures shared information
#   between predicted and true clusters
# - **Silhouette Score**: measures cluster cohesion vs separation

# %%
from calibrax.metrics.functional.clustering import (
    adjusted_rand_index,
    normalized_mutual_information_clustering,
    silhouette_score,
)

# Get final cluster predictions
final_result, _, _ = clusterer.apply(data, {}, None)
predicted_labels = final_result["cluster_labels"]

# Compute calibrax evaluation metrics
ari = adjusted_rand_index(true_labels, predicted_labels)
nmi = normalized_mutual_information_clustering(true_labels, predicted_labels)
sil = silhouette_score(embeddings, predicted_labels)

print("=== Calibrax Evaluation Metrics ===")
print(f"  Adjusted Rand Index (ARI): {float(ari):.4f}")
print(f"  Normalized Mutual Info:    {float(nmi):.4f}")
print(f"  Silhouette Score:          {float(sil):.4f}")
print(
    f"\n  Predicted cluster sizes: "
    f"{[int(jnp.sum(predicted_labels == k)) for k in range(N_CLUSTERS)]}"
)

# %%
# Figure 2: Evaluation metrics bar chart
metric_names = ["ARI", "NMI", "Silhouette"]
metric_values = [float(ari), float(nmi), float(sil)]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(metric_names, metric_values, color=["tab:blue", "tab:orange", "tab:green"],
              edgecolor="k", linewidth=0.5)
for bar, val in zip(bars, metric_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Score")
ax.set_title("Calibrax Evaluation Metrics")
ax.set_ylim(0, max(metric_values) * 1.2 if max(metric_values) > 0 else 1.0)
plt.tight_layout()
plt.savefig("docs/assets/examples/ecosystem/calibrax_metrics_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. ShannonDiversityLoss as Training Regularizer
#
# `ShannonDiversityLoss` wraps calibrax's `entropy` function to compute
# the mean Shannon entropy of soft cluster assignments. This can
# regularize training to prevent cluster collapse (all cells assigned
# to one cluster).

# %%
from diffbio.losses import ShannonDiversityLoss, SimpsonDiversityLoss

shannon_loss = ShannonDiversityLoss()
simpson_loss = SimpsonDiversityLoss()

# Compute on the trained cluster assignments
assignments = final_result["cluster_assignments"]
shannon_val = shannon_loss(assignments)
simpson_val = simpson_loss(assignments)

print("=== Diversity Metrics on Trained Clusters ===")
print(f"  Shannon entropy (higher = more diverse): {float(shannon_val):.4f}")
print(f"  Max Shannon entropy (log {N_CLUSTERS}):            {float(jnp.log(N_CLUSTERS)):.4f}")
print(f"  Simpson index (lower = more diverse):    {float(simpson_val):.4f}")


# %%
# Demonstrate that ShannonDiversityLoss is differentiable (backed by calibrax entropy)
def diversity_loss_fn(model: SoftKMeansClustering) -> jax.Array:
    """Shannon diversity as a differentiable training objective."""
    result, _, _ = model.apply(data, {}, None)
    # Negate because higher entropy is better (maximization as minimization)
    return -shannon_loss(result["cluster_assignments"])


diversity_grad = nnx.grad(diversity_loss_fn, argnums=nnx.DiffState(0, nnx.Param))(clusterer)
centroid_grad = diversity_grad.centroids[...]
print(f"\nShannon diversity gradient shape: {centroid_grad.shape}")
print(f"Shannon diversity gradient non-zero: {bool(jnp.any(centroid_grad != 0))}")
print(f"Shannon diversity gradient finite: {bool(jnp.all(jnp.isfinite(centroid_grad)))}")

# %% [markdown]
# ## 5. Combined Training: Compactness + Diversity
#
# A practical training objective combines compactness (tight clusters)
# with diversity (prevent collapse), using differentiable losses that
# are then validated with exact calibrax metrics.

# %%
# Reset the clusterer for a combined training demo
combined_clusterer = SoftKMeansClustering(cluster_config, rngs=nnx.Rngs(30))
combined_optimizer = nnx.Optimizer(combined_clusterer, optax.adam(0.05), wrt=nnx.Param)
combined_compactness = ClusteringCompactnessLoss(separation_weight=1.0, min_separation=2.0)
combined_shannon = ShannonDiversityLoss()

diversity_weight = 0.5


@nnx.jit
def combined_train_step(
    model: SoftKMeansClustering,
    opt: nnx.Optimizer,
    input_data: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Training step with compactness + diversity regularization."""

    def loss_fn(m: SoftKMeansClustering) -> jax.Array:
        result, _, _ = m.apply(input_data, {}, None)
        compact = combined_compactness(
            result["embeddings"],
            result["cluster_assignments"],
            result["centroids"],
        )
        # Negate shannon: higher entropy = more diverse, which is good
        diversity = -combined_shannon(result["cluster_assignments"])
        return compact + diversity_weight * diversity

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
    opt.update(model, grads)

    # Return individual components for monitoring
    result, _, _ = model.apply(input_data, {}, None)
    compact_val = combined_compactness(
        result["embeddings"],
        result["cluster_assignments"],
        result["centroids"],
    )
    diversity_val = combined_shannon(result["cluster_assignments"])
    return loss, compact_val, diversity_val


print("=== Combined Training: Compactness + Diversity ===")
for epoch in range(n_epochs):
    total_loss, compact_val, div_val = combined_train_step(
        combined_clusterer, combined_optimizer, data
    )
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(
            f"  Epoch {epoch:3d}: total={float(total_loss):.4f}, "
            f"compact={float(compact_val):.4f}, "
            f"diversity={float(div_val):.4f}"
        )

# Evaluate with calibrax metrics
combined_result, _, _ = combined_clusterer.apply(data, {}, None)
combined_pred = combined_result["cluster_labels"]
combined_ari = adjusted_rand_index(true_labels, combined_pred)
combined_nmi = normalized_mutual_information_clustering(true_labels, combined_pred)
combined_sil = silhouette_score(embeddings, combined_pred)

print("\n  Evaluation (calibrax):")
print(f"    ARI:        {float(combined_ari):.4f}")
print(f"    NMI:        {float(combined_nmi):.4f}")
print(f"    Silhouette: {float(combined_sil):.4f}")

# %% [markdown]
# ## 6. JIT Compilation
#
# The DiffBio differentiable losses (ClusteringCompactnessLoss,
# ShannonDiversityLoss, DifferentiableAUROC) are all JIT-compatible.
# Calibrax's clustering evaluation metrics (ARI, NMI, silhouette)
# use data-dependent array sizes internally, so they run eagerly.


# %%
# JIT the differentiable training losses
@jax.jit
def jit_training_losses(
    emb: jax.Array,
    soft_assign: jax.Array,
    cents: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled training losses."""
    compact = compactness_loss(emb, soft_assign, cents)
    diversity = shannon_loss(soft_assign)
    return compact, diversity


# JIT the DifferentiableAUROC
jit_diff_auroc = jax.jit(lambda p, l: diff_auroc(p, l))

# Run JIT-compiled losses
jit_compact, jit_div = jit_training_losses(
    embeddings,
    combined_result["cluster_assignments"],
    combined_result["centroids"],
)
jit_auroc_val = jit_diff_auroc(predictions, labels)

print("=== JIT-Compiled Training Losses ===")
print(f"  Compactness loss (JIT):       {float(jit_compact):.4f}")
print(f"  Shannon diversity (JIT):      {float(jit_div):.4f}")
print(f"  DifferentiableAUROC (JIT):    {float(jit_auroc_val):.4f}")

# Calibrax evaluation metrics run eagerly
print(f"\n  Calibrax ARI (eager):         {float(combined_ari):.4f}")
print(f"  Calibrax NMI (eager):         {float(combined_nmi):.4f}")
print(f"  Calibrax Silhouette (eager):  {float(combined_sil):.4f}")

# %% [markdown]
# ## 7. Experiments
#
# Explore how the temperature parameter affects clustering quality.

# %%
print("=== Experiment: Temperature vs Clustering Quality ===")
temp_values = [0.1, 0.5, 1.0, 5.0]
ari_by_temp = []
sil_by_temp = []

for temp in temp_values:
    exp_config = SoftClusteringConfig(n_clusters=N_CLUSTERS, n_features=N_GENES, temperature=temp)
    exp_clusterer = SoftKMeansClustering(exp_config, rngs=nnx.Rngs(20))
    exp_optimizer = nnx.Optimizer(exp_clusterer, optax.adam(0.05), wrt=nnx.Param)

    # Quick training
    for _ in range(30):
        train_step(exp_clusterer, exp_optimizer, data)

    exp_result, _, _ = exp_clusterer.apply(data, {}, None)
    exp_pred = exp_result["cluster_labels"]
    exp_ari = adjusted_rand_index(true_labels, exp_pred)
    exp_sil = silhouette_score(embeddings, exp_pred)
    exp_div = shannon_loss(exp_result["cluster_assignments"])
    ari_by_temp.append(float(exp_ari))
    sil_by_temp.append(float(exp_sil))
    print(
        f"  T={temp:4.1f}: ARI={float(exp_ari):.4f}, "
        f"Silhouette={float(exp_sil):.4f}, "
        f"Shannon={float(exp_div):.4f}"
    )

# %%
# Figure 3: ARI and Silhouette vs temperature
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(temp_values, ari_by_temp, "o-", label="ARI", color="tab:blue", linewidth=2, markersize=7)
ax.plot(temp_values, sil_by_temp, "s--", label="Silhouette", color="tab:orange",
        linewidth=2, markersize=7)
ax.set_xlabel("Temperature")
ax.set_ylabel("Score")
ax.set_title("ARI and Silhouette vs Temperature")
ax.legend()
plt.tight_layout()
plt.savefig("docs/assets/examples/ecosystem/calibrax_temperature.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# This example demonstrated the training-vs-evaluation metric split:
# - **Training**: `DifferentiableAUROC`, `ClusteringCompactnessLoss`,
#   `ShannonDiversityLoss` -- all provide gradients for optimization
# - **Evaluation**: `ExactAUROC`, `adjusted_rand_index`, `normalized_mutual_information_clustering`,
#   `silhouette_score` -- exact metrics from calibrax for reporting
#
# The key principle: train with differentiable surrogates, evaluate
# with exact metrics. Both are JIT-compatible.
#
# ## Next Steps
#
# - [scVI Benchmark](scvi_benchmark.py) -- full VAE training with calibrax
#   evaluation
# - [Single-Cell Pipeline](../pipelines/singlecell_pipeline.py) -- chaining
#   operators end-to-end
