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
# # Doublet Detection with Scrublet and Solo
#
# **Duration:** 15 minutes | **Level:** Intermediate
# | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Understand how doublets arise in scRNA-seq and why detection matters
# 2. Apply DifferentiableDoubletScorer (Bayesian k-NN scoring, Scrublet-style)
# 3. Apply DifferentiableSoloDetector (VAE latent-space classifier, Solo-style)
# 4. Compare doublet scores between known singlets and doublets
#
# ## Prerequisites
#
# - DiffBio installed (see setup instructions)
# - Basic understanding of droplet-based single-cell sequencing artifacts
#
# ```bash
# source ./activate.sh
# uv run python examples/singlecell/doublet_detection.py
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
# ## 1. Create Synthetic Data with Known Doublets
#
# Generate 80 singlet cells with distinct type signatures, then create
# 20 known doublets by averaging pairs of random singlets. This provides
# ground truth for evaluating detection performance.

# %%
# Generate singlet profiles: 80 cells, 50 genes
n_singlets = 80
n_doublets = 20
n_genes = 50

key = jax.random.key(42)
k1, k2, k3, k4 = jax.random.split(key, 4)

# Two cell types with different expression patterns
singlets_type_a = jax.random.poisson(k1, jnp.ones((40, n_genes)) * 3.0)
singlets_type_b = jax.random.poisson(k2, jnp.ones((40, n_genes)) * 3.0)

# Add type-specific marker signal
singlets_type_a = singlets_type_a.at[:, :15].add(8.0)
singlets_type_b = singlets_type_b.at[:, 25:40].add(8.0)

singlet_counts = jnp.concatenate([singlets_type_a, singlets_type_b], axis=0).astype(jnp.float32)

# Create doublets by averaging random pairs from different types
idx_a = jax.random.randint(k3, (n_doublets,), 0, 40)
idx_b = jax.random.randint(k4, (n_doublets,), 40, 80)
doublet_counts = (singlet_counts[idx_a] + singlet_counts[idx_b]).astype(jnp.float32)

# Combine into full dataset
counts = jnp.concatenate([singlet_counts, doublet_counts], axis=0)
n_total = counts.shape[0]

# Ground truth labels: 0 = singlet, 1 = doublet
true_labels = jnp.concatenate([
    jnp.zeros(n_singlets),
    jnp.ones(n_doublets),
])

print(f"Total cells: {n_total} ({n_singlets} singlets + {n_doublets} doublets)")
print(f"Counts shape: {counts.shape}")
print(
    f"Mean expression - singlets: {singlet_counts.mean():.2f},"
    f" doublets: {doublet_counts.mean():.2f}"
)

# %% [markdown]
# ## 2. DoubletScorer (Scrublet-Style Bayesian k-NN)
#
# The DifferentiableDoubletScorer generates synthetic doublets from random
# cell pairs, embeds everything into PCA space, and scores each real cell
# by how many synthetic doublets appear in its k-nearest neighborhood.
# Higher scores indicate likely doublets.

# %%
from diffbio.operators.singlecell import (
    DifferentiableDoubletScorer,
    DoubletScorerConfig,
)

config_scrub = DoubletScorerConfig(
    n_neighbors=15,
    expected_doublet_rate=0.06,
    sim_doublet_ratio=2.0,
    n_pca_components=20,
    n_genes=n_genes,
    threshold_temperature=10.0,
    stochastic=True,
    stream_name="sample",
)
scorer = DifferentiableDoubletScorer(config_scrub, rngs=nnx.Rngs(0))
print(f"DoubletScorer created: {type(scorer).__name__}")

# %%
# Generate random params and run scorer
rng_scrub = jax.random.key(10)
random_params = scorer.generate_random_params(rng_scrub, {"counts": counts.shape})

data_scrub = {"counts": counts}
result_scrub, state_scrub, meta_scrub = scorer.apply(
    data_scrub, {}, None, random_params=random_params,
)

scores_scrub = result_scrub["doublet_scores"]
preds_scrub = result_scrub["predicted_doublets"]

print(f"Doublet scores shape: {scores_scrub.shape}")
print(f"Predicted doublets shape: {preds_scrub.shape}")

# Compare scores between known singlets and doublets
singlet_scores = scores_scrub[:n_singlets]
doublet_scores_known = scores_scrub[n_singlets:]

print("\nDoublet score statistics:")
print(f"  Singlets - mean: {float(singlet_scores.mean()):.4f}, "
      f"std: {float(singlet_scores.std()):.4f}")
print(f"  Doublets - mean: {float(doublet_scores_known.mean()):.4f}, "
      f"std: {float(doublet_scores_known.std()):.4f}")

# %%
# Figure 1: DoubletScorer score distributions
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(
    singlet_scores.tolist(), bins=20, alpha=0.6, color="blue", label="Singlets",
)
ax.hist(
    doublet_scores_known.tolist(), bins=20, alpha=0.6, color="red", label="Doublets",
)
ax.set_xlabel("Doublet Score")
ax.set_ylabel("Count")
ax.set_title("DoubletScorer: Singlet vs Doublet Score Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(
    "docs/assets/examples/singlecell/doublet_scorer_histogram.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## 3. SoloDetector (VAE Latent-Space Classifier)
#
# The DifferentiableSoloDetector encodes cells through a VAE, generates
# synthetic doublets, then trains a binary classifier in latent space to
# distinguish singlets from doublets. The classifier probability serves
# as the doublet score.

# %%
from diffbio.operators.singlecell import (
    DifferentiableSoloDetector,
    SoloDetectorConfig,
)

config_solo = SoloDetectorConfig(
    n_genes=n_genes,
    latent_dim=8,
    hidden_dims=[32, 16],
    classifier_hidden_dim=16,
    sim_doublet_ratio=2.0,
    stochastic=True,
    stream_name="sample",
)
detector = DifferentiableSoloDetector(config_solo, rngs=nnx.Rngs(1))
print(f"SoloDetector created: {type(detector).__name__}")

# %%
# Generate random params and run Solo detector
rng_solo = jax.random.key(20)
random_params_solo = detector.generate_random_params(rng_solo, {"counts": counts.shape})

data_solo = {"counts": counts}
result_solo, state_solo, meta_solo = detector.apply(
    data_solo, {}, None, random_params=random_params_solo,
)

probs_solo = result_solo["doublet_probabilities"]
labels_solo = result_solo["doublet_labels"]
latent_solo = result_solo["latent"]

print(f"Doublet probabilities shape: {probs_solo.shape}")
print(f"Latent shape: {latent_solo.shape}")

# Compare probabilities between known singlets and doublets
singlet_probs = probs_solo[:n_singlets]
doublet_probs_known = probs_solo[n_singlets:]

print("\nSolo doublet probability statistics:")
print(f"  Singlets - mean: {float(singlet_probs.mean()):.4f}, "
      f"std: {float(singlet_probs.std()):.4f}")
print(f"  Doublets - mean: {float(doublet_probs_known.mean()):.4f}, "
      f"std: {float(doublet_probs_known.std()):.4f}")

# %%
# Figure 2: SoloDetector score distributions
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(
    singlet_probs.tolist(), bins=20, alpha=0.6, color="blue", label="Singlets",
)
ax.hist(
    doublet_probs_known.tolist(), bins=20, alpha=0.6, color="red", label="Doublets",
)
ax.set_xlabel("Doublet Probability")
ax.set_ylabel("Count")
ax.set_title("SoloDetector: Singlet vs Doublet Score Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(
    "docs/assets/examples/singlecell/doublet_solo_histogram.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## 4. Compare Both Methods
#
# Both methods aim to assign higher scores to doublets than singlets.
# With untrained models, the discrimination may be weak, but the
# differentiable formulation allows training to improve these scores.

# %%
print("=== Method Comparison ===\n")
print(f"{'Method':<15} {'Singlet Mean':>14} {'Doublet Mean':>14} {'Gap':>8}")
print("-" * 53)

# Scrublet scores
s_mean = float(singlet_scores.mean())
d_mean = float(doublet_scores_known.mean())
print(f"{'Scrublet':<15} {s_mean:>14.4f} {d_mean:>14.4f} {d_mean - s_mean:>8.4f}")

# Solo probabilities
s_mean_solo = float(singlet_probs.mean())
d_mean_solo = float(doublet_probs_known.mean())
print(f"{'Solo':<15} {s_mean_solo:>14.4f} {d_mean_solo:>14.4f} "
      f"{d_mean_solo - s_mean_solo:>8.4f}")

# %% [markdown]
# ## 5. Verify Differentiability
#
# Both doublet detection operators are fully differentiable, enabling
# joint optimization with downstream analysis tasks.

# %%
# Gradient check for DoubletScorer
print("=== Gradient Flow Verification ===\n")


def loss_fn_scrub(input_data):
    """Scalar loss from Scrublet-style doublet scores."""
    res, _, _ = scorer.apply(input_data, {}, None, random_params=random_params)
    return res["doublet_scores"].sum()


grad_scrub = jax.grad(loss_fn_scrub)(data_scrub)
print("DoubletScorer:")
print(f"  Gradient shape: {grad_scrub['counts'].shape}")
print(f"  Non-zero: {bool(jnp.any(grad_scrub['counts'] != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(grad_scrub['counts'])))}")

# %%
# Gradient check for SoloDetector


def loss_fn_solo(input_data):
    """Scalar loss from Solo-style doublet probabilities."""
    res, _, _ = detector.apply(input_data, {}, None, random_params=random_params_solo)
    return res["doublet_probabilities"].sum()


grad_solo = jax.grad(loss_fn_solo)(data_solo)
print("SoloDetector:")
print(f"  Gradient shape: {grad_solo['counts'].shape}")
print(f"  Non-zero: {bool(jnp.any(grad_solo['counts'] != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(grad_solo['counts'])))}")

# %% [markdown]
# ## 6. JIT Compilation
#
# Both operators support JIT compilation for efficient batched execution.

# %%
# JIT DoubletScorer
print("=== JIT Compilation ===\n")

jit_scrub = jax.jit(lambda d: scorer.apply(d, {}, None, random_params=random_params))
result_jit_scrub, _, _ = jit_scrub(data_scrub)
match_scrub = jnp.allclose(
    result_scrub["doublet_scores"],
    result_jit_scrub["doublet_scores"],
    atol=1e-5,
)
print(f"DoubletScorer JIT matches eager: {bool(match_scrub)}")

# %%
# JIT SoloDetector
jit_solo = jax.jit(lambda d: detector.apply(d, {}, None, random_params=random_params_solo))
result_jit_solo, _, _ = jit_solo(data_solo)
match_solo = jnp.allclose(
    result_solo["doublet_probabilities"],
    result_jit_solo["doublet_probabilities"],
    atol=1e-5,
)
print(f"SoloDetector JIT matches eager: {bool(match_solo)}")

# %% [markdown]
# ## 7. Experiments
#
# ### Vary the synthetic doublet ratio
#
# The `sim_doublet_ratio` controls how many synthetic doublets are generated
# relative to real cells. Higher ratios give the k-NN scoring more reference
# doublets to compare against.

# %%
print("=== Experiment: Synthetic Doublet Ratio Effect ===\n")

ratio_values = [0.5, 1.0, 2.0, 4.0]
score_gaps = []

for ratio in ratio_values:
    cfg = DoubletScorerConfig(
        n_neighbors=15,
        expected_doublet_rate=0.06,
        sim_doublet_ratio=ratio,
        n_pca_components=20,
        n_genes=n_genes,
        stochastic=True,
    )
    sc = DifferentiableDoubletScorer(cfg, rngs=nnx.Rngs(0))
    rp = sc.generate_random_params(jax.random.key(10), {"counts": counts.shape})
    res, _, _ = sc.apply({"counts": counts}, {}, None, random_params=rp)
    scores = res["doublet_scores"]

    s_mean = float(scores[:n_singlets].mean())
    d_mean = float(scores[n_singlets:].mean())
    score_gaps.append(d_mean - s_mean)
    print(f"  ratio={ratio:.1f} -> singlet mean: {s_mean:.4f}, "
          f"doublet mean: {d_mean:.4f}, gap: {d_mean - s_mean:.4f}")

# %%
# Figure 3: Score gap vs synthetic doublet ratio
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ratio_values, score_gaps, "o-", color="#4C72B0", linewidth=2, markersize=7)
ax.set_xlabel("Synthetic Doublet Ratio")
ax.set_ylabel("Score Gap (Doublet Mean - Singlet Mean)")
ax.set_title("Score Gap vs Synthetic Doublet Ratio")
ax.set_xticks(ratio_values)
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/doublet_ratio_sweep.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Vary the number of PCA components
#
# The PCA embedding dimension affects how well the k-NN scoring captures
# the difference between singlets and doublets.

# %%
print("\n=== Experiment: PCA Components Effect ===\n")

for n_pca in [5, 10, 20, 30]:
    cfg = DoubletScorerConfig(
        n_neighbors=15,
        n_pca_components=n_pca,
        n_genes=n_genes,
        stochastic=True,
    )
    sc = DifferentiableDoubletScorer(cfg, rngs=nnx.Rngs(0))
    rp = sc.generate_random_params(jax.random.key(10), {"counts": counts.shape})
    res, _, _ = sc.apply({"counts": counts}, {}, None, random_params=rp)
    scores = res["doublet_scores"]

    s_mean = float(scores[:n_singlets].mean())
    d_mean = float(scores[n_singlets:].mean())
    print(f"  n_pca={n_pca:>2} -> singlet mean: {s_mean:.4f}, "
          f"doublet mean: {d_mean:.4f}, gap: {d_mean - s_mean:.4f}")

# %% [markdown]
# ## Summary
#
# Two doublet detection strategies were demonstrated:
#
# - **DoubletScorer (Scrublet-style)**: Bayesian k-NN scoring in PCA space --
#   lightweight, no training required, uses synthetic doublet generation
# - **SoloDetector (Solo-style)**: VAE with latent-space classifier --
#   learnable, benefits from training via `compute_solo_loss()`
#
# Both operators are differentiable and JIT-compatible, enabling integration
# into end-to-end optimizable single-cell analysis pipelines.
#
# ## Next Steps
#
# - Train SoloDetector with `compute_solo_loss()` for improved detection
# - Chain doublet detection with quality filtering and clustering
# - Explore the API reference for DoubletScorerConfig options
