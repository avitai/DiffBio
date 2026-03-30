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
# # Spatial Transcriptomics: Domain Identification and Slice Alignment
#
# **Duration:** 25 minutes | **Level:** Advanced
# | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Understand spatial domain identification with STAGATE-inspired GATv2 autoencoders
# 2. Apply DifferentiableSpatialDomain on a synthetic spatial grid with known domains
# 3. Align two synthetic slices using DifferentiablePASTEAlignment (Sinkhorn OT)
# 4. Verify that spatial domains are coherent and gradients flow end-to-end
#
# ## Prerequisites
#
# - DiffBio installed (see setup instructions)
# - Understanding of spatial transcriptomics concepts (tissue slices, spot coordinates)
# - Familiarity with graph attention networks and optimal transport (conceptual)
#
# ```bash
# source ./activate.sh
# uv run python examples/singlecell/spatial_analysis.py
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
# ## 1. Create Synthetic Spatial Grid
#
# Generate a 10x10 spatial grid (100 spots) divided into 3 spatial domains.
# Each domain has a distinct gene expression signature, and spots are
# assigned to domains based on spatial location. This mimics tissue
# sections with distinct compartments (e.g., cortex, medulla, interface).

# %%
# Build 10x10 spatial grid
grid_size = 10
n_spots = grid_size * grid_size
n_genes = 30
n_domains_true = 3

key = jax.random.key(42)
k1, k2, k3 = jax.random.split(key, 3)

# Spatial coordinates: regular grid
x_coords = jnp.repeat(jnp.arange(grid_size, dtype=jnp.float32), grid_size)
y_coords = jnp.tile(jnp.arange(grid_size, dtype=jnp.float32), grid_size)
spatial_coords = jnp.stack([x_coords, y_coords], axis=1)

print(f"Spatial coords shape: {spatial_coords.shape}")
print(f"Grid extent: x=[{float(x_coords.min())}, {float(x_coords.max())}], "
      f"y=[{float(y_coords.min())}, {float(y_coords.max())}]")

# %%
# Assign domains based on spatial location:
# Domain 0: left third (x < 3.5), Domain 1: middle (3.5 <= x < 7),
# Domain 2: right third (x >= 7)
domain_labels = jnp.where(
    x_coords < 3.5,
    0,
    jnp.where(x_coords < 7.0, 1, 2),
).astype(jnp.int32)

print(f"Domain distribution: {jnp.bincount(domain_labels, length=n_domains_true)}")

# %%
# Generate expression profiles with domain-specific signatures
base_expression = jax.random.poisson(k1, jnp.ones((n_spots, n_genes)) * 2.0)

# Domain signatures: each domain upregulates a distinct gene block
domain_signal = jnp.zeros((n_spots, n_genes))
for d in range(n_domains_true):
    mask = (domain_labels == d).astype(jnp.float32)
    gene_start = d * 10
    gene_end = gene_start + 10
    signal = jnp.zeros((n_spots, n_genes))
    signal = signal.at[:, gene_start:gene_end].set(8.0)
    domain_signal = domain_signal + signal * mask[:, None]

counts = (base_expression + domain_signal).astype(jnp.float32)
# Add small noise
counts = counts + jax.random.uniform(k2, counts.shape) * 0.1

print(f"Expression shape: {counts.shape}")
print("Mean expression per domain (first 5 genes):")
for d in range(n_domains_true):
    d_mask = domain_labels == d
    mean_expr = jnp.where(d_mask[:, None], counts, 0.0).sum(axis=0) / d_mask.sum()
    print(f"  Domain {d}: {mean_expr[:5]}")

# %% [markdown]
# ## 2. Spatial Domain Identification (STAGATE-inspired)
#
# DifferentiableSpatialDomain builds a spatial k-NN graph from coordinates,
# applies dual-graph GATv2 attention (full + pruned mutual k-NN), and
# computes soft domain assignments via learned prototypes.

# %%
from diffbio.operators.singlecell import (
    DifferentiableSpatialDomain,
    SpatialDomainConfig,
)

config_domain = SpatialDomainConfig(
    n_genes=n_genes,
    hidden_dim=32,
    num_heads=4,
    n_domains=n_domains_true,
    alpha=0.8,
    n_neighbors=8,
)
domain_op = DifferentiableSpatialDomain(config_domain, rngs=nnx.Rngs(0))
print(f"SpatialDomain operator created: {type(domain_op).__name__}")
print(f"  hidden_dim={config_domain.hidden_dim}, "
      f"num_heads={config_domain.num_heads}, "
      f"n_domains={config_domain.n_domains}")

# %%
# Run spatial domain identification
data_domain = {
    "counts": counts,
    "spatial_coords": spatial_coords,
}
result_domain, state_d, meta_d = domain_op.apply(data_domain, {}, None)

assignments = result_domain["domain_assignments"]
embeddings = result_domain["spatial_embeddings"]

print(f"Domain assignments shape: {assignments.shape}")
print(f"Spatial embeddings shape: {embeddings.shape}")

# Predicted domains (argmax of soft assignments)
predicted_domains = jnp.argmax(assignments, axis=-1)
print(f"Predicted domain distribution: "
      f"{jnp.bincount(predicted_domains, length=n_domains_true)}")

# %%
# Verify spatial coherence: check that neighboring spots tend to share domains
# For each spot, count how many of its 4-nearest spatial neighbors have the same domain

from diffbio.core import compute_pairwise_distances

dists = compute_pairwise_distances(spatial_coords)
dists = dists + jnp.eye(n_spots) * 1e10
nn_4 = jnp.argsort(dists, axis=1)[:, :4]
nn_domains = predicted_domains[nn_4]

# Fraction of neighbors with same domain
same_domain = (nn_domains == predicted_domains[:, None]).astype(jnp.float32)
coherence = float(same_domain.mean())
print(f"\nSpatial coherence (fraction of neighbors with same domain): {coherence:.4f}")
print(f"  Random baseline (1/{n_domains_true}): {1.0 / n_domains_true:.4f}")

# Check assignment confidence
mean_confidence = float(assignments.max(axis=1).mean())
print(f"Mean assignment confidence: {mean_confidence:.4f}")

# %%
# Figure 1: Spatial domain assignments
fig, ax = plt.subplots(figsize=(6, 5))
scatter = ax.scatter(
    spatial_coords[:, 0],
    spatial_coords[:, 1],
    c=predicted_domains,
    cmap="Set2",
    s=60,
    edgecolors="k",
    linewidths=0.3,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Spatial Domain Assignments")
plt.colorbar(scatter, ax=ax, label="Domain")
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/spatial_domains.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. PASTE-Style Slice Alignment
#
# DifferentiablePASTEAlignment aligns two spatial transcriptomics slices
# by computing a fused Gromov-Wasserstein cost that balances expression
# dissimilarity with spatial structure preservation, then solves via
# differentiable Sinkhorn optimal transport.

# %%
# Create two synthetic slices with shared biology but different coordinates
# Slice 1: the grid data from above
# Slice 2: same expression patterns, spatially shifted and slightly rotated

k4, k5 = jax.random.split(k3, 2)

# Slice 2 coordinates: shift by (2, 1) and add small noise
coords_slice2 = spatial_coords + jnp.array([2.0, 1.0])
coords_slice2 = coords_slice2 + jax.random.normal(k4, coords_slice2.shape) * 0.1

# Slice 2 expression: same domains, small independent noise
counts_slice2 = counts + jax.random.normal(k5, counts.shape) * 0.5

print(f"Slice 1: {counts.shape[0]} spots, coords range: "
      f"[{float(spatial_coords.min()):.1f}, {float(spatial_coords.max()):.1f}]")
print(f"Slice 2: {counts_slice2.shape[0]} spots, coords range: "
      f"[{float(coords_slice2.min()):.1f}, {float(coords_slice2.max()):.1f}]")

# %%
from diffbio.operators.singlecell import (
    DifferentiablePASTEAlignment,
    PASTEAlignmentConfig,
)

config_paste = PASTEAlignmentConfig(
    alpha=0.1,
    sinkhorn_epsilon=0.1,
    sinkhorn_iters=50,
)
paste_op = DifferentiablePASTEAlignment(config_paste, rngs=nnx.Rngs(1))
print(f"PASTE alignment operator created: {type(paste_op).__name__}")

# %%
# Run PASTE alignment
data_paste = {
    "slice1_counts": counts,
    "slice2_counts": counts_slice2,
    "slice1_coords": spatial_coords,
    "slice2_coords": coords_slice2,
}
result_paste, state_p, meta_p = paste_op.apply(data_paste, {}, None)

transport_plan = result_paste["transport_plan"]
aligned_coords = result_paste["aligned_coords"]

print(f"Transport plan shape: {transport_plan.shape}")
print(f"Aligned coords shape: {aligned_coords.shape}")
print(f"Transport plan sum: {float(transport_plan.sum()):.4f} (should be ~1.0)")

# %%
# Evaluate alignment quality: compare aligned coords to original slice 1 coords
coord_error = jnp.sqrt(jnp.sum((aligned_coords - spatial_coords) ** 2, axis=1))
print(f"Alignment error (mean Euclidean distance): {float(coord_error.mean()):.4f}")
print(f"Alignment error (max): {float(coord_error.max()):.4f}")

# Check that the transport plan is sparse (concentrates on matching spots)
plan_entropy = -jnp.sum(
    transport_plan * jnp.log(transport_plan + 1e-10),
)
max_entropy = -jnp.log(jnp.array(1.0 / (n_spots * n_spots))) * n_spots * n_spots
print(f"Transport plan entropy: {float(plan_entropy):.4f} "
      f"(max uniform: {float(max_entropy):.4f})")

# %%
# Figure 2: PASTE transport plan heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(transport_plan, cmap="viridis", aspect="auto")
ax.set_xlabel("Slice 2 spots")
ax.set_ylabel("Slice 1 spots")
ax.set_title("PASTE Transport Plan")
plt.colorbar(im, ax=ax, label="Transport weight")
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/spatial_transport.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Verify Differentiability
#
# Both spatial operators are end-to-end differentiable, enabling gradient-based
# optimization of domain assignments and slice alignment.

# %%
print("=== Gradient Flow Verification ===\n")

# SpatialDomain gradient


def loss_fn_domain(input_data):
    """Scalar loss from spatial domain assignments."""
    res, _, _ = domain_op.apply(input_data, {}, None)
    return res["domain_assignments"].sum()


grad_domain = jax.grad(loss_fn_domain)(data_domain)
print("SpatialDomain:")
print(f"  Gradient shape (counts): {grad_domain['counts'].shape}")
print(f"  Non-zero: {bool(jnp.any(grad_domain['counts'] != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(grad_domain['counts'])))}")

# %%
# PASTE alignment gradient


def loss_fn_paste(input_data):
    """Scalar loss from PASTE alignment transport plan."""
    res, _, _ = paste_op.apply(input_data, {}, None)
    return res["transport_plan"].sum()


grad_paste = jax.grad(loss_fn_paste)(data_paste)
print("PASTE Alignment:")
print(f"  Gradient shape (slice1_counts): {grad_paste['slice1_counts'].shape}")
print(f"  Non-zero: {bool(jnp.any(grad_paste['slice1_counts'] != 0))}")
print(f"  Finite: {bool(jnp.all(jnp.isfinite(grad_paste['slice1_counts'])))}")

# %% [markdown]
# ## 5. JIT Compilation
#
# Both spatial operators support JIT compilation. The graph construction
# and Sinkhorn iterations are fully traceable by JAX.

# %%
print("=== JIT Compilation ===\n")

# SpatialDomain JIT
jit_domain = jax.jit(lambda d: domain_op.apply(d, {}, None))
result_jit_d, _, _ = jit_domain(data_domain)
match_d = jnp.allclose(
    result_domain["domain_assignments"],
    result_jit_d["domain_assignments"],
    atol=1e-5,
)
print(f"SpatialDomain JIT matches eager: {bool(match_d)}")

# %%
# PASTE JIT
jit_paste = jax.jit(lambda d: paste_op.apply(d, {}, None))
result_jit_p, _, _ = jit_paste(data_paste)
match_p = jnp.allclose(
    result_paste["transport_plan"],
    result_jit_p["transport_plan"],
    atol=1e-5,
)
print(f"PASTE JIT matches eager: {bool(match_p)}")

# %% [markdown]
# ## 6. Experiments
#
# ### Vary the alpha parameter for dual-graph attention
#
# Alpha controls the balance between full k-NN and pruned (mutual k-NN)
# graphs. At alpha=0, only the full graph is used; at alpha=1, only the
# pruned graph contributes.

# %%
print("=== Experiment: STAGATE Alpha Effect ===\n")

alpha_values = [0.0, 0.2, 0.5, 0.8, 1.0]
coherence_values = []

for alpha_val in alpha_values:
    cfg = SpatialDomainConfig(
        n_genes=n_genes,
        hidden_dim=32,
        num_heads=4,
        n_domains=n_domains_true,
        alpha=alpha_val,
        n_neighbors=8,
    )
    op = DifferentiableSpatialDomain(cfg, rngs=nnx.Rngs(0))
    res, _, _ = op.apply(data_domain, {}, None)
    pred = jnp.argmax(res["domain_assignments"], axis=-1)

    # Compute coherence
    nn_doms = pred[nn_4]
    coh = float((nn_doms == pred[:, None]).astype(jnp.float32).mean())
    conf = float(res["domain_assignments"].max(axis=1).mean())
    coherence_values.append(coh)
    print(f"  alpha={alpha_val:.1f} -> coherence: {coh:.4f}, confidence: {conf:.4f}")

# %%
# Figure 3: Spatial coherence vs alpha parameter
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(alpha_values, coherence_values, "o-", color="tab:blue", linewidth=2, markersize=7)
ax.axhline(1.0 / n_domains_true, color="gray", linestyle="--", label="Random baseline")
ax.set_xlabel("Alpha")
ax.set_ylabel("Spatial coherence")
ax.set_title("Spatial Coherence vs Alpha")
ax.legend()
plt.tight_layout()
plt.savefig("docs/assets/examples/singlecell/spatial_alpha_sweep.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Vary the PASTE alpha (expression vs spatial weight)
#
# The PASTE alpha balances expression dissimilarity (linear term) with
# spatial Gromov-Wasserstein cost (quadratic term).

# %%
print("\n=== Experiment: PASTE Alpha Effect ===\n")

for paste_alpha in [0.0, 0.1, 0.3, 0.5, 0.9]:
    cfg = PASTEAlignmentConfig(
        alpha=paste_alpha,
        sinkhorn_epsilon=0.1,
        sinkhorn_iters=50,
    )
    op = DifferentiablePASTEAlignment(cfg, rngs=nnx.Rngs(1))
    res, _, _ = op.apply(data_paste, {}, None)
    aligned = res["aligned_coords"]
    err = float(jnp.sqrt(jnp.sum((aligned - spatial_coords) ** 2, axis=1)).mean())
    plan_max = float(res["transport_plan"].max())
    print(f"  alpha={paste_alpha:.1f} -> alignment error: {err:.4f}, "
          f"plan max: {plan_max:.6f}")

# %% [markdown]
# ## Summary
#
# Two spatial transcriptomics operators were demonstrated:
#
# - **DifferentiableSpatialDomain (STAGATE-inspired)**: GATv2 autoencoder with
#   dual-graph attention for spatial domain identification. Learned prototypes
#   provide soft domain assignments that respect spatial locality.
#
# - **DifferentiablePASTEAlignment**: Fused Gromov-Wasserstein optimal transport
#   for aligning spatial transcriptomics slices. Sinkhorn iterations produce
#   a differentiable transport plan that balances expression and spatial structure.
#
# Both operators are fully differentiable and JIT-compatible, enabling
# gradient-based optimization of spatial analysis workflows.
#
# ## Next Steps
#
# - Train the spatial domain operator with autoencoder reconstruction loss
# - Apply PASTE alignment to multi-slice tissue reconstruction
# - Combine spatial domain identification with differential expression analysis
