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
# # Perturbation Experiment Data Loading
#
# **Duration:** 15 minutes | **Level:** Intermediate
# | **Device:** CPU-compatible
#
# ## Learning Objectives
#
# By the end of this example, you will:
# 1. Load single-cell perturbation data using `PerturbationAnnDataSource`
# 2. Configure zero-shot and few-shot splitting strategies
# 3. Map perturbed cells to control cells using batch and random strategies
# 4. Build sentence-based batch samplers for training
# 5. Compose the full workflow via `PerturbationPipeline`
#
# ## Prerequisites
#
# - DiffBio installed with anndata and h5py
# - Familiarity with single-cell perturbation experiments (e.g., Perturb-seq)
# - Understanding of AnnData (.h5ad) format
#
# ```bash
# source ./activate.sh
# uv run python examples/perturbation/perturbation_data_loading.py
# ```
#
# ---

# %% [markdown]
# ## Pipeline Overview
#
# This example demonstrates DiffBio's perturbation data loading stack,
# which ports the cell-load library's features to JAX/datarax:
#
# ```
# PerturbationAnnDataSource
#   -> loads H5AD with perturbation/cell_type/batch metadata
#   -> integer-encodes categories, builds one-hot maps
#
# OnTargetKnockdownFilter
#   -> 3-stage QC: perturbation-level -> cell-level -> min count
#
# ZeroShotSplitter / FewShotSplitter
#   -> hold out cell types or perturbations for evaluation
#
# RandomControlMapping / BatchControlMapping
#   -> pair perturbed cells with matched controls
#
# PerturbationBatchSampler
#   -> group cells by (cell_type, perturbation) into sentence-based batches
#
# PerturbationPipeline
#   -> composes all of the above into a single setup() call
# ```

# %% [markdown]
# ## Setup

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

print(f"JAX backend: {jax.default_backend()}")

# %% [markdown]
# ## Create Synthetic Perturbation Data
#
# We create a realistic synthetic dataset with 3 cell types, 5 gene
# perturbations plus a non-targeting control, across 2 batches.

# %%
import anndata
import pandas as pd

rng = np.random.default_rng(42)

# Dataset parameters
cell_types = ["Neuron", "Astrocyte", "Microglia"]
perturbations = ["BRCA1", "TP53", "KRAS", "MYC", "EGFR"]
control_label = "non-targeting"
all_perts = [control_label] + perturbations
batches = ["plate_1", "plate_2"]
n_cells_per_group = 50
n_genes = 200

# Build obs metadata
obs_records = []
for ct in cell_types:
    for pert in all_perts:
        for i in range(n_cells_per_group):
            obs_records.append({
                "perturbation": pert,
                "cell_type": ct,
                "batch": batches[i % len(batches)],
            })

n_cells = len(obs_records)
print(f"Total cells: {n_cells}")
print(f"  {len(cell_types)} cell types x {len(all_perts)} perturbations x {n_cells_per_group} cells")

# Count matrix with knockdown signal
counts = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
gene_names = [f"gene_{i}" for i in range(n_genes)]

# Map first 5 genes to perturbation target genes
for idx, pert in enumerate(perturbations):
    gene_names[idx] = pert

# Inject knockdown: reduce target gene expression by 90% in perturbed cells
for cell_idx, rec in enumerate(obs_records):
    pert = rec["perturbation"]
    if pert in perturbations:
        gene_idx = perturbations.index(pert)
        counts[cell_idx, gene_idx] *= 0.1

obs = pd.DataFrame(obs_records)
obs.index = [f"cell_{i}" for i in range(n_cells)]
for col in ["perturbation", "cell_type", "batch"]:
    obs[col] = pd.Categorical(obs[col])

var = pd.DataFrame({"gene_name": gene_names}, index=gene_names)
adata = anndata.AnnData(X=counts, obs=obs, var=var)

# Add a low-dimensional embedding
adata.obsm["X_pca"] = rng.standard_normal((n_cells, 20)).astype(np.float32)

# Write to temp file
import tempfile
from pathlib import Path

tmp_dir = Path(tempfile.mkdtemp())
h5ad_path = tmp_dir / "perturb_seq.h5ad"
adata.write_h5ad(h5ad_path)
print(f"Saved to: {h5ad_path}")

# %% [markdown]
# ## Step 1: Load with PerturbationAnnDataSource
#
# `PerturbationAnnDataSource` extends the base `AnnDataSource` with
# perturbation metadata: integer-encoded categories, control cell mask,
# group codes for efficient batching, and one-hot perturbation embeddings.

# %%
from diffbio.sources.perturbation import (
    PerturbationAnnDataSource,
    PerturbationSourceConfig,
)

config = PerturbationSourceConfig(
    file_path=str(h5ad_path),
    pert_col="perturbation",
    cell_type_col="cell_type",
    batch_col="batch",
    control_pert="non-targeting",
    output_space="all",
)
source = PerturbationAnnDataSource(config)

print(f"Loaded {len(source)} cells")
print(f"Perturbation categories: {list(source.get_pert_categories())}")
print(f"Cell type categories: {list(source.get_cell_type_categories())}")
print(f"Batch categories: {list(source.get_batch_categories())}")
print(f"Control cells: {source.get_control_mask().sum()}")
print(f"Unique groups: {len(np.unique(source.get_group_codes()))}")

# %% [markdown]
# ### Inspect a Single Element
#
# Each element is a dictionary with counts, perturbation metadata,
# and one-hot embeddings.

# %%
elem = source[0]
print("Element keys:")
for key, value in elem.items():
    if hasattr(value, "shape"):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {value}")

# %% [markdown]
# ## Step 2: Control Cell Mapping
#
# Map each perturbed cell to a control cell from the same cell type.
# Two strategies are available:
# - **Random**: pools controls by cell type, samples randomly
# - **Batch**: prefers controls from the same batch, falls back to cell type

# %%
from diffbio.sources.perturbation import (
    ControlMappingConfig,
    RandomControlMapping,
    BatchControlMapping,
)

# Random mapping
random_mapper = RandomControlMapping(
    ControlMappingConfig(n_basal_samples=1, seed=42)
)
random_mapping = random_mapper.build_mapping(source)
print(f"Random mapping shape: {random_mapping.shape}")

# Batch mapping
batch_mapper = BatchControlMapping(
    ControlMappingConfig(strategy="batch", n_basal_samples=1, seed=42)
)
batch_mapping = batch_mapper.build_mapping(source)
print(f"Batch mapping shape: {batch_mapping.shape}")

# Verify controls are from the same cell type
ct_codes = source.get_cell_type_codes()
ctrl_mask = source.get_control_mask()
pert_indices = np.where(~ctrl_mask)[0]
same_ct = sum(
    ct_codes[pert_indices[i]] == ct_codes[random_mapping[i, 0]]
    for i in range(min(100, len(pert_indices)))
)
print(f"Same cell type (random, first 100): {same_ct}/100")

# %% [markdown]
# ## Step 3: Perturbation Batch Sampler
#
# Groups cells by (cell_type, perturbation) into "sentences" of fixed
# size, then combines sentences into batches. This ensures each batch
# contains coherent groups of same-condition cells.

# %%
from diffbio.samplers.perturbation_sampler import (
    PerturbationBatchSampler,
    PerturbationSamplerConfig,
)

sampler = PerturbationBatchSampler(
    PerturbationSamplerConfig(sentence_size=25, sentences_per_batch=4, seed=42),
    source.get_group_codes(),
)

batches = list(sampler)
print(f"Number of batches: {len(batches)}")
print(f"Batch sizes: {[len(b) for b in batches[:5]]}...")
print(f"Total cells sampled: {sum(len(b) for b in batches)}")

# Verify within-sentence homogeneity
group_codes = source.get_group_codes()
batch0 = batches[0]
for start in range(0, min(len(batch0), 100), 25):
    sentence = batch0[start:start + 25]
    groups = {group_codes[i] for i in sentence}
    print(f"  Sentence [{start}:{start+25}] groups: {groups} (homogeneous: {len(groups) == 1})")

# %% [markdown]
# ## Step 4: Splitting Strategies
#
# ### Zero-Shot: Hold out entire cell types

# %%
from diffbio.splitters.perturbation import (
    ZeroShotSplitter,
    ZeroShotSplitterConfig,
    FewShotSplitter,
    FewShotSplitterConfig,
)

zs_splitter = ZeroShotSplitter(
    ZeroShotSplitterConfig(
        held_out_cell_types=("Microglia",),
        seed=42,
    ),
    rngs=nnx.Rngs(42),
)
zs_split = zs_splitter.split(source)

print("Zero-shot split:")
print(f"  Train: {len(zs_split.train_indices)} cells")
print(f"  Valid: {len(zs_split.valid_indices)} cells")
print(f"  Test:  {len(zs_split.test_indices)} cells")

test_types = {source[int(i)]["cell_type_name"] for i in zs_split.test_indices[:50]}
print(f"  Test cell types: {test_types}")

# %% [markdown]
# ### Few-Shot: Hold out specific perturbations

# %%
fs_splitter = FewShotSplitter(
    FewShotSplitterConfig(
        held_out_perturbations=("BRCA1", "TP53"),
        control_pert="non-targeting",
        seed=42,
    ),
    rngs=nnx.Rngs(42),
)
fs_split = fs_splitter.split(source)

print("Few-shot split:")
print(f"  Train: {len(fs_split.train_indices)} cells")
print(f"  Valid: {len(fs_split.valid_indices)} cells")
print(f"  Test:  {len(fs_split.test_indices)} cells")

test_perts = {source[int(i)]["pert_name"] for i in fs_split.test_indices}
print(f"  Test perturbations: {test_perts}")

# %% [markdown]
# ## Step 5: Knockdown QC Filter
#
# The three-stage filter removes:
# 1. Perturbations with insufficient average knockdown
# 2. Individual cells with weak knockdown
# 3. Perturbations with too few remaining cells

# %%
from diffbio.operators.singlecell.knockdown_filter import (
    OnTargetKnockdownFilter,
    KnockdownFilterConfig,
)

filt = OnTargetKnockdownFilter(
    KnockdownFilterConfig(
        residual_expression=0.30,
        cell_residual_expression=0.50,
        min_cells=10,
        var_gene_col="gene_name",
    )
)
mask = filt.process(source)

print(f"Cells passing QC: {mask.sum()} / {len(mask)}")
print(f"Controls preserved: {mask[ctrl_mask].sum()} / {ctrl_mask.sum()}")

# %% [markdown]
# ## Visualize Split and Mapping Structure

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Cell type distribution per split
for ax, (name, indices) in zip(
    axes[:2],
    [("Zero-shot", zs_split), ("Few-shot", fs_split)],
):
    train_cts = [source[int(i)]["cell_type_name"] for i in indices.train_indices[:200]]
    test_cts = [source[int(i)]["cell_type_name"] for i in indices.test_indices[:200]]
    ct_names = sorted(set(train_cts + test_cts))
    train_counts = [train_cts.count(ct) for ct in ct_names]
    test_counts = [test_cts.count(ct) for ct in ct_names]
    x = np.arange(len(ct_names))
    ax.bar(x - 0.2, train_counts, 0.4, label="Train", color="#4c72b0")
    ax.bar(x + 0.2, test_counts, 0.4, label="Test", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(ct_names, rotation=30, ha="right")
    ax.set_ylabel("Cell count (sample)")
    ax.set_title(f"{name} Split")
    ax.legend()

# 3. Group code distribution
unique_codes, code_counts = np.unique(source.get_group_codes(), return_counts=True)
axes[2].bar(range(len(unique_codes)), code_counts, color="#55a868")
axes[2].set_xlabel("Group code")
axes[2].set_ylabel("Cell count")
axes[2].set_title("Cells per (cell_type, pert) Group")

plt.tight_layout()
plt.savefig(
    "docs/assets/examples/perturbation/split_overview.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## Step 6: Full Pipeline via PerturbationPipeline
#
# `PerturbationPipeline` composes all steps into a single `setup()` call.

# %%
from diffbio.pipelines.perturbation import (
    PerturbationPipeline,
    PerturbationPipelineConfig,
)

pipeline = PerturbationPipeline(
    PerturbationPipelineConfig(
        split_mode="fewshot",
        held_out_perturbations=("BRCA1", "TP53"),
        output_space="all",
        mapping_strategy="random",
        n_basal_samples=1,
        sentence_size=25,
        sentences_per_batch=4,
        enable_knockdown_filter=True,
        residual_expression=0.30,
        cell_residual_expression=0.50,
        min_cells=10,
        var_gene_col="gene_name",
        seed=42,
    )
)
result = pipeline.setup([h5ad_path])

print("Pipeline result:")
print(f"  Train: {len(result.train_indices)} cells")
print(f"  Valid: {len(result.valid_indices)} cells")
print(f"  Test:  {len(result.test_indices)} cells")
print(f"  Control mapping: {result.control_mapping.shape}")
print(f"  Train batches: {len(result.train_sampler)}")

dims = result.get_var_dims()
print(f"  Dimensions: {dims}")

# %% [markdown]
# ### Iterate Training Batches

# %%
# Simulate one epoch of training iteration
for epoch in range(2):
    result.train_sampler.set_epoch(epoch)
    batch_count = 0
    total_cells = 0
    for batch_indices in result.train_sampler:
        batch_count += 1
        total_cells += len(batch_indices)
    print(f"Epoch {epoch}: {batch_count} batches, {total_cells} cells")

# %% [markdown]
# ## Visualize Knockdown Effect

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Knockdown ratios for each perturbation
data_dict = source.load()
full_counts = np.asarray(data_dict["counts"])
pert_labels = np.asarray(data_dict["obs"]["perturbation"])
ctrl_cells = pert_labels == "non-targeting"

ratios = {}
for pert in perturbations:
    if pert in gene_names:
        gene_idx = gene_names.index(pert)
        ctrl_mean = full_counts[ctrl_cells, gene_idx].mean()
        if ctrl_mean > 0:
            pert_cells = pert_labels == pert
            pert_mean = full_counts[pert_cells, gene_idx].mean()
            ratios[pert] = pert_mean / ctrl_mean

ax = axes[0]
ax.bar(ratios.keys(), ratios.values(), color="#c44e52")
ax.axhline(y=0.30, color="black", linestyle="--", label="Threshold (0.30)")
ax.set_ylabel("Residual Expression Ratio")
ax.set_title("On-Target Knockdown per Perturbation")
ax.legend()
ax.set_xticklabels(list(ratios.keys()), rotation=30, ha="right")

# QC filter result
ax = axes[1]
labels = ["Pass", "Fail"]
sizes = [int(mask.sum()), int((~mask).sum())]
colors = ["#55a868", "#c44e52"]
ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90)
ax.set_title("Knockdown QC Filter Result")

plt.tight_layout()
plt.savefig(
    "docs/assets/examples/perturbation/knockdown_qc.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## TOML Configuration (cell-load Compatible)
#
# Experiment configurations can be loaded from TOML files following the
# cell-load schema:
#
# ```toml
# [datasets]
# screen1 = "/path/to/screen1/"
#
# [training]
# screen1 = "train"
#
# [zeroshot]
# "screen1.Microglia" = "test"
#
# [fewshot."screen1.Neuron"]
# val = ["BRCA1"]
# test = ["TP53", "KRAS"]
# ```

# %%
from diffbio.sources.perturbation import ExperimentConfig, load_experiment_config

# Create a sample TOML config
toml_path = tmp_dir / "experiment.toml"
toml_path.write_text(f"""\
[datasets]
screen1 = "{tmp_dir}"

[training]
screen1 = "train"

[zeroshot]
"screen1.Microglia" = "test"

[fewshot."screen1.Neuron"]
val = ["BRCA1"]
test = ["TP53", "KRAS"]
""")

config = load_experiment_config(toml_path)
print(f"Datasets: {[d.name for d in config.datasets]}")
print(f"Training: {config.training_datasets}")
print(f"Zeroshot: {[(z.cell_type, z.split) for z in config.zeroshot]}")
print(f"Fewshot: {[(f.cell_type, f.val_perturbations, f.test_perturbations) for f in config.fewshot]}")

# %% [markdown]
# ## Differentiable Downsampling
#
# The `ReadDownsampler` provides differentiable count downsampling with
# gradient flow via the straight-through estimator.

# %%
from diffbio.operators.singlecell.downsampling import (
    ReadDownsampler,
    DownsamplingConfig,
)

downsampler = ReadDownsampler(
    DownsamplingConfig(
        mode="fraction",
        fraction=0.5,
        apply_log1p=False,
        is_log1p_input=False,
    ),
    rngs=nnx.Rngs(0),
)

sample_counts = jnp.array([[100.0, 200.0, 50.0, 300.0, 10.0]])
data_in = {"counts": sample_counts}
result_ds, _, _ = downsampler.apply(data_in, {}, None)

print(f"Original total: {float(sample_counts.sum()):.0f}")
print(f"Downsampled total: {float(result_ds['counts'].sum()):.0f}")

# Verify differentiability
def ds_loss(counts):
    res, _, _ = downsampler.apply({"counts": counts}, {}, None)
    return res["counts"].sum()

grad = jax.grad(ds_loss)(sample_counts)
print(f"Gradient shape: {grad.shape}")
print(f"Gradient is non-zero: {bool(jnp.any(grad != 0))}")
print(f"Gradient is finite: {bool(jnp.all(jnp.isfinite(grad)))}")

# %% [markdown]
# ## Downsampling Fraction Sweep

# %%
fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
original_total = float(sample_counts.sum())
downsampled_totals = []

for frac in fractions:
    ds = ReadDownsampler(
        DownsamplingConfig(
            mode="fraction", fraction=frac, apply_log1p=False, is_log1p_input=False,
        ),
        rngs=nnx.Rngs(0),
    )
    res, _, _ = ds.apply({"counts": sample_counts}, {}, None)
    downsampled_totals.append(float(res["counts"].sum()))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(fractions, downsampled_totals, "o-", linewidth=2, markersize=7, color="#4c72b0")
ax.axhline(y=original_total, color="gray", linestyle="--", label=f"Original ({original_total:.0f})")
ax.set_xlabel("Downsampling Fraction")
ax.set_ylabel("Total Counts")
ax.set_title("Effect of Downsampling Fraction")
ax.legend()
plt.tight_layout()
plt.savefig(
    "docs/assets/examples/perturbation/downsampling_sweep.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## Cleanup

# %%
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
print("Temporary files cleaned up.")

# %% [markdown]
# ## Summary
#
# This example demonstrated DiffBio's perturbation data loading stack:
#
# | Component | Purpose |
# |---|---|
# | `PerturbationAnnDataSource` | Load H5AD with perturbation metadata |
# | `RandomControlMapping` / `BatchControlMapping` | Pair perturbed to control cells |
# | `PerturbationBatchSampler` | Sentence-based batch construction |
# | `ZeroShotSplitter` / `FewShotSplitter` | Evaluation split strategies |
# | `OnTargetKnockdownFilter` | 3-stage knockdown QC |
# | `ReadDownsampler` | Differentiable count downsampling |
# | `PerturbationPipeline` | End-to-end orchestration |
# | `ExperimentConfig` | TOML-based experiment configuration |
#
# ## Next Steps
#
# - Combine with `SingleCellPipeline` for differentiable downstream analysis
# - Use calibrax metrics (MMD, Sinkhorn divergence) to evaluate perturbation effects
# - Build perturbation response prediction models with artifex MLP/Transformer
