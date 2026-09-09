# Changelog

All notable changes to DiffBio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-09-09

### Changed

- The sibling floors are datarax 0.1.6, avitai-artifex 0.1.5, opifex 0.2.2 and
  calibrax 0.1.5, and `substrax` joins the runtime dependencies and the ecosystem
  packages the runtime check verifies. The declared jax floor is 0.11.1 and the flax
  floor 0.12.9, what every lock already resolved. Python 3.13 is supported.
- `ProteinExtensionConfig` is imported from its home,
  `artifex.generative_models.core.configuration`.
- `scripts/verify_gpu_setup.py` reports the device identity substrax detects
  (platform, device kind, count, kinds) and `--require-gpu` reads the kind.
- `diffbio.__version__` is read from the installed distribution's metadata instead
  of a hand-maintained string, which had stayed at 0.1.0 through the 0.1.1 release.
- The four advanced example pages that built `nnx.Optimizer` without `wrt=` and
  called `update(grads)` now match flax 0.12: `wrt=nnx.Param` and
  `update(model, grads)`.
- Publishing uses PyPI trusted publishing (OIDC) and `twine check --strict`; the
  README installs from PyPI, with `setup.sh` for a source checkout.
- CI checks that `uv.lock` matches `pyproject.toml`; the ruff hooks no longer
  receive `uv.lock`, which `identify` classifies as TOML.

### Removed

- The direct `orbax-checkpoint` dependency (nothing in this package imports it; it
  stays a transitive dependency of the siblings).
- `scripts/generate_benchmark_plots.py` and the sixteen images it rendered under
  `docs/assets/images/benchmarks/`, which no page referenced, and the unused
  `annotate_heatmap` helper.

## [0.1.1] - 2026-08-30

### Changed

- **Requires Python 3.12 or later.** jax 0.11.0 dropped 3.11, and this release
  takes that jax line.
- **The `gpu` extra is renamed `cuda12`.** JAX names its own extras for the CUDA
  major version and publishes no `gpu` extra, and this package also ships
  `metal`, which is a GPU.
- Resolves to jax 0.11.1, jaxlib 0.11.1, flax 0.12.9, optax 0.2.8 and grain
  0.2.18, and raises the sibling floors to opifex 0.2.1, avitai-artifex 0.1.4,
  calibrax 0.1.2 and datarax 0.1.5.
- `flax>=0.12.1` and `optax>=0.2.8` are now required; below those, flax lacks
  `nnx.Variable.set_value` and optax sets a jax config option removed in 0.10,
  which takes out collection for the whole suite.
- `dm-haiku>=0.0.17` is constrained. It arrives through `jax-md`, and up to
  0.0.16 it binds `jax.core.DropVar` at import, which jax removed in 0.11.0.
- `datasets` is declared in the `benchmark` extra, where
  `benchmarks/crossmodality` uses it. It had been arriving transitively through
  the sibling packages, which now keep their heavy dependencies behind extras.

### Fixed

- `top_k_mask` stays inside the `[0, 1]` bound it documents. The isotonic solver
  could finish one float32 ULP outside it; the forward value is saturated onto
  the bound while the backward pass keeps the projection's own gradient, so the
  gradients are unchanged.
- Benchmark training constructs its optimizer through the shared Opifex boundary
  rather than calling `optax.adamw` directly, preserving the optimizer exactly.

## [0.1.0] - 2026-05-02

Initial public release of DiffBio: end-to-end differentiable bioinformatics
pipelines built on JAX, Flax NNX, and the Datarax / Artifex / Opifex /
Calibrax ecosystem.

### Added

- 40+ differentiable operators across alignment, variant calling, single-cell
  analysis, drug discovery, epigenomics, multi-omics, RNA structure, protein
  structure, molecular dynamics, foundation models, and preprocessing.
- Six end-to-end pipelines: `VariantCallingPipeline`,
  `EnhancedVariantCallingPipeline`, `SingleCellPipeline`,
  `DifferentialExpressionPipeline`, `PerturbationPipeline`, and
  `PreprocessingPipeline`.
- Soft-operations primitive layer (`diffbio.core.soft_ops`) with
  straight-through and gradient-replacement variants for use inside
  differentiable bioinformatics workflows.
- Dataset sources for FASTA, BAM, AnnData, MoleculeNet, and indexed views.
- Dataset splitters for random, stratified, scaffold, Tanimoto cluster, and
  sequence-identity splits.
- Loss functions for alignment, biological regularization, single-cell
  analysis, statistical models, and metric learning.
- Training utilities (`Trainer`, `TrainingConfig`, optimizer factories,
  synthetic data generation, gradient clipping).
- Documentation site: getting-started guides, user-guide, API reference,
  examples (basic / intermediate / advanced), and contributor guides.
- Benchmark suite under `benchmarks/` with tier-based runner
  (`run_all.py --tier ci|nightly|full`) and SOTA baseline comparisons across
  single-cell, alignment, RNA structure, protein, molecular dynamics, and
  statistical domains.
- CI/CD: sharded unit tests with `pytest-xdist`, integration / e2e /
  performance jobs, coverage aggregation, security scanning, build
  verification, and documentation deployment workflows.
