# Changelog

All notable changes to DiffBio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
