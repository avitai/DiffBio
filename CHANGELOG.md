# Changelog

All notable changes to DiffBio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
