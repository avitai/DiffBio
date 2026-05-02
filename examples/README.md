# DiffBio Examples

Runnable examples demonstrating differentiable bioinformatics operators and pipelines.

## Running Examples

Activate the environment and run any example with `uv`:

```bash
source ./activate.sh
uv run python examples/basics/operator_pattern.py
```

Each example generates synthetic data and runs entirely on CPU. No external datasets or GPU required unless noted.

## Directory Map

| Directory | Contents |
|-----------|----------|
| `basics/` | Foundational patterns: operator construction, apply, gradients, JIT |
| `singlecell/` | Single-cell analysis: clustering, batch correction, annotation, trajectory, imputation, GRN inference, doublet detection, spatial analysis |
| `pipelines/` | End-to-end pipeline composition across multiple operators |
| `perturbation/` | Cell-load-style perturbation experiment data loading |
| `ecosystem/` | Integration with datarax, artifex, calibrax, and opifex |

## Examples

### Basics

| Example | Level | Duration | Description |
|---------|-------|----------|-------------|
| [`basics/operator_pattern.py`](basics/operator_pattern.py) | Basic | 5–10 min | Universal DiffBio operator pattern using SoftKMeansClustering |

### Single-Cell

| Example | Level | Duration | Description |
|---------|-------|----------|-------------|
| [`singlecell/clustering.py`](singlecell/clustering.py) | Basic | 5–10 min | Soft k-means clustering with optax training loop |
| [`singlecell/batch_correction.py`](singlecell/batch_correction.py) | Intermediate | 10–20 min | Differentiable batch-effect correction across donors |
| [`singlecell/cell_annotation.py`](singlecell/cell_annotation.py) | Intermediate | 10–20 min | Soft cell-type annotation with marker-gene priors |
| [`singlecell/imputation.py`](singlecell/imputation.py) | Intermediate | 10–20 min | MAGIC-style diffusion imputation with dropout recovery |
| [`singlecell/trajectory.py`](singlecell/trajectory.py) | Intermediate | 10–20 min | Pseudotime ordering and fate probability estimation |
| [`singlecell/doublet_detection.py`](singlecell/doublet_detection.py) | Intermediate | 10–20 min | Differentiable doublet scoring on simulated multiplets |
| [`singlecell/grn_inference.py`](singlecell/grn_inference.py) | Advanced | 15–30 min | Gene regulatory network inference with gradient-trained weights |
| [`singlecell/spatial_analysis.py`](singlecell/spatial_analysis.py) | Advanced | 15–30 min | Spatial transcriptomics analysis with neighborhood operators |

### Pipelines

| Example | Level | Duration | Description |
|---------|-------|----------|-------------|
| [`pipelines/singlecell_pipeline.py`](pipelines/singlecell_pipeline.py) | Advanced | 20–40 min | End-to-end single-cell pipeline composing multiple operators |

### Perturbation

| Example | Level | Duration | Description |
|---------|-------|----------|-------------|
| [`perturbation/perturbation_data_loading.py`](perturbation/perturbation_data_loading.py) | Intermediate | 10–20 min | Cell-load-style perturbation experiment data loading via datarax sources |

### Ecosystem

| Example | Level | Duration | Description |
|---------|-------|----------|-------------|
| [`ecosystem/calibrax_metrics.py`](ecosystem/calibrax_metrics.py) | Intermediate | 5–15 min | Using calibrax functional metrics for differentiable evaluation |
| [`ecosystem/scvi_benchmark.py`](ecosystem/scvi_benchmark.py) | Advanced | 20–40 min | Comparing DiffBio operators against scVI on a shared dataset |

## Dual-Format Support

Examples are written in Jupytext percent format. Every `.py` file ships with a paired `.ipynb`. To regenerate the notebook after editing the `.py`:

```bash
uv run python scripts/jupytext_converter.py sync examples/path/to/example.py
```

## Prerequisites

- DiffBio installed (`uv sync` from the project root)
- JAX (CPU backend is sufficient for all examples)
- optax (for training loop examples)
