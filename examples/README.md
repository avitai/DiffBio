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
| `singlecell/` | Single-cell analysis: clustering, imputation, trajectory inference |
| `pipelines/` | End-to-end pipeline composition across multiple operators |
| `ecosystem/` | Integration with datarax, artifex, calibrax, and opifex |

## Examples

### Basics

| Example | Level | Duration | Description |
|---------|-------|----------|-------------|
| [`operator_pattern.py`](basics/operator_pattern.py) | Basic | 5-10 min | Universal DiffBio operator pattern using SoftKMeansClustering |

### Single-Cell

| Example | Level | Duration | Description |
|---------|-------|----------|-------------|
| [`clustering.py`](singlecell/clustering.py) | Basic | 5-10 min | Soft k-means clustering with optax training loop |
| [`imputation.py`](singlecell/imputation.py) | Intermediate | 10-20 min | MAGIC-style diffusion imputation with dropout recovery |
| [`trajectory.py`](singlecell/trajectory.py) | Intermediate | 10-20 min | Pseudotime ordering and fate probability estimation |

## Dual-Format Support

Examples are written in Jupytext percent format. To convert to a paired notebook:

```bash
uv run python scripts/jupytext_converter.py sync examples/path/to/example.py
```

## Prerequisites

- DiffBio installed (`uv pip install -e .`)
- JAX (CPU backend is sufficient for all examples)
- optax (for training loop examples)
