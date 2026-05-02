# DiffBio

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://jax.readthedocs.io/"><img src="https://img.shields.io/badge/JAX-0.6.1+-green.svg" alt="JAX"></a>
  <a href="https://flax.readthedocs.io/"><img src="https://img.shields.io/badge/Flax-0.12+-orange.svg" alt="Flax"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

<p align="center">
  <strong>End-to-End Differentiable Bioinformatics Pipelines</strong>
</p>

<p align="center">
  Built on <a href="https://github.com/avitai/datarax">Datarax</a>, <a href="https://github.com/avitai/artifex">Artifex</a>, <a href="https://github.com/avitai/Opifex">Opifex</a>, and <a href="https://github.com/avitai/calibrax">Calibrax</a> | Powered by <a href="https://jax.readthedocs.io/">JAX</a> & <a href="https://flax.readthedocs.io/">Flax NNX</a>
</p>

---

> **⚠️ Early Development - API Unstable**
>
> DiffBio is currently in early development and undergoing rapid iteration. Please be aware of the following implications:
>
> | Area | Status | Impact |
> |------|--------|--------|
> | **API** | 🔄 Unstable | Breaking changes are expected. Public interfaces may change without deprecation warnings. Pin to specific commits if stability is required. |
> | **Tests** | 🔄 In Flux | Test suite is being expanded. Some tests may fail or be skipped. Coverage metrics are improving but not yet full. |
> | **Documentation** | 🔄 Evolving | Docs may not reflect current implementation. Code examples might be outdated. Refer to source code and tests for accurate usage. |
>
> We recommend waiting for a stable release (v1.0) before using DiffBio in production. For research and experimentation, proceed with the understanding that APIs will evolve.

---

## Overview

DiffBio is a framework for building **end-to-end differentiable bioinformatics
pipelines**. By replacing discrete operations with differentiable relaxations,
DiffBio enables gradient-based optimization through entire analysis workflows.

DiffBio is the biology-specific differentiable operator layer of a wider
JAX/NNX scientific ML ecosystem. It uses:

- **Datarax** for operator and dataflow contracts
- **Artifex** for reusable model-building and transformer components
- **Opifex** for scientific ML and advanced optimization primitives
- **Calibrax** for metrics, benchmarking, comparison, and regression control

Traditional bioinformatics pipelines use discrete operations (hard thresholds, argmax decisions) that block gradient flow. DiffBio addresses this by:

- **Soft quality filtering** using sigmoid-based weights instead of hard cutoffs
- **Differentiable pileup** with soft position assignments via temperature-controlled softmax
- **Soft alignment scoring** replacing discrete Smith-Waterman with continuous relaxations
- **End-to-end training** of complete pipelines using gradient descent

This enables learning optimal pipeline parameters directly from data, rather than manual tuning.

## Features

- **40+ Differentiable Operators** covering alignment, variant calling, single-cell analysis, epigenomics, RNA-seq, preprocessing, normalization, multi-omics, drug discovery, and protein/RNA structure
- **6 End-to-End Pipelines** for variant calling, enhanced variant calling, single-cell analysis, differential expression, perturbation, and preprocessing
- **GPU-Accelerated** computation via JAX's XLA compilation
- **Composable Architecture** built on the Datarax, Artifex, Opifex, and Calibrax stack
- **Training Utilities** with gradient clipping, custom loss functions, and synthetic data generation

For complete operator and pipeline listings, see the [Operators Overview](https://diffbio.readthedocs.io/en/latest/user-guide/operators/overview/) and [Pipelines Overview](https://diffbio.readthedocs.io/en/latest/user-guide/pipelines/overview/) in the documentation.

## Installation

```bash
# Clone the repository
git clone https://github.com/avitai/DiffBio.git
cd DiffBio

# Install with uv
uv sync
```

## Quick Start

### Using Individual Operators

```python
import jax
import jax.numpy as jnp
from flax import nnx

from diffbio.operators import DifferentiableQualityFilter, QualityFilterConfig

# Quality filtering with learnable threshold (default initial_threshold=20.0)
quality_filter = DifferentiableQualityFilter(
    QualityFilterConfig(initial_threshold=20.0),
    rngs=nnx.Rngs(0),
)

# Apply to a one-hot encoded sequence with per-position quality scores
quality_scores = jnp.array([35.0, 15.0, 28.0, 10.0])
sequence = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 4)  # (length, alphabet=4)
data = {"sequence": sequence, "quality_scores": quality_scores}

filtered_data, _, _ = quality_filter.apply(data, {}, None)
# filtered_data["sequence"]        — sequence with low-quality positions softly suppressed
# filtered_data["quality_scores"]  — pass-through quality values
```

### Using the Variant Calling Pipeline

```python
from diffbio.pipelines import (
    VariantCallingPipeline,
    VariantCallingPipelineConfig,
    create_variant_calling_pipeline,
)

# Create pipeline with default configuration
pipeline = create_variant_calling_pipeline(
    reference_length=100,
    num_classes=3,  # ref, SNP, indel
    hidden_dim=32,
    seed=42,
)

# Process reads
batch_data = {
    "reads": reads,           # (num_reads, read_length, 4)
    "positions": positions,   # (num_reads,)
    "quality": quality,       # (num_reads, read_length)
}

result, _, _ = pipeline.apply(batch_data, {}, None)
# result["logits"] contains per-position variant predictions
# result["probabilities"] contains class probabilities
```

### Training a Pipeline

```python
from diffbio.utils import (
    Trainer,
    TrainingConfig,
    cross_entropy_loss,
    create_synthetic_training_data,
    data_iterator,
)

# Generate synthetic training data
inputs, targets = create_synthetic_training_data(
    num_samples=100,
    num_reads=10,
    read_length=50,
    reference_length=100,
    variant_rate=0.1,
)

# Configure training
config = TrainingConfig(
    learning_rate=1e-3,
    num_epochs=50,
    log_every=10,
    grad_clip_norm=1.0,
)

# Create trainer
trainer = Trainer(pipeline, config)

# Define loss function
def loss_fn(predictions, targets):
    return cross_entropy_loss(
        predictions["logits"],
        targets["labels"],
        num_classes=3,
    )

# Train
trainer.train(
    data_iterator_fn=lambda: data_iterator(inputs, targets),
    loss_fn=loss_fn,
)

# Access trained pipeline
trained_pipeline = trainer.pipeline
```

## Architecture

DiffBio sits on a layered ecosystem rather than standing alone:

| Layer | Library | Role In DiffBio |
|---|---|---|
| Execution contracts | [Datarax](https://github.com/avitai/datarax) | Operator, data-source, and pipeline contracts |
| Modeling substrate | [Artifex](https://github.com/avitai/artifex) | Reusable transformer and generative-model components |
| Scientific ML substrate | [Opifex](https://github.com/avitai/Opifex) | Scientific optimization, operator learning, and advanced training methods |
| Evaluation substrate | [Calibrax](https://github.com/avitai/calibrax) | Metrics, benchmarking, comparison, profiling, and regression checks |

DiffBio itself sits on top of these as the biology-specific layer: differentiable
biological operators and end-to-end pipeline compositions (alignment, variant
calling, single-cell analysis, drug discovery, structural biology, multi-omics).

Each DiffBio operator inherits from Datarax's `OperatorModule` and implements:

```
apply(data, state, metadata) -> (output_data, output_state, output_metadata)
```

This enables:
- **Composition**: Chain operators into pipelines
- **Batch processing**: Automatic vectorization via `apply_batch()`
- **Gradient flow**: End-to-end differentiability through the pipeline

### Operator Composition

`apply()` runs an operator on a single element (no batch dimension). Operators
are chained by threading the `(data, state, metadata)` triple returned by
`apply()` into the next operator:

```python
data, state, metadata = quality_filter.apply(element_data, {}, None)
data, state, metadata = pileup.apply(data, state, metadata)
data, state, metadata = classifier.apply(data, state, metadata)

# `data` is a dict of JAX arrays — read out the per-position predictions
predictions = data["logits"]
```

For batched data wrapped in a Datarax `Batch`, call the operator directly
(or use `apply_batch()`); both delegate to the same code path:

```python
from datarax import Batch

batch = Batch.from_parts(...)        # construct from a list of elements
batch = quality_filter(batch)         # equivalent to quality_filter.apply_batch(batch)
batch = pileup(batch)
batch = classifier(batch)
```

## Testing

```bash
# Run all tests
uv run pytest -vv

# Run with coverage
uv run pytest -vv --cov=src/ --cov-report=term-missing

# Run specific test modules
uv run pytest tests/operators/ -vv
uv run pytest tests/pipelines/ -vv
uv run pytest tests/integration/ -vv
```

## Project Structure

```
DiffBio/
├── src/diffbio/
│   ├── core/                # Base operators, graph utils, soft ops, neural components
│   ├── operators/           # 40+ differentiable operators
│   │   ├── alignment/       # Smith-Waterman, profile HMM, soft MSA
│   │   ├── assembly/        # GNN assembly, metagenomic binning
│   │   ├── crispr/          # Guide RNA scoring
│   │   ├── drug_discovery/  # Fingerprints, ADMET, AttentiveFP, MACCS keys
│   │   ├── epigenomics/     # Peak calling, chromatin state, contextual epigenomics
│   │   ├── foundation_models/ # Geneformer/scGPT adapters, transformer encoders
│   │   ├── mapping/         # Neural read mapping
│   │   ├── metabolomics/    # Spectral similarity
│   │   ├── molecular_dynamics/ # Force fields, MD integrators
│   │   ├── multiomics/      # Hi-C, spatial deconvolution, multi-omics VAE
│   │   ├── normalization/   # VAE normalizer, UMAP, PHATE, embeddings
│   │   ├── population/      # Ancestry estimation
│   │   ├── preprocessing/   # Adapter removal, duplicate weighting, error correction
│   │   ├── protein/         # Secondary structure
│   │   ├── rna_structure/   # RNA folding
│   │   ├── rnaseq/          # Splicing PSI, motif discovery
│   │   ├── singlecell/      # Clustering, trajectory, velocity, GRN, batch correction, ...
│   │   ├── statistical/     # HMM, NB GLM, EM quantification
│   │   └── variant/         # Pileup, classifiers, CNV segmentation
│   ├── pipelines/           # 6 end-to-end pipelines
│   ├── losses/              # Alignment, biological-regularization, single-cell, statistical, metric
│   ├── sources/             # Data loaders (FASTA, BAM, AnnData, MoleculeNet, indexed views)
│   ├── splitters/           # Random, stratified, scaffold, Tanimoto, sequence-identity
│   ├── samplers/            # Perturbation samplers
│   ├── sequences/           # DNA / RNA encoding utilities
│   ├── evaluation/          # Evaluation runner and graders
│   └── utils/               # Training utilities, dependency-runtime checks
├── tests/                   # Unit, integration, and benchmark tests
├── benchmarks/              # Domain benchmarks with training + baselines
├── examples/                # Runnable example scripts paired with notebooks
└── docs/                    # MkDocs documentation
```

## Requirements

- Python 3.11+
- JAX 0.6.1+
- Flax 0.12+
- Optax 0.1.4+
- jaxtyping 0.2.20+
- Datarax, Artifex, Opifex, and Calibrax (installed automatically from PyPI)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

DiffBio builds on ideas from:
- [Datarax](https://github.com/avitai/datarax): Composable data processing framework
- [Artifex](https://github.com/avitai/artifex): Generative-model and transformer substrate
- [Opifex](https://github.com/avitai/Opifex): Scientific ML and advanced optimization substrate
- [Calibrax](https://github.com/avitai/calibrax): Benchmarking, comparison, and regression substrate
- [Flax NNX](https://flax.readthedocs.io/): Neural network library for JAX
