# Examples Overview

This section provides practical examples demonstrating DiffBio's capabilities.

## Example Categories

<div class="performance-metrics">
<div class="metric-card">
  <a href="basic/simple-alignment/">
    <div class="metric-value">Basic</div>
    <div class="metric-label">Getting Started</div>
  </a>
</div>
<div class="metric-card">
  <a href="advanced/variant-calling/">
    <div class="metric-value">Advanced</div>
    <div class="metric-label">Complete Pipelines</div>
  </a>
</div>
</div>

## Basic Examples

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [Simple Alignment](basic/simple-alignment.md) | Basic Smith-Waterman alignment | Operators, one-hot encoding |
| [Pileup Generation](basic/pileup-generation.md) | Generate pileups from reads | Pileup operator, quality weighting |
| [Single-Cell Clustering](basic/single-cell-clustering.md) | Soft k-means cell clustering | Single-cell, differentiability |
| [Preprocessing](basic/preprocessing.md) | Read preprocessing pipeline | Quality filtering, adapters |

## Advanced Examples

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [Variant Calling Pipeline](advanced/variant-calling.md) | End-to-end variant calling | Full pipeline, training |
| [Differential Expression](advanced/differential-expression.md) | DESeq2-style DE analysis | Statistical testing, NB-GLM |
| [Epigenomics Analysis](advanced/epigenomics-analysis.md) | Peak calling & chromatin states | ChIP-seq, ATAC-seq |
| [Multi-omics Integration](advanced/multiomics-integration.md) | Spatial deconvolution & Hi-C | Data integration |

## Running Examples

All examples can be run directly as Python scripts or in Jupyter notebooks:

```bash
# Clone the repository
git clone https://github.com/mahdi-shafiei/DiffBio.git
cd DiffBio

# Install dependencies
pip install -e ".[dev]"

# Run examples
python examples/basic_alignment.py
```

## Example Structure

Each example follows this structure:

1. **Setup**: Import libraries and configure operators
2. **Data Preparation**: Create or load input data
3. **Processing**: Apply operators or pipelines
4. **Analysis**: Interpret and visualize results
5. **Training** (if applicable): Gradient-based optimization

## Quick Reference

### Minimal Alignment Example

```python
import jax.numpy as jnp
from diffbio.operators.alignment import (
    SmoothSmithWaterman,
    SmithWatermanConfig,
    create_dna_scoring_matrix,
)

# Setup
config = SmithWatermanConfig(temperature=1.0)
scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
aligner = SmoothSmithWaterman(config, scoring_matrix=scoring)

# Align
seq1 = jnp.eye(4)[jnp.array([0, 1, 2, 3])]
seq2 = jnp.eye(4)[jnp.array([0, 1, 0, 3])]
result = aligner.align(seq1, seq2)
print(f"Score: {result.score:.2f}")
```

### Minimal Pipeline Example

```python
from diffbio.pipelines import create_variant_calling_pipeline
import jax

# Create pipeline
pipeline = create_variant_calling_pipeline(reference_length=100)

# Process data
data = {
    "reads": reads_tensor,
    "positions": positions_tensor,
    "quality": quality_tensor,
}
result, _, _ = pipeline.apply(data, {}, None)
predictions = jax.numpy.argmax(result["probabilities"], axis=-1)
```

## Contributing Examples

We welcome example contributions! See the [Contributing Guide](../development/contributing.md) for details on:

- Example formatting guidelines
- Testing requirements
- Documentation standards
