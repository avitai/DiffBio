# DiffBio

**End-to-end differentiable bioinformatics pipelines built on JAX/Flax and Datarax**

DiffBio provides differentiable implementations of core bioinformatics algorithms, enabling gradient-based optimization of entire genomic analysis pipelines. Built on top of the [Datarax](https://github.com/mahdi-shafiei/workshop-data) framework, DiffBio brings the power of automatic differentiation to sequence alignment, variant calling, and genomic analysis.

---

## Key Features

<div class="performance-metrics">
<div class="metric-card">
  <div class="metric-value">JAX</div>
  <div class="metric-label">GPU-Accelerated</div>
</div>
<div class="metric-card">
  <div class="metric-value">100%</div>
  <div class="metric-label">Differentiable</div>
</div>
<div class="metric-card">
  <div class="metric-value">Modular</div>
  <div class="metric-label">Composable Operators</div>
</div>
</div>

### Differentiable Operators

- **Smith-Waterman Alignment**: Smooth, differentiable sequence alignment with soft gap penalties
- **Pileup Generation**: Differentiable read pileup computation for variant detection
- **Quality Filtering**: Soft-thresholded quality score filtering

### End-to-End Pipelines

- **Variant Calling Pipeline**: Complete differentiable pipeline from reads to variants
- **Composable Architecture**: Chain operators using the Datarax framework

### Training Utilities

- **Gradient-based Optimization**: Train alignment and variant calling models end-to-end
- **Custom Loss Functions**: Flexible loss definitions for bioinformatics tasks

---

## Quick Start

### Installation

```bash
pip install diffbio
```

Or install from source:

```bash
git clone https://github.com/mahdi-shafiei/DiffBio.git
cd DiffBio
pip install -e ".[dev]"
```

### Basic Usage

```python
import jax.numpy as jnp
from diffbio.operators import DifferentiableSmithWaterman

# Initialize the Smith-Waterman operator
sw = DifferentiableSmithWaterman(
    match_score=2.0,
    mismatch_penalty=-1.0,
    gap_open=-2.0,
    gap_extend=-0.5,
    temperature=1.0
)

# Encode sequences (A=0, C=1, G=2, T=3)
query = jnp.array([0, 1, 2, 3, 0, 1])  # ACGTAC
target = jnp.array([0, 1, 0, 3, 0, 1]) # ACATAC

# Compute differentiable alignment
result = sw.apply_batch(query, target)
print(f"Alignment score: {result['score']}")
```

### Gradient Computation

```python
import jax

def alignment_loss(params, query, target):
    sw = DifferentiableSmithWaterman(**params)
    result = sw.apply_batch(query, target)
    return -result['score']  # Maximize alignment score

# Compute gradients
grads = jax.grad(alignment_loss)(params, query, target)
```

---

## Architecture

DiffBio follows the **Datarax operator pattern** for composable, differentiable data processing:

```mermaid
graph LR
    A[Raw Reads] --> B[Quality Filter]
    B --> C[Smith-Waterman Alignment]
    C --> D[Pileup Generation]
    D --> E[Variant Calling]
    E --> F[Variants]

    style A fill:#0d9488,color:#fff
    style B fill:#6366f1,color:#fff
    style C fill:#0891b2,color:#fff
    style D fill:#7c3aed,color:#fff
    style E fill:#dc2626,color:#fff
    style F fill:#059669,color:#fff
```

Each operator in the pipeline is fully differentiable, allowing gradients to flow from the final output back through all processing steps.

---

## Documentation Structure

<div class="performance-metrics">
<div class="metric-card">
  <a href="getting-started/installation/">
    <div class="metric-value">Getting Started</div>
    <div class="metric-label">Installation & Quick Start</div>
  </a>
</div>
<div class="metric-card">
  <a href="user-guide/concepts/differentiable-bioinformatics/">
    <div class="metric-value">User Guide</div>
    <div class="metric-label">Concepts & Tutorials</div>
  </a>
</div>
<div class="metric-card">
  <a href="api/core/base/">
    <div class="metric-value">API Reference</div>
    <div class="metric-label">Complete API Docs</div>
  </a>
</div>
<div class="metric-card">
  <a href="examples/overview/">
    <div class="metric-value">Examples</div>
    <div class="metric-label">Code Examples</div>
  </a>
</div>
</div>

---

## Why Differentiable Bioinformatics?

Traditional bioinformatics pipelines consist of discrete, non-differentiable operations that prevent end-to-end optimization. DiffBio addresses this by:

1. **Smooth Approximations**: Using temperature-scaled softmax and sigmoid functions to create differentiable versions of discrete operations
2. **Gradient Flow**: Enabling gradients to propagate through entire pipelines for joint optimization
3. **Learnable Parameters**: Allowing alignment scores, gap penalties, and quality thresholds to be learned from data
4. **GPU Acceleration**: Leveraging JAX's XLA compilation for high-performance computation

### Applications

- **Adaptive Alignment**: Learn optimal scoring parameters for specific sequence types
- **Joint Optimization**: Train variant callers end-to-end with alignment quality
- **Transfer Learning**: Fine-tune pre-trained models on domain-specific data
- **Neural Integration**: Combine traditional algorithms with neural network components

---

## Citation

If you use DiffBio in your research, please cite:

```bibtex
@software{diffbio2024,
  title={DiffBio: End-to-End Differentiable Bioinformatics Pipelines},
  author={Shafiei, Mahdi},
  year={2024},
  url={https://github.com/mahdi-shafiei/DiffBio}
}
```

---

## License

DiffBio is released under the [MIT License](https://github.com/mahdi-shafiei/DiffBio/blob/main/LICENSE).

## Contributing

We welcome contributions! See our [Contributing Guide](development/contributing.md) for details.
