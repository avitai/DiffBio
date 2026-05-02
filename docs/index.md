# DiffBio

**End-to-end differentiable bioinformatics pipelines built on JAX/Flax and Datarax**

DiffBio provides differentiable implementations of core bioinformatics algorithms, enabling gradient-based optimization of entire genomic analysis pipelines. Built on top of the [Datarax](https://github.com/avitai/datarax) framework, DiffBio brings the power of automatic differentiation to sequence alignment, variant calling, and genomic analysis.

---

## Key Features

<div class="stat-grid">
<div class="stat-card">
  <div class="stat-value">JAX</div>
  <div class="stat-label">GPU-Accelerated</div>
</div>
<div class="stat-card">
  <div class="stat-value">End-to-End</div>
  <div class="stat-label">Differentiable</div>
</div>
<div class="stat-card">
  <div class="stat-value">Modular</div>
  <div class="stat-label">Composable Operators</div>
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
git clone https://github.com/avitai/DiffBio.git
cd DiffBio
./setup.sh
source ./activate.sh
```

### Basic Usage

```python
import jax
import jax.numpy as jnp
from flax import nnx
from diffbio.operators.singlecell import SoftKMeansClustering, SoftClusteringConfig

# Configure and create an operator
config = SoftClusteringConfig(n_clusters=5, n_features=20)
operator = SoftKMeansClustering(config, rngs=nnx.Rngs(42))

# Generate synthetic data and run
data = {"embeddings": jax.random.normal(jax.random.key(0), (100, 20))}
result, state, metadata = operator.apply(data, {}, None)
print(f"Cluster assignments: {result['cluster_assignments'].shape}")
```

### Gradient Computation

```python
# Gradients flow through all operators
def loss_fn(input_data):
    result, _, _ = operator.apply(input_data, {}, None)
    return result["cluster_assignments"].sum()

grad = jax.grad(loss_fn)(data)
print(f"Gradient is non-zero: {bool(jnp.any(grad['embeddings'] != 0))}")
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

    style A fill:#e0e7ff,stroke:#4338ca,color:#312e81
    style B fill:#e0e7ff,stroke:#4338ca,color:#312e81
    style C fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    style D fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    style E fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    style F fill:#d1fae5,stroke:#059669,color:#064e3b
```

Each operator in the pipeline is fully differentiable, allowing gradients to flow from the final output back through all processing steps.

---

## Documentation Structure

<div class="nav-grid">
<div class="nav-card">
  <a href="getting-started/installation/">
    <div class="nav-title">Getting Started</div>
    <div class="nav-description">Installation & Quick Start</div>
  </a>
</div>
<div class="nav-card">
  <a href="user-guide/concepts/differentiable-bioinformatics/">
    <div class="nav-title">User Guide</div>
    <div class="nav-description">Concepts & Tutorials</div>
  </a>
</div>
<div class="nav-card">
  <a href="api/core/base/">
    <div class="nav-title">API Reference</div>
    <div class="nav-description">Complete API Docs</div>
  </a>
</div>
<div class="nav-card">
  <a href="examples/overview/">
    <div class="nav-title">Examples</div>
    <div class="nav-description">Runnable Code Examples</div>
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
@software{diffbio2026,
  title={DiffBio: End-to-End Differentiable Bioinformatics Pipelines},
  author={Shafiei, Mahdi},
  year={2026},
  url={https://github.com/avitai/DiffBio},
  version={0.1.0}
}
```

---

## License

DiffBio is released under the [MIT License](https://github.com/avitai/DiffBio/blob/main/LICENSE).

## Contributing

We welcome contributions! See our [Contributing Guide](development/contributing.md) for details.
