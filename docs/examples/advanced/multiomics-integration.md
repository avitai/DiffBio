# Multi-omics Integration Example

This example demonstrates differentiable multi-omics integration using DiffBio's spatial deconvolution and Hi-C contact analysis operators.

## Setup

```python
import jax
import jax.numpy as jnp
from flax import nnx
from diffbio.operators.multiomics import (
    SpatialDeconvolution,
    SpatialDeconvolutionConfig,
    HiCContactAnalysis,
    HiCContactAnalysisConfig,
)
```

## Spatial Transcriptomics Deconvolution

### Generate Synthetic Data

```python
def generate_spatial_data(n_spots=500, n_genes=200, n_cell_types=5, seed=42):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 3)

    # Cell type signatures
    signatures = jax.random.gamma(keys[0], 2.0, (n_cell_types, n_genes))
    signatures = signatures / signatures.sum(axis=1, keepdims=True) * 100

    # True proportions
    proportions = jax.random.dirichlet(keys[1], jnp.ones(n_cell_types) * 2.0, (n_spots,))

    # Spot expression
    expression = jnp.einsum('sc,cg->sg', proportions, signatures)
    expression = jax.random.poisson(keys[2], expression).astype(jnp.float32)

    coords = jax.random.uniform(jax.random.key(0), (n_spots, 2)) * 100

    return {"expression": expression, "signatures": signatures, "proportions": proportions, "coordinates": coords}

spatial_data = generate_spatial_data()
```

### Run Deconvolution

```python
config = SpatialDeconvolutionConfig(n_genes=200, n_cell_types=5, hidden_dim=64)
deconv = SpatialDeconvolution(config, rngs=nnx.Rngs(42))

data = {
    "spot_expression": spatial_data["expression"],
    "reference_profiles": spatial_data["signatures"],
    "coordinates": spatial_data["coordinates"],
}
result, _, _ = deconv.apply(data, {}, None)
cell_proportions = result["cell_proportions"]
print(f"Cell proportions shape: {cell_proportions.shape}")
```

## Hi-C Contact Analysis

```python
# Generate synthetic Hi-C matrix
n_bins = 50
i, j = jnp.meshgrid(jnp.arange(n_bins), jnp.arange(n_bins))
contacts = 100 * jnp.exp(-jnp.abs(i - j) / 10)

# Generate bin features
bin_features = jax.random.normal(jax.random.key(0), (n_bins, 16))

config = HiCContactAnalysisConfig(n_bins=n_bins, hidden_dim=64, bin_features=16)
hic_analyzer = HiCContactAnalysis(config, rngs=nnx.Rngs(42))

result, _, _ = hic_analyzer.apply(
    {"contact_matrix": contacts, "bin_features": bin_features}, {}, None
)
tad_boundary_scores = result["tad_boundary_scores"]
compartment_scores = result["compartment_scores"]
print(f"TAD boundary scores shape: {tad_boundary_scores.shape}")
print(f"Compartment scores shape: {compartment_scores.shape}")
```

## Next Steps

- See [Epigenomics Analysis](epigenomics-analysis.md) for chromatin state annotation
- Explore [Multi-omics Operators](../../user-guide/operators/multiomics.md) for details
