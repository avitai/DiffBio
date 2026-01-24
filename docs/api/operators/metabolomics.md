# Metabolomics Operators API

Differentiable operators for metabolomics analysis, including MS/MS spectral similarity.

## DifferentiableSpectralSimilarity

::: diffbio.operators.metabolomics.spectral_similarity.DifferentiableSpectralSimilarity
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply
        - encode
        - cosine_similarity

## SpectralSimilarityConfig

::: diffbio.operators.metabolomics.spectral_similarity.SpectralSimilarityConfig
    options:
      show_root_heading: true
      members: []

## bin_spectrum

::: diffbio.operators.metabolomics.spectral_similarity.bin_spectrum
    options:
      show_root_heading: true
      show_source: false

## create_spectral_similarity

::: diffbio.operators.metabolomics.spectral_similarity.create_spectral_similarity
    options:
      show_root_heading: true
      show_source: false

## Usage Examples

### Basic Spectral Similarity

```python
from flax import nnx
import jax
from diffbio.operators.metabolomics import (
    DifferentiableSpectralSimilarity,
    SpectralSimilarityConfig,
    create_spectral_similarity,
    bin_spectrum,
)

# Using config
config = SpectralSimilarityConfig(
    n_bins=1000,
    embedding_dim=200,
    hidden_dims=(512, 256),
)
operator = DifferentiableSpectralSimilarity(config, rngs=nnx.Rngs(42))

# Or using factory function
operator = create_spectral_similarity(n_bins=1000)

# Compute embeddings
spectra = jax.random.uniform(jax.random.PRNGKey(0), (100, 1000))
result, _, _ = operator.apply({"spectra": spectra}, {}, None)
embeddings = result["embeddings"]  # (100, 200)
```

### Pairwise Similarity

```python
# Compare pairs of spectra
spectra_a = jax.random.uniform(jax.random.PRNGKey(0), (50, 1000))
spectra_b = jax.random.uniform(jax.random.PRNGKey(1), (50, 1000))

data = {"spectra_a": spectra_a, "spectra_b": spectra_b}
result, _, _ = operator.apply(data, {}, None)

similarity = result["similarity_scores"]  # (50,) in [-1, 1]
```

### Spectrum Binning

```python
import jax.numpy as jnp
from diffbio.operators.metabolomics import bin_spectrum

# Raw mass spectrum
mz_values = jnp.array([100.0, 150.5, 200.0, 350.2, 500.0])
intensities = jnp.array([0.3, 1.0, 0.5, 0.8, 0.2])

# Discretize into bins
binned = bin_spectrum(
    mz_values,
    intensities,
    n_bins=1000,
    min_mz=0.0,
    max_mz=1000.0,
    normalize=True,
)
# binned.shape == (1000,)
```

## Input Specifications

### Single Spectra Mode

| Key | Shape | Description |
|-----|-------|-------------|
| `spectra` | (n_spectra, n_bins) | Binned mass spectra |

### Paired Spectra Mode

| Key | Shape | Description |
|-----|-------|-------------|
| `spectra_a` | (n_pairs, n_bins) | First set of binned spectra |
| `spectra_b` | (n_pairs, n_bins) | Second set of binned spectra |

## Output Specifications

### Single Spectra Mode

| Key | Shape | Description |
|-----|-------|-------------|
| `spectra` | (n_spectra, n_bins) | Original input spectra |
| `embeddings` | (n_spectra, embedding_dim) | Spectral embeddings |

### Paired Spectra Mode

| Key | Shape | Description |
|-----|-------|-------------|
| `spectra_a` | (n_pairs, n_bins) | Original first spectra |
| `spectra_b` | (n_pairs, n_bins) | Original second spectra |
| `embeddings_a` | (n_pairs, embedding_dim) | First set embeddings |
| `embeddings_b` | (n_pairs, embedding_dim) | Second set embeddings |
| `similarity_scores` | (n_pairs,) | Cosine similarity scores |
