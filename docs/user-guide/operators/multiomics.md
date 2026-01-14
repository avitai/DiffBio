# Multi-omics Operators

DiffBio provides differentiable operators for multi-omics data analysis, including spatial transcriptomics and chromatin conformation.

<span class="operator-multiomics">Multi-omics</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Multi-omics operators enable end-to-end optimization of:

- **SpatialDeconvolution**: Cell type deconvolution for spatial transcriptomics
- **HiCContactAnalysis**: Chromatin contact analysis for Hi-C data

## SpatialDeconvolution

Cell type deconvolution for spatial transcriptomics data.

### Quick Start

```python
from flax import nnx
from diffbio.operators.multiomics import SpatialDeconvolution, SpatialDeconvConfig

# Configure deconvolution
config = SpatialDeconvConfig(
    n_cell_types=10,
    n_genes=2000,
    hidden_dim=128,
    temperature=1.0,
)

# Create operator
rngs = nnx.Rngs(42)
deconv = SpatialDeconvolution(config, rngs=rngs)

# Apply deconvolution
data = {
    "spatial_expression": spot_expression,     # (n_spots, n_genes)
    "reference_profiles": cell_type_profiles,  # (n_cell_types, n_genes)
}
result, state, metadata = deconv.apply(data, {}, None)

# Get cell type proportions
proportions = result["proportions"]            # (n_spots, n_cell_types)
reconstructed = result["reconstructed"]        # Reconstructed expression
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_cell_types` | int | 10 | Number of cell types |
| `n_genes` | int | 2000 | Number of genes |
| `hidden_dim` | int | 128 | Neural network hidden dimension |
| `temperature` | float | 1.0 | Softmax temperature |

### Deconvolution Model

The deconvolution estimates cell type proportions $\pi$ such that:

$$X_{spot} \approx \sum_k \pi_k \cdot R_k$$

Where:
- $X_{spot}$ = observed expression at spot
- $\pi_k$ = proportion of cell type $k$
- $R_k$ = reference profile for cell type $k$

DiffBio uses a neural network to predict proportions with soft constraints:

```python
# Soft proportion constraints (sum to 1, non-negative)
proportions = jax.nn.softmax(logits / temperature, axis=-1)
```

### Training for Deconvolution

```python
def deconv_loss(deconv, spatial_expr, reference):
    """Train deconvolution model."""
    data = {
        "spatial_expression": spatial_expr,
        "reference_profiles": reference,
    }
    result, _, _ = deconv.apply(data, {}, None)

    # Reconstruction loss
    recon_loss = jnp.mean((result["reconstructed"] - spatial_expr) ** 2)

    # Entropy regularization (encourage sparse proportions)
    entropy = -jnp.sum(result["proportions"] * jnp.log(result["proportions"] + 1e-8))

    return recon_loss - 0.01 * entropy
```

## HiCContactAnalysis

Chromatin contact analysis for Hi-C and related 3C data.

### Quick Start

```python
from diffbio.operators.multiomics import HiCContactAnalysis, HiCConfig

# Configure Hi-C analysis
config = HiCConfig(
    resolution=10000,        # 10kb resolution
    hidden_dim=64,
    n_layers=4,
    distance_decay=True,
)

# Create operator
rngs = nnx.Rngs(42)
hic_analysis = HiCContactAnalysis(config, rngs=rngs)

# Analyze contact matrix
data = {"contact_matrix": hic_matrix}  # (n_bins, n_bins)
result, state, metadata = hic_analysis.apply(data, {}, None)

# Get results
normalized = result["normalized_contacts"]  # Distance-normalized
compartments = result["compartments"]        # A/B compartments
tad_boundaries = result["tad_boundaries"]    # TAD boundary scores
loops = result["loop_scores"]                # Loop/enhancer contacts
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | int | 10000 | Bin size in base pairs |
| `hidden_dim` | int | 64 | Network hidden dimension |
| `n_layers` | int | 4 | Number of network layers |
| `distance_decay` | bool | True | Model distance decay |

### Hi-C Analysis Components

#### Distance Normalization

Hi-C contacts decay with genomic distance. DiffBio learns the decay function:

```python
# Learned distance decay
expected_contacts = decay_network(distances)
normalized = observed / (expected_contacts + epsilon)
```

#### Compartment Calling

A/B compartments from correlation matrix:

```python
# PCA-style compartment calling
correlation = normalize(contact_matrix)
compartment_scores = svd_projection(correlation)
compartments = jax.nn.tanh(compartment_scores)  # Soft A/B
```

#### TAD Detection

Topologically Associating Domains from insulation scores:

```python
# Insulation score calculation
insulation = sliding_window_mean(contacts)
boundaries = jax.nn.sigmoid(-gradient(insulation) / temperature)
```

### Training Hi-C Models

```python
def hic_loss(hic_model, contact_matrix, known_loops):
    """Train Hi-C analysis model."""
    data = {"contact_matrix": contact_matrix}
    result, _, _ = hic_model.apply(data, {}, None)

    # Loop detection loss
    loop_loss = binary_cross_entropy(result["loop_scores"], known_loops)

    # TAD boundary loss (if labeled)
    # tad_loss = ...

    return loop_loss
```

## Multi-omics Integration

Combine multiple data modalities:

```python
from diffbio.operators.multiomics import SpatialDeconvolution
from diffbio.operators.epigenomics import ChromatinStateAnnotator

# Spatial transcriptomics + ATAC-seq
spatial_expr = ...  # Gene expression per spot
atac_signal = ...   # Chromatin accessibility

# Deconvolve cell types
deconv_result, _, _ = spatial_deconv.apply(
    {"spatial_expression": spatial_expr, "reference_profiles": ref},
    {}, None
)

# Annotate chromatin states
chrom_result, _, _ = chromatin_annotator.apply(
    {"histone_marks": atac_signal},
    {}, None
)

# Combine for multi-modal analysis
combined_features = jnp.concatenate([
    deconv_result["proportions"],
    chrom_result["state_probabilities"],
], axis=-1)
```

## Use Cases

| Application | Operator | Description |
|-------------|----------|-------------|
| Cell type mapping | SpatialDeconvolution | Spatial transcriptomics |
| Tissue architecture | SpatialDeconvolution | Understand tissue structure |
| Chromatin structure | HiCContactAnalysis | 3D genome organization |
| Enhancer-promoter | HiCContactAnalysis | Find regulatory contacts |
| TAD analysis | HiCContactAnalysis | Domain boundaries |

## Next Steps

- See [Epigenomics Operators](epigenomics.md) for chromatin analysis
- Explore [Single-Cell Operators](singlecell.md) for reference profiles
