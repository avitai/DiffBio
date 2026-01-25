# Multi-omics Operators

DiffBio provides differentiable operators for multi-omics data analysis, including spatial transcriptomics and chromatin conformation.

<span class="operator-multiomics">Multi-omics</span> <span class="diff-high">Fully Differentiable</span>

## Overview

Multi-omics operators enable end-to-end optimization of:

- **SpatialDeconvolution**: Cell type deconvolution for spatial transcriptomics
- **HiCContactAnalysis**: Chromatin contact analysis for Hi-C data
- **DifferentiableSpatialGeneDetector**: SpatialDE-style spatial gene detection

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

## DifferentiableSpatialGeneDetector

SpatialDE-style differentiable spatial gene detection using Gaussian processes. Identifies spatially variable genes by decomposing expression variability into spatial and non-spatial components.

### Quick Start

```python
from flax import nnx
from diffbio.operators.multiomics import (
    DifferentiableSpatialGeneDetector,
    SpatialGeneDetectorConfig,
)

# Configure detector
config = SpatialGeneDetectorConfig(
    n_genes=2000,
    lengthscale=1.0,          # RBF kernel lengthscale
    variance=1.0,             # Signal variance
    noise_variance=0.1,       # Noise variance
    hidden_dims=[64, 32],     # Smoothing network layers
    temperature=1.0,          # Classification temperature
    pvalue_threshold=0.05,    # Spatial gene threshold
)

# Create operator
rngs = nnx.Rngs(42)
detector = DifferentiableSpatialGeneDetector(config, rngs=rngs)

# Apply spatial gene detection
data = {
    "spatial_coords": coords,       # (n_spots, 2) - Spatial coordinates
    "expression": expression,        # (n_spots, n_genes) - Gene expression
    "total_counts": total_counts,   # (n_spots,) - Optional normalization
}
result, state, metadata = detector.apply(data, {}, None)

# Get results
fsv = result["fsv"]                      # Fraction of Spatial Variance
spatial_pvalues = result["spatial_pvalues"]  # P-values for spatial patterns
is_spatial = result["is_spatial"]        # Soft spatial gene indicator
smoothed = result["smoothed_expression"] # GP-smoothed expression
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_genes` | int | 2000 | Number of genes to analyze |
| `lengthscale` | float | 1.0 | RBF kernel lengthscale (spatial range) |
| `variance` | float | 1.0 | Signal variance (σ²_s) |
| `noise_variance` | float | 0.1 | Noise variance (σ²_e) |
| `hidden_dims` | list[int] | [64, 32] | Smoothing network dimensions |
| `temperature` | float | 1.0 | Temperature for soft thresholding |
| `pvalue_threshold` | float | 0.05 | Threshold for spatial classification |
| `learnable_kernel` | bool | True | Whether kernel params are learnable |

### Spatial Variance Model

The model decomposes gene expression as:

$$y = f(x) + \epsilon$$

Where:

- $f(x) \sim \mathcal{GP}(0, K)$ is the spatial component with RBF kernel
- $\epsilon \sim \mathcal{N}(0, \sigma^2_e)$ is the non-spatial noise

The **Fraction of Spatial Variance (FSV)** quantifies spatial structure:

$$\text{FSV} = \frac{\sigma^2_s}{\sigma^2_s + \sigma^2_e}$$

### RBF Kernel

The squared exponential (RBF) kernel models spatial covariance:

$$K(x_1, x_2) = \sigma^2_s \exp\left(-\frac{||x_1 - x_2||^2}{2\ell^2}\right)$$

Where:

- $\sigma^2_s$ = signal variance (spatial component strength)
- $\ell$ = lengthscale (characteristic spatial range)

### Training for Spatial Detection

```python
def spatial_loss(detector, data):
    """Train spatial gene detector."""
    result, _, _ = detector.apply(data, {}, None)

    # Maximize spatial variance detection
    fsv_loss = -result["fsv"].mean()

    # Smoothing quality (reconstruction)
    smooth_loss = jnp.mean((result["smoothed_expression"] - data["expression"]) ** 2)

    return fsv_loss + 0.1 * smooth_loss
```

### Interpreting Results

```python
# Identify spatially variable genes
spatial_genes = result["is_spatial"] > 0.5
n_spatial = spatial_genes.sum()

# Get top spatial genes by FSV
top_spatial_idx = jnp.argsort(result["fsv"])[::-1][:100]

# Visualize smoothed expression
import matplotlib.pyplot as plt
gene_idx = top_spatial_idx[0]
plt.scatter(
    data["spatial_coords"][:, 0],
    data["spatial_coords"][:, 1],
    c=result["smoothed_expression"][:, gene_idx],
)
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
| Spatial gene detection | DifferentiableSpatialGeneDetector | Find spatially variable genes |
| Spatial patterns | DifferentiableSpatialGeneDetector | Identify spatial expression patterns |

## Next Steps

- See [Epigenomics Operators](epigenomics.md) for chromatin analysis
- Explore [Single-Cell Operators](singlecell.md) for reference profiles
