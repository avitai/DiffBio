# Multi-omics Operators API

Differentiable operators for multi-omics analysis including spatial transcriptomics, Hi-C, and spatial gene detection.

## SpatialDeconvolution

::: diffbio.operators.multiomics.spatial_deconvolution.SpatialDeconvolution
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## SpatialDeconvolutionConfig

::: diffbio.operators.multiomics.spatial_deconvolution.SpatialDeconvolutionConfig
    options:
      show_root_heading: true
      members: []

## HiCContactAnalysis

::: diffbio.operators.multiomics.hic_contact.HiCContactAnalysis
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## HiCContactAnalysisConfig

::: diffbio.operators.multiomics.hic_contact.HiCContactAnalysisConfig
    options:
      show_root_heading: true
      members: []

## DifferentiableSpatialGeneDetector

::: diffbio.operators.multiomics.spatial_gene_detection.DifferentiableSpatialGeneDetector
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply
        - compute_kernel
        - compute_spatial_variance
        - compute_pvalues

## SpatialGeneDetectorConfig

::: diffbio.operators.multiomics.spatial_gene_detection.SpatialGeneDetectorConfig
    options:
      show_root_heading: true
      members: []

## create_spatial_gene_detector

::: diffbio.operators.multiomics.spatial_gene_detection.create_spatial_gene_detector
    options:
      show_root_heading: true
      show_source: false

## DifferentiableMultiOmicsVAE

::: diffbio.operators.multiomics.multiomics_vae.DifferentiableMultiOmicsVAE
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - apply

## MultiOmicsVAEConfig

::: diffbio.operators.multiomics.multiomics_vae.MultiOmicsVAEConfig
    options:
      show_root_heading: true
      members: []

## Usage Examples

### Spatial Deconvolution

```python
from flax import nnx
from diffbio.operators.multiomics import SpatialDeconvolution, SpatialDeconvolutionConfig

config = SpatialDeconvolutionConfig(n_cell_types=10, n_genes=2000)
deconv = SpatialDeconvolution(config, rngs=nnx.Rngs(42))

data = {
    "spatial_expression": spot_expression,     # (n_spots, n_genes)
    "reference_profiles": cell_type_profiles,  # (n_cell_types, n_genes)
}
result, _, _ = deconv.apply(data, {}, None)
proportions = result["proportions"]
```

### Hi-C Contact Analysis

```python
from diffbio.operators.multiomics import HiCContactAnalysis, HiCContactAnalysisConfig

config = HiCContactAnalysisConfig(n_bins=1000, hidden_dim=64)
hic_analysis = HiCContactAnalysis(config, rngs=nnx.Rngs(42))

data = {"contact_matrix": hic_matrix}  # (n_bins, n_bins)
result, _, _ = hic_analysis.apply(data, {}, None)
compartments = result["compartments"]
tad_boundaries = result["tad_boundaries"]
```

### Spatial Gene Detection

```python
from flax import nnx
from diffbio.operators.multiomics import (
    DifferentiableSpatialGeneDetector,
    SpatialGeneDetectorConfig,
    create_spatial_gene_detector,
)

# Using config
config = SpatialGeneDetectorConfig(
    n_genes=2000,
    lengthscale=1.0,
    variance=1.0,
    pvalue_threshold=0.05,
)
detector = DifferentiableSpatialGeneDetector(config, rngs=nnx.Rngs(42))

# Or using factory function
detector = create_spatial_gene_detector(
    n_genes=2000,
    lengthscale=1.0,
)

# Apply spatial gene detection
data = {
    "spatial_coords": coords,        # (n_spots, 2)
    "expression": expression,        # (n_spots, n_genes)
    "total_counts": total_counts,    # (n_spots,) optional
}
result, _, _ = detector.apply(data, {}, None)

# Get spatial gene results
fsv = result["fsv"]                      # Fraction of Spatial Variance
is_spatial = result["is_spatial"]        # Soft spatial indicator
smoothed = result["smoothed_expression"] # GP-smoothed expression
```
