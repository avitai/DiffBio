# Multi-omics Operators API

Differentiable operators for multi-omics analysis including spatial transcriptomics and Hi-C.

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

## Usage Examples

### Spatial Deconvolution

```python
from flax import nnx
from diffbio.operators.multiomics import SpatialDeconvolution, SpatialDeconvConfig

config = SpatialDeconvConfig(n_cell_types=10, n_genes=2000)
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
from diffbio.operators.multiomics import HiCContactAnalysis, HiCConfig

config = HiCConfig(resolution=10000, hidden_dim=64)
hic_analysis = HiCContactAnalysis(config, rngs=nnx.Rngs(42))

data = {"contact_matrix": hic_matrix}  # (n_bins, n_bins)
result, _, _ = hic_analysis.apply(data, {}, None)
compartments = result["compartments"]
tad_boundaries = result["tad_boundaries"]
```
