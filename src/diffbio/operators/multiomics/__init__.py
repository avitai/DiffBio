"""Multi-omics analysis operators for differentiable integration.

This module provides differentiable operators for multi-omics data analysis:
- SpatialDeconvolution: Cell type deconvolution for spatial transcriptomics
- HiCContactAnalysis: Chromatin contact analysis for Hi-C data
"""

from diffbio.operators.multiomics.hic_contact import (
    HiCContactAnalysis,
    HiCContactAnalysisConfig,
)
from diffbio.operators.multiomics.spatial_deconvolution import (
    SpatialDeconvolution,
    SpatialDeconvolutionConfig,
)

__all__ = [
    "SpatialDeconvolution",
    "SpatialDeconvolutionConfig",
    "HiCContactAnalysis",
    "HiCContactAnalysisConfig",
]
