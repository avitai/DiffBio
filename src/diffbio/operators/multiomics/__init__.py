"""Multi-omics analysis operators for differentiable integration.

This module provides differentiable operators for multi-omics data analysis:
- SpatialDeconvolution: Cell type deconvolution for spatial transcriptomics
- HiCContactAnalysis: Chromatin contact analysis for Hi-C data
- DifferentiableSpatialGeneDetector: SpatialDE-style spatial gene detection
"""

from diffbio.operators.multiomics.hic_contact import (
    HiCContactAnalysis,
    HiCContactAnalysisConfig,
)
from diffbio.operators.multiomics.spatial_deconvolution import (
    SpatialDeconvolution,
    SpatialDeconvolutionConfig,
)
from diffbio.operators.multiomics.spatial_gene_detection import (
    DifferentiableSpatialGeneDetector,
    SpatialGeneDetectorConfig,
    create_spatial_gene_detector,
)

__all__ = [
    "SpatialDeconvolution",
    "SpatialDeconvolutionConfig",
    "HiCContactAnalysis",
    "HiCContactAnalysisConfig",
    "DifferentiableSpatialGeneDetector",
    "SpatialGeneDetectorConfig",
    "create_spatial_gene_detector",
]
