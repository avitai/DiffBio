"""Differentiable bioinformatics operators for DiffBio.

This module provides differentiable operators for common bioinformatics
operations such as quality filtering, sequence alignment, variant calling,
epigenomics analysis, and RNA-seq processing.

All operators extend Datarax's OperatorModule for seamless integration.
"""

from diffbio.operators.quality_filter import (
    DifferentiableQualityFilter,
    QualityFilterConfig,
)

# Import submodules for convenient access
from diffbio.operators import alignment
from diffbio.operators import assembly
from diffbio.operators import epigenomics
from diffbio.operators import mapping
from diffbio.operators import multiomics
from diffbio.operators import normalization
from diffbio.operators import preprocessing
from diffbio.operators import rnaseq
from diffbio.operators import singlecell
from diffbio.operators import statistical
from diffbio.operators import variant

__all__ = [
    # Core quality filter
    "DifferentiableQualityFilter",
    "QualityFilterConfig",
    # Submodules
    "alignment",
    "assembly",
    "epigenomics",
    "mapping",
    "multiomics",
    "normalization",
    "preprocessing",
    "rnaseq",
    "singlecell",
    "statistical",
    "variant",
]
