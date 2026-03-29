"""Epigenomics operators for differentiable ChIP-seq and ATAC-seq analysis.

This module provides differentiable operators for epigenomic data analysis,
including peak calling and chromatin state annotation.
"""

from diffbio.operators.epigenomics.chromatin_state import (
    ChromatinStateAnnotator,
    ChromatinStateConfig,
)
from diffbio.operators.epigenomics.fno_peak_calling import (
    FNOPeakCaller,
    FNOPeakCallerConfig,
)
from diffbio.operators.epigenomics.peak_calling import (
    DifferentiablePeakCaller,
    PeakCallerConfig,
)

__all__ = [
    "ChromatinStateAnnotator",
    "ChromatinStateConfig",
    "DifferentiablePeakCaller",
    "FNOPeakCaller",
    "FNOPeakCallerConfig",
    "PeakCallerConfig",
]
