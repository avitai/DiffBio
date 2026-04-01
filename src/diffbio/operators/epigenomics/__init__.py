"""Epigenomics operators for differentiable ChIP-seq and ATAC-seq analysis.

This module provides differentiable operators for epigenomic data analysis,
including peak calling and chromatin state annotation.
"""

from diffbio.operators.epigenomics.chromatin_state import (
    ChromatinStateAnnotator,
    ChromatinStateConfig,
)
from diffbio.operators.epigenomics.contextual import (
    ContextualEpigenomicsConfig,
    ContextualEpigenomicsOperator,
    compute_contextual_epigenomics_loss,
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
    "ContextualEpigenomicsConfig",
    "ContextualEpigenomicsOperator",
    "DifferentiablePeakCaller",
    "FNOPeakCaller",
    "FNOPeakCallerConfig",
    "PeakCallerConfig",
    "compute_contextual_epigenomics_loss",
]
