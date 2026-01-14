"""RNA-seq operators for differentiable transcriptomics analysis.

This module provides differentiable operators for RNA-seq data analysis,
including splicing PSI calculation, motif discovery, and differential expression.
"""

from diffbio.operators.rnaseq.motif_discovery import (
    DifferentiableMotifDiscovery,
    MotifDiscoveryConfig,
)
from diffbio.operators.rnaseq.splicing_psi import (
    SplicingPSI,
    SplicingPSIConfig,
)

__all__ = [
    # Splicing PSI
    "SplicingPSI",
    "SplicingPSIConfig",
    # Motif Discovery
    "DifferentiableMotifDiscovery",
    "MotifDiscoveryConfig",
]
