"""CRISPR guide design operators.

This module provides differentiable operators for CRISPR guide RNA
design and scoring, including on-target efficiency prediction.
"""

from diffbio.operators.crispr.guide_scoring import (
    CRISPRScorerConfig,
    DifferentiableCRISPRScorer,
    create_crispr_scorer,
)

__all__ = [
    "CRISPRScorerConfig",
    "DifferentiableCRISPRScorer",
    "create_crispr_scorer",
]
