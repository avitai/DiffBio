"""Differentiable sequence alignment operators.

This module provides differentiable implementations of sequence alignment
algorithms including smooth Smith-Waterman for local alignment.
"""

from diffbio.operators.alignment.scoring import (
    BLOSUM62,
    DNA_SIMPLE,
    PROTEIN_ALPHABET,
    RNA_SIMPLE,
    ScoringMatrix,
    create_dna_scoring_matrix,
    create_rna_scoring_matrix,
)
from diffbio.operators.alignment.smith_waterman import (
    AlignmentResult,
    SmithWatermanConfig,
    SmoothSmithWaterman,
)


__all__ = [
    # Scoring
    "BLOSUM62",
    "DNA_SIMPLE",
    "PROTEIN_ALPHABET",
    "RNA_SIMPLE",
    "ScoringMatrix",
    "create_dna_scoring_matrix",
    "create_rna_scoring_matrix",
    # Smith-Waterman
    "AlignmentResult",
    "SmithWatermanConfig",
    "SmoothSmithWaterman",
]
