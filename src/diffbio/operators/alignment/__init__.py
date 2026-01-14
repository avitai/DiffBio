"""Differentiable sequence alignment operators.

This module provides differentiable implementations of sequence alignment
algorithms including smooth Smith-Waterman for local alignment,
profile HMM search for domain detection, and soft progressive MSA.
"""

from diffbio.operators.alignment.profile_hmm import (
    ProfileHMMConfig,
    ProfileHMMSearch,
)
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
from diffbio.operators.alignment.soft_msa import (
    SoftProgressiveMSA,
    SoftProgressiveMSAConfig,
)


__all__ = [
    # Profile HMM
    "ProfileHMMConfig",
    "ProfileHMMSearch",
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
    # Soft MSA
    "SoftProgressiveMSA",
    "SoftProgressiveMSAConfig",
]
