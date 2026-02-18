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
    PROTEIN_ALPHABET,
    ScoringMatrix,
    create_dna_scoring_matrix,
    create_rna_scoring_matrix,
    get_blosum62,
    get_dna_simple,
    get_rna_simple,
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
    "PROTEIN_ALPHABET",
    "ScoringMatrix",
    "create_dna_scoring_matrix",
    "create_rna_scoring_matrix",
    "get_blosum62",
    "get_dna_simple",
    "get_rna_simple",
    # Smith-Waterman
    "AlignmentResult",
    "SmithWatermanConfig",
    "SmoothSmithWaterman",
    # Soft MSA
    "SoftProgressiveMSA",
    "SoftProgressiveMSAConfig",
]
