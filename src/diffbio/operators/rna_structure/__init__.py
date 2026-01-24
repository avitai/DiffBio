"""RNA secondary structure prediction operators for DiffBio.

This module provides differentiable operators for RNA secondary structure
prediction, following the McCaskill partition function algorithm for
computing base pair probabilities.

Operators:
    DifferentiableRNAFold: McCaskill-style RNA folding with base pair probs

Factory Functions:
    create_rna_fold_predictor: Create RNA fold predictor with defaults

References:
    McCaskill (1990). The equilibrium partition function and base pair
    binding probabilities for RNA secondary structure.

    Matthies et al. (2024). Differentiable partition function calculation
    for RNA. Nucleic Acids Research.
"""

from diffbio.operators.rna_structure.rna_folding import (
    DifferentiableRNAFold,
    RNAFoldConfig,
    compute_base_pair_probabilities,
    compute_pair_energy_matrix,
    create_rna_fold_predictor,
)

__all__ = [
    "DifferentiableRNAFold",
    "RNAFoldConfig",
    "create_rna_fold_predictor",
    "compute_pair_energy_matrix",
    "compute_base_pair_probabilities",
]
