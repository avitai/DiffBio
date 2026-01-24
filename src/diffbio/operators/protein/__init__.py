"""Protein structure operators for DiffBio.

This module provides differentiable operators for protein structure analysis,
including secondary structure prediction using the DSSP algorithm.

Operators:
    DifferentiableSecondaryStructure: PyDSSP-style secondary structure prediction
        with continuous hydrogen bond matrix for gradient-based optimization.

Example:
    >>> from diffbio.operators.protein import create_secondary_structure_predictor
    >>> predictor = create_secondary_structure_predictor()
    >>> result, _, _ = predictor.apply({"coordinates": coords}, {}, None)
    >>> ss_probs = result["ss_onehot"]  # (batch, length, 3)
"""

from diffbio.operators.protein.secondary_structure import (
    DifferentiableSecondaryStructure,
    SecondaryStructureConfig,
    compute_hydrogen_position,
    create_secondary_structure_predictor,
)

__all__ = [
    "DifferentiableSecondaryStructure",
    "SecondaryStructureConfig",
    "compute_hydrogen_position",
    "create_secondary_structure_predictor",
]
