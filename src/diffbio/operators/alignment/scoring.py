"""Scoring matrices for sequence alignment.

This module provides pre-defined scoring matrices and utilities for
creating custom scoring matrices for DNA, RNA, and protein alignment.
"""

import functools
import logging
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float

logger = logging.getLogger(__name__)


class ScoringMatrix(NamedTuple):
    """Scoring matrix with metadata.

    Attributes:
        matrix: The scoring matrix array.
        alphabet: The alphabet string (e.g., "ACGT" for DNA).
        name: Optional name for the matrix.
    """

    matrix: Float[Array, "alphabet alphabet"]
    alphabet: str
    name: str = ""


def create_dna_scoring_matrix(
    match: float = 2.0,
    mismatch: float = -1.0,
) -> Float[Array, "4 4"]:
    """Create a simple DNA scoring matrix.

    Args:
        match: Score for matching nucleotides (diagonal).
        mismatch: Score for mismatching nucleotides (off-diagonal).

    Returns:
        4x4 scoring matrix for DNA (A, C, G, T order).
    """
    # Create identity matrix scaled by match score
    identity = jnp.eye(4) * match
    # Create off-diagonal with mismatch score
    off_diag = (jnp.ones((4, 4)) - jnp.eye(4)) * mismatch
    return identity + off_diag


def create_rna_scoring_matrix(
    match: float = 2.0,
    mismatch: float = -1.0,
) -> Float[Array, "4 4"]:
    """Create a simple RNA scoring matrix.

    Args:
        match: Score for matching nucleotides (diagonal).
        mismatch: Score for mismatching nucleotides (off-diagonal).

    Returns:
        4x4 scoring matrix for RNA (A, C, G, U order).
    """
    # RNA uses same simple scoring as DNA
    return create_dna_scoring_matrix(match, mismatch)


@functools.cache
def get_dna_simple() -> Float[Array, "4 4"]:
    """Get pre-defined DNA scoring matrix (simple match/mismatch).

    Returns:
        4x4 scoring matrix with match=2.0, mismatch=-1.0.
    """
    return create_dna_scoring_matrix(match=2.0, mismatch=-1.0)


@functools.cache
def get_rna_simple() -> Float[Array, "4 4"]:
    """Get pre-defined RNA scoring matrix (simple match/mismatch).

    Returns:
        4x4 scoring matrix with match=2.0, mismatch=-1.0.
    """
    return create_rna_scoring_matrix(match=2.0, mismatch=-1.0)


# BLOSUM62 scoring matrix for proteins (20 amino acids)
# Standard BLOSUM62 matrix values
# Amino acid order: A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
_BLOSUM62_VALUES = [
    # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
]


@functools.cache
def get_blosum62() -> Float[Array, "20 20"]:
    """Get BLOSUM62 scoring matrix for protein alignment.

    Returns:
        20x20 scoring matrix for 20 standard amino acids.
    """
    return jnp.array(_BLOSUM62_VALUES, dtype=jnp.float32)


# Amino acid alphabet for BLOSUM62
PROTEIN_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
