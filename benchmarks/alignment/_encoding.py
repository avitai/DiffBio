"""Shared protein sequence encoding utilities for alignment benchmarks.

Provides one-hot encoding of protein sequences used by both the
pairwise and MSA alignment benchmarks.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from diffbio.operators.alignment import PROTEIN_ALPHABET

ALPHABET_INDEX: dict[str, int] = {
    aa: i for i, aa in enumerate(PROTEIN_ALPHABET)
}
ALPHABET_SIZE = len(PROTEIN_ALPHABET)


def onehot_encode_sequence(
    sequence: str,
    max_length: int,
    alphabet_size: int = ALPHABET_SIZE,
) -> jnp.ndarray:
    """One-hot encode a protein sequence, padded to max_length.

    Unknown residues (not in the standard 20 amino acid alphabet)
    are encoded as uniform distributions over all residues.

    Args:
        sequence: Amino acid sequence string (uppercase).
        max_length: Pad/truncate to this length.
        alphabet_size: Size of amino acid alphabet.

    Returns:
        One-hot array of shape (max_length, alphabet_size).
    """
    result = np.zeros((max_length, alphabet_size), dtype=np.float32)
    for i, aa in enumerate(sequence[:max_length]):
        idx = ALPHABET_INDEX.get(aa.upper())
        if idx is not None:
            result[i, idx] = 1.0
        else:
            result[i, :] = 1.0 / alphabet_size
    return jnp.array(result)
