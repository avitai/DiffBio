"""k-mer spectrum featurization of DNA sequences.

Represents each sequence by the frequency of its overlapping length-``k`` subsequences
(the k-mer spectrum), a fixed, task-agnostic featurization analogous to gene-expression
counts in single cell. It is the frozen frontend for the sequence-classification case
study: the high-dimensional k-mer vector is reduced (by PCA or a learnable projection)
before a classifier, mirroring the highly-variable-gene + PCA reduction of scRNA-seq.

Canonical featurization collapses each k-mer with its reverse complement, making the
representation strand-invariant.
"""

from __future__ import annotations

import numpy as np

_BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
_COMPLEMENT = {0: 3, 1: 2, 2: 1, 3: 0}  # A<->T, C<->G


def _canonical_map(k: int) -> np.ndarray:
    """Return a ``(4**k,)`` array mapping each k-mer index to its canonical slot.

    Each k-mer and its reverse complement share the smaller of the two indices; the
    returned array is then densified to contiguous slots ``0..n_canonical-1``.
    """
    dimension = 4**k
    representative = np.arange(dimension, dtype=np.int64)
    for index in range(dimension):
        digits = [(index // (4**position)) % 4 for position in range(k)]
        rc_digits = [_COMPLEMENT[d] for d in reversed(digits)]
        rc_index = sum(d * (4**position) for position, d in enumerate(rc_digits))
        representative[index] = min(index, rc_index)
    unique = np.unique(representative)
    remap = {slot: new for new, slot in enumerate(unique.tolist())}
    return np.array([remap[slot] for slot in representative.tolist()], dtype=np.int64)


def kmer_dimension(k: int, *, canonical: bool) -> int:
    """Return the feature dimension of the k-mer spectrum.

    Args:
        k: k-mer length.
        canonical: Whether reverse-complement k-mers are collapsed.

    Returns:
        ``4**k`` for the full spectrum, or the number of canonical classes.
    """
    if canonical:
        return int(_canonical_map(k).max()) + 1
    return 4**k


def _sequence_indices(sequence: str, k: int) -> np.ndarray:
    """Return the valid (ACGT-only) k-mer indices of ``sequence`` in base-4."""
    codes = np.array([_BASE_TO_INDEX.get(base, -1) for base in sequence.upper()], dtype=np.int64)
    if codes.size < k:
        return np.empty(0, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(codes, k)
    valid = (windows >= 0).all(axis=1)
    powers = 4 ** np.arange(k, dtype=np.int64)
    return (windows[valid] * powers).sum(axis=1)


def kmer_featurize(sequences: list[str], k: int, *, canonical: bool = True) -> np.ndarray:
    """Return the ``(n_sequences, dimension)`` frequency-normalized k-mer spectrum.

    Args:
        sequences: DNA sequences (characters outside ``ACGT`` are skipped per k-mer).
        k: k-mer length.
        canonical: Collapse reverse-complement k-mers for strand invariance.

    Returns:
        A float32 matrix whose rows are k-mer frequencies (each valid row sums to 1;
        sequences shorter than ``k`` or with no valid k-mer are all-zero).
    """
    dimension = kmer_dimension(k, canonical=canonical)
    canonical_map = _canonical_map(k) if canonical else None
    features = np.zeros((len(sequences), dimension), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        indices = _sequence_indices(sequence, k)
        if indices.size == 0:
            continue
        if canonical_map is not None:
            indices = canonical_map[indices]
        counts = np.bincount(indices, minlength=dimension).astype(np.float32)
        features[row] = counts / counts.sum()
    return features
