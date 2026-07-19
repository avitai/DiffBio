"""Tests for k-mer spectrum featurization of DNA sequences."""

from __future__ import annotations

import numpy as np

from diffbio.sequences.kmer import kmer_dimension, kmer_featurize


def test_kmer_dimension_counts_canonical_and_full() -> None:
    assert kmer_dimension(2, canonical=False) == 16
    assert kmer_dimension(1, canonical=False) == 4
    # k=1 canonical: {A,C,G,T} collapse to {A/T, C/G} -> 2.
    assert kmer_dimension(1, canonical=True) == 2


def test_featurize_shape_and_frequency_normalization() -> None:
    sequences = ["ACGTACGT", "AAAAAAAA"]
    features = kmer_featurize(sequences, k=2, canonical=False)
    assert features.shape == (2, 16)
    # Each row sums to 1 (frequencies over the L-k+1 = 7 k-mers).
    np.testing.assert_allclose(features.sum(axis=1), [1.0, 1.0], atol=1e-6)


def test_featurize_counts_are_correct() -> None:
    # "AAAA" (k=2) -> three "AA" 2-mers, index of AA = 0.
    features = kmer_featurize(["AAAA"], k=2, canonical=False)
    assert features[0, 0] == np.float32(1.0)  # all mass on AA
    assert features[0, 1:].sum() == np.float32(0.0)


def test_canonical_collapses_reverse_complement() -> None:
    # "AA" and its reverse complement "TT" map to the same canonical k-mer.
    full = kmer_featurize(["AATT"], k=2, canonical=False)
    canonical = kmer_featurize(["AATT"], k=2, canonical=True)
    assert canonical.shape[1] == kmer_dimension(2, canonical=True)
    # AATT (k=2) -> AA, AT, TT ; AA+TT collapse -> canonical mass concentrates.
    assert np.count_nonzero(canonical[0]) <= np.count_nonzero(full[0])


def test_non_acgt_kmers_are_skipped() -> None:
    # A k-mer containing N is skipped; remaining valid k-mers still normalize to 1.
    features = kmer_featurize(["ACNGT"], k=2, canonical=False)
    assert bool(np.isfinite(features).all())
    np.testing.assert_allclose(features.sum(), 1.0, atol=1e-6)


def test_empty_or_short_sequence_is_all_zero() -> None:
    features = kmer_featurize(["A"], k=2, canonical=False)  # shorter than k
    np.testing.assert_array_equal(features[0], np.zeros(16, dtype=np.float32))
