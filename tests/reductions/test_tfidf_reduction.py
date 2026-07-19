"""Tests for the TF-IDF + truncated-SVD (LSI) frozen reduction for scATAC."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from diffbio.reductions import FrozenReduction
from diffbio.reductions.tfidf_reduction import TFIDFReduction, fit_tfidf_reduction

_N_CELLS, _N_BINS = 12, 16


def _counts() -> sp.csr_matrix:
    """A fixed, moderately sized cell x bin count matrix (some structural zeros)."""
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 5, size=(_N_CELLS, _N_BINS)).astype(np.float32)
    dense[dense < 2] = 0.0  # inject sparsity but keep every row/column non-empty
    dense[0, 0] = dense[-1, -1] = 3.0
    return sp.csr_matrix(dense)


# --- TF-IDF formula (Signac method 1: log(1 + TF*IDF*1e4)) ----------------------


def test_scaled_matches_signac_tfidf_formula() -> None:
    counts = _counts()
    reduction = fit_tfidf_reduction(counts, n_components=3)
    scaled = np.asarray(reduction.scaled(counts).todense())

    dense = np.asarray(counts.todense())
    tf = dense / dense.sum(axis=1, keepdims=True)
    idf = dense.shape[0] / dense.sum(axis=0)
    expected = np.log1p(tf * idf * 1.0e4)
    np.testing.assert_allclose(scaled, expected, rtol=1e-5)


def test_idf_uses_column_totals_and_cell_count() -> None:
    counts = _counts()
    reduction = fit_tfidf_reduction(counts, n_components=3)
    dense = np.asarray(counts.todense())
    expected_idf = dense.shape[0] / dense.sum(axis=0)
    np.testing.assert_allclose(reduction.idf, expected_idf, rtol=1e-6)


def test_scaled_preserves_sparsity() -> None:
    counts = _counts()
    scaled = fit_tfidf_reduction(counts, n_components=3).scaled(counts)
    assert sp.issparse(scaled)
    # zero counts stay exactly zero (log1p(0) = 0), so no fill-in.
    assert scaled.nnz == counts.nnz


# --- SVD / component dropping ---------------------------------------------------


def test_drops_first_component_and_keeps_k() -> None:
    reduction = fit_tfidf_reduction(_counts(), n_components=4)
    # loadings keep k=4 components (the depth-correlated component 0 is dropped).
    assert reduction.loadings.shape == (_N_BINS, 4)
    assert reduction.eigenvalues.shape == (4,)


def test_eigenvalues_descending() -> None:
    reduction = fit_tfidf_reduction(_counts(), n_components=5)
    assert np.all(np.diff(reduction.eigenvalues) <= 1e-4)


def test_pca_mean_is_zero() -> None:
    # LSI runs SVD on the TF-IDF directly (no centering), so pca_mean must be zeros.
    reduction = fit_tfidf_reduction(_counts(), n_components=3)
    np.testing.assert_array_equal(reduction.pca_mean, np.zeros(_N_BINS, dtype=np.float32))


# --- transform ------------------------------------------------------------------


def test_transform_shape_and_equals_scaled_matmul_loadings() -> None:
    counts = _counts()
    reduction = fit_tfidf_reduction(counts, n_components=3)
    projected = reduction.transform(counts)
    assert projected.shape == (_N_CELLS, 3)
    expected = np.asarray(reduction.scaled(counts).todense()) @ reduction.loadings
    np.testing.assert_allclose(projected, expected, rtol=1e-5)


def test_transform_uses_training_idf_on_new_cells() -> None:
    # scaled() on a held-out cell must reuse the fitted IDF, not recompute it.
    train = _counts()
    reduction = fit_tfidf_reduction(train, n_components=3)
    new_dense = np.zeros((1, _N_BINS), dtype=np.float32)
    new_dense[0, 1] = new_dense[0, 3] = 2.0
    scaled_new = np.asarray(reduction.scaled(sp.csr_matrix(new_dense)).todense())
    tf = new_dense / new_dense.sum()
    expected = np.log1p(tf * reduction.idf * 1.0e4)
    np.testing.assert_allclose(scaled_new, expected, rtol=1e-5)


# --- interface / dense input ----------------------------------------------------


def test_satisfies_frozen_reduction_interface() -> None:
    reduction = fit_tfidf_reduction(_counts(), n_components=3)
    assert isinstance(reduction, FrozenReduction)
    assert isinstance(reduction, TFIDFReduction)


def test_accepts_dense_input() -> None:
    dense = np.asarray(_counts().todense())
    reduction = fit_tfidf_reduction(dense, n_components=3)
    assert reduction.transform(dense).shape == (_N_CELLS, 3)


def test_rejects_non_positive_components() -> None:
    with pytest.raises(ValueError, match="n_components"):
        fit_tfidf_reduction(_counts(), n_components=0)
