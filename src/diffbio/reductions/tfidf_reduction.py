"""Frozen TF-IDF + truncated-SVD (LSI) reduction for scATAC-seq accessibility.

This is the standard scATAC dimensionality-reduction frontend (Signac ``RunTFIDF``
method 1 followed by ``RunSVD``): term-frequency-inverse-document-frequency weighting of
the cell x bin count matrix, ``log(1 + TF*IDF*1e4)``, then a truncated SVD whose first
component -- which correlates with sequencing depth -- is dropped. It implements the
:class:`FrozenReduction` interface so the frozen-vs-learnable-projection comparison ports
to scATAC by swapping only the featurizer: the SVD loadings (depth component removed)
initialize a learnable projection, and ``scaled``/``transform`` feed the downstream model.

Unlike the PCA reduction, the SVD runs on the TF-IDF matrix directly with no mean
centering (so ``pca_mean`` is zero), and ``scaled`` preserves sparsity -- ``log1p`` maps
structural zeros to zero -- so the large cell x bin matrix never densifies here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

_TFIDF_SCALE = 1.0e4
"""Signac's fixed TF-IDF scale factor inside ``log(1 + TF*IDF*scale)``."""


@dataclass(frozen=True, slots=True)
class TFIDFReduction:
    """A fitted TF-IDF + truncated-SVD (LSI) reduction over a cell x bin matrix.

    Attributes:
        idf: ``(n_bins,)`` inverse-document-frequency fitted on the training split.
        pca_mean: ``(n_bins,)`` zeros -- LSI applies SVD to the TF-IDF with no centering.
        loadings: ``(n_bins, k)`` right singular vectors, depth component 0 dropped.
        eigenvalues: ``(k,)`` singular values for the kept components, descending.
    """

    idf: np.ndarray
    pca_mean: np.ndarray
    loadings: np.ndarray
    eigenvalues: np.ndarray

    def scaled(self, features: np.ndarray | sp.spmatrix) -> sp.csr_matrix:
        """Return the sparse TF-IDF representation ``log(1 + TF*IDF*1e4)``.

        Term frequency is per-cell (each cell's counts sum to one); the inverse document
        frequency is the fitted training ``idf``, so held-out cells reuse it unchanged.
        """
        counts = sp.csr_matrix(features, dtype=np.float32)
        row_totals = np.asarray(counts.sum(axis=1)).ravel()
        row_totals[row_totals == 0.0] = 1.0
        term_frequency = sp.diags(1.0 / row_totals) @ counts
        term_frequency = term_frequency.tocsr()
        weighted = term_frequency.data * self.idf[term_frequency.indices] * _TFIDF_SCALE
        return sp.csr_matrix(
            (np.log1p(weighted), term_frequency.indices, term_frequency.indptr),
            shape=term_frequency.shape,
        )

    def transform(self, features: np.ndarray | sp.spmatrix) -> np.ndarray:
        """Project ``features`` onto the fitted LSI components (depth component removed)."""
        projected = self.scaled(features) @ self.loadings
        return np.asarray(projected, dtype=np.float32)


def fit_tfidf_reduction(features: np.ndarray | sp.spmatrix, n_components: int) -> TFIDFReduction:
    """Fit TF-IDF weighting and a truncated SVD on ``features`` (the training split).

    The SVD keeps ``n_components`` components *after* discarding the depth-correlated
    first component, so ``n_components + 1`` singular vectors are computed and the first
    is dropped.

    Args:
        features: ``(n_cells, n_bins)`` cell x bin accessibility counts (sparse or dense).
        n_components: Number of LSI components to keep after dropping the depth component.

    Returns:
        The fitted :class:`TFIDFReduction`.

    Raises:
        ValueError: If ``n_components`` is not strictly positive.
    """
    if n_components <= 0:
        raise ValueError(f"n_components must be strictly positive, got {n_components}")
    from sklearn.decomposition import TruncatedSVD  # noqa: PLC0415

    counts = sp.csr_matrix(features, dtype=np.float32)
    n_cells = counts.shape[0]
    column_totals = np.asarray(counts.sum(axis=0)).ravel()
    column_totals[column_totals == 0.0] = 1.0
    idf = (n_cells / column_totals).astype(np.float32)

    reduction = TFIDFReduction(
        idf=idf,
        pca_mean=np.zeros(counts.shape[1], dtype=np.float32),
        loadings=np.empty((counts.shape[1], 0), dtype=np.float32),
        eigenvalues=np.empty((0,), dtype=np.float32),
    )
    tfidf = reduction.scaled(counts)

    n_output = min(n_components + 1, min(tfidf.shape[0], tfidf.shape[1]) - 1)
    svd = TruncatedSVD(n_components=n_output, algorithm="randomized", random_state=0)
    svd.fit(tfidf)
    # Drop component 0 (depth-correlated); keep the rest as the LSI subspace.
    return TFIDFReduction(
        idf=idf,
        pca_mean=reduction.pca_mean,
        loadings=np.asarray(svd.components_[1:].T, dtype=np.float32),
        eigenvalues=np.asarray(svd.singular_values_[1:], dtype=np.float32),
    )
