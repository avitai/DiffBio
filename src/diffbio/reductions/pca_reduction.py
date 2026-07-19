"""Modality-agnostic frozen PCA reduction over an arbitrary feature matrix.

Fits per-feature standardization plus an exact PCA once on a training feature matrix --
gene expression, DNA k-mer spectra, scATAC peak features, and so on -- and applies the
identical transform to any split. It implements the :class:`FrozenReduction` interface,
so the frozen-vs-learnable-projection comparison ports to a new modality by swapping only
the featurizer: the loadings initialize a learnable projection, the eigenvalues drive the
soft-dimensionality operator, and ``scaled``/``transform`` feed the downstream model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SCALE_CLIP = 10.0
"""Symmetric clip bound applied after per-feature standardization."""


@dataclass(frozen=True, slots=True)
class PCAReduction:
    """A fitted standardize-then-PCA reduction over a feature matrix.

    Attributes:
        mean: ``(n_features,)`` per-feature mean for standardization.
        std: ``(n_features,)`` per-feature standard deviation (zeros replaced by one).
        pca_mean: ``(n_features,)`` mean removed by PCA before projection.
        loadings: ``(n_features, k)`` principal-component loadings.
        eigenvalues: ``(k,)`` explained variance per component, descending.
    """

    mean: np.ndarray
    std: np.ndarray
    pca_mean: np.ndarray
    loadings: np.ndarray
    eigenvalues: np.ndarray

    def scaled(self, features: np.ndarray) -> np.ndarray:
        """Return the clipped, standardized features (the pre-PCA representation)."""
        standardized = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
        return np.clip(standardized, -_SCALE_CLIP, _SCALE_CLIP)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Project ``features`` onto the fitted principal components."""
        projected = (self.scaled(features) - self.pca_mean) @ self.loadings
        return np.asarray(projected, dtype=np.float32)


def fit_pca_reduction(features: np.ndarray, n_components: int) -> PCAReduction:
    """Fit standardization and an exact PCA on ``features`` (the training split).

    Args:
        features: ``(n_samples, n_features)`` featurized training matrix.
        n_components: Number of principal components to keep.

    Returns:
        The fitted :class:`PCAReduction`.
    """
    from sklearn.decomposition import PCA  # noqa: PLC0415

    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    scaled = np.clip((features - mean) / std, -_SCALE_CLIP, _SCALE_CLIP)

    n_output = min(n_components, scaled.shape[0], scaled.shape[1])
    # Exact ("full") SVD, not sklearn's "auto" randomized path: on a near-degenerate
    # spectrum randomized and exact solvers pick different subspaces, and only the
    # exact solver is deterministic and matched to the differentiable PCA operator.
    pca = PCA(n_components=n_output, svd_solver="full").fit(scaled)
    return PCAReduction(
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        loadings=np.asarray(pca.components_.T, dtype=np.float32),
        eigenvalues=np.asarray(pca.explained_variance_, dtype=np.float32),
    )
