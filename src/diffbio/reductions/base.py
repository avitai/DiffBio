"""The frozen-reduction interface shared across modalities.

A frozen reduction is fit once on a training feature matrix and then applies the
identical transform to any split. Keeping the interface modality-agnostic lets the
frozen-vs-learnable-projection comparison port between single-cell expression, DNA
k-mer spectra, scATAC peak features, and so on by swapping only the featurizer and the
concrete reduction, while the learnable-projection arm and trainer stay unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrozenReduction(Protocol):
    """A fitted, frozen dimensionality reduction over a feature matrix.

    Implementations expose the loadings (so a learnable projection can be initialized
    from them), the PCA mean removed before projection, the per-component eigenvalues,
    and the ``scaled``/``transform`` application methods.
    """

    loadings: np.ndarray
    pca_mean: np.ndarray
    eigenvalues: np.ndarray

    def scaled(self, features: np.ndarray) -> np.ndarray:
        """Return the standardized pre-projection representation of ``features``."""
        ...

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Project ``features`` onto the fitted components."""
        ...
