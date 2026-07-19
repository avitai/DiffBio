"""Frozen, fit-once dimensionality reductions with a shared modality-agnostic interface.

These pair with the differentiable learnable projection: a frozen reduction supplies the
loadings that initialize a learnable projection, so a task model can be trained against
either the frozen features or a jointly-optimized projection of the same features.
"""

from diffbio.reductions.base import FrozenReduction
from diffbio.reductions.pca_reduction import PCAReduction, fit_pca_reduction
from diffbio.reductions.tfidf_reduction import TFIDFReduction, fit_tfidf_reduction

__all__ = [
    "FrozenReduction",
    "PCAReduction",
    "fit_pca_reduction",
    "TFIDFReduction",
    "fit_tfidf_reduction",
]
