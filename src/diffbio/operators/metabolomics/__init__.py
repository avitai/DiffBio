"""Differentiable metabolomics operators for DiffBio.

This module provides differentiable operators for metabolomics analysis,
including spectral similarity computation using deep learning.

Operators:
    DifferentiableSpectralSimilarity: MS2DeepScore-style Siamese network
        for predicting molecular structural similarity from MS/MS spectra.

Example:
    ```python
    from diffbio.operators.metabolomics import (
        DifferentiableSpectralSimilarity,
        SpectralSimilarityConfig,
        create_spectral_similarity,
        bin_spectrum,
    )
    # Create operator
    operator = create_spectral_similarity(n_bins=1000, embedding_dim=200)
    # Compute spectral embeddings
    spectra = jax.random.uniform(jax.random.PRNGKey(0), (10, 1000))
    result, _, _ = operator.apply({"spectra": spectra}, {}, None)
    embeddings = result["embeddings"]  # (10, 200)
    ```
"""

from diffbio.operators.metabolomics.spectral_similarity import (
    DifferentiableSpectralSimilarity,
    SpectralSimilarityConfig,
    bin_spectrum,
    create_spectral_similarity,
)

__all__ = [
    "DifferentiableSpectralSimilarity",
    "SpectralSimilarityConfig",
    "bin_spectrum",
    "create_spectral_similarity",
]
