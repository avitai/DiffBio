"""Differentiable normalization and embedding operators.

This module provides operators for:

- VAE-based count normalization (scVI-style)
- Sequence embedding with learned representations
- Differentiable dimensionality reduction

All operators maintain gradient flow for end-to-end training.
"""

from diffbio.operators.normalization.embedding import (
    SequenceEmbedding,
    SequenceEmbeddingConfig,
)
from diffbio.operators.normalization.umap import (
    DifferentiableUMAP,
    UMAPConfig,
)
from diffbio.operators.normalization.vae_normalizer import (
    VAENormalizer,
    VAENormalizerConfig,
)

__all__ = [
    # VAE Normalization
    "VAENormalizerConfig",
    "VAENormalizer",
    # Sequence Embedding
    "SequenceEmbeddingConfig",
    "SequenceEmbedding",
    # UMAP Dimensionality Reduction
    "UMAPConfig",
    "DifferentiableUMAP",
]
