"""Differentiable normalization and embedding operators.

This module provides operators for:

- VAE-based count normalization (scVI-style)
- Sequence embedding with learned representations
- Differentiable dimensionality reduction (UMAP, PHATE)

All operators maintain gradient flow for end-to-end training.
"""

from diffbio.operators.normalization.differentiable_pca import (
    DifferentiablePCA,
    DifferentiablePCAConfig,
)
from diffbio.operators.normalization.embedding import (
    SequenceEmbedding,
    SequenceEmbeddingConfig,
)
from diffbio.operators.normalization.learnable_normalization import (
    LearnableNormalization,
    LearnableNormalizationConfig,
)
from diffbio.operators.normalization.learnable_orthogonal_projection import (
    LearnableOrthogonalProjection,
    LearnableOrthogonalProjectionConfig,
)
from diffbio.operators.normalization.learnable_projection import (
    LearnableProjection,
    LearnableProjectionConfig,
)
from diffbio.operators.normalization.matrix_free_pca import (
    MatrixFreePCA,
    MatrixFreePCAConfig,
)
from diffbio.operators.normalization.phate import (
    DifferentiablePHATE,
    PHATEConfig,
)
from diffbio.operators.normalization.scaling import (
    DifferentiableScaler,
    ScalerConfig,
)
from diffbio.operators.normalization.soft_pca import (
    SoftComponentSelection,
    SoftComponentSelectionConfig,
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
    # PCA Dimensionality Reduction
    "DifferentiablePCAConfig",
    "DifferentiablePCA",
    "MatrixFreePCAConfig",
    "MatrixFreePCA",
    # Learnable count normalization
    "LearnableNormalizationConfig",
    "LearnableNormalization",
    # Learnable projection (task-aware dimensionality reduction)
    "LearnableProjectionConfig",
    "LearnableProjection",
    "LearnableOrthogonalProjection",
    "LearnableOrthogonalProjectionConfig",
    # Soft component selection (learnable PCA dimensionality)
    "SoftComponentSelectionConfig",
    "SoftComponentSelection",
    # Per-gene standardization
    "ScalerConfig",
    "DifferentiableScaler",
    # Sequence Embedding
    "SequenceEmbeddingConfig",
    "SequenceEmbedding",
    # UMAP Dimensionality Reduction
    "UMAPConfig",
    "DifferentiableUMAP",
    # PHATE Dimensionality Reduction
    "PHATEConfig",
    "DifferentiablePHATE",
]
