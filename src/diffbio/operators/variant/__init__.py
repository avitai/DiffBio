"""Variant calling operators for differentiable variant detection.

This module provides differentiable components for variant calling:
- DifferentiablePileup: Generates pileup from aligned reads
- DeepVariantStylePileup: DeepVariant-style multi-channel pileup images
- VariantClassifier: MLP-based variant classifier
- CNNVariantClassifier: CNN-based pileup image classifier (DeepVariant-style)
- SoftVariantQualityFilter: GMM-based quality filtering (VQSR-style)
- DifferentiableCNVSegmentation: Attention-based CNV detection
"""

from diffbio.operators.variant.classifier import (
    VariantClassifier,
    VariantClassifierConfig,
)
from diffbio.operators.variant.cnn_classifier import (
    CNNVariantClassifier,
    CNNVariantClassifierConfig,
)
from diffbio.operators.variant.cnv_segmentation import (
    CNVSegmentationConfig,
    DifferentiableCNVSegmentation,
    EnhancedCNVSegmentation,
    EnhancedCNVSegmentationConfig,
)
from diffbio.operators.variant.deepvariant_pileup import (
    DeepVariantPileupConfig,
    DeepVariantStylePileup,
)
from diffbio.operators.variant.pileup import DifferentiablePileup, PileupConfig
from diffbio.operators.variant.quality_recalibration import (
    SoftVariantQualityFilter,
    VariantQualityFilterConfig,
)

__all__ = [
    # Pileup
    "DifferentiablePileup",
    "PileupConfig",
    # DeepVariant-style Pileup
    "DeepVariantStylePileup",
    "DeepVariantPileupConfig",
    # MLP Classifier
    "VariantClassifier",
    "VariantClassifierConfig",
    # CNN Classifier
    "CNNVariantClassifier",
    "CNNVariantClassifierConfig",
    # Quality Filter
    "SoftVariantQualityFilter",
    "VariantQualityFilterConfig",
    # CNV Segmentation
    "CNVSegmentationConfig",
    "DifferentiableCNVSegmentation",
    # Enhanced CNV Segmentation
    "EnhancedCNVSegmentation",
    "EnhancedCNVSegmentationConfig",
]
