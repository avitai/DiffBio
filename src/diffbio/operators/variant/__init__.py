"""Variant calling operators for differentiable variant detection.

This module provides differentiable components for variant calling:
- DifferentiablePileup: Generates pileup from aligned reads
- VariantClassifier: Classifies variants from pileup data
"""

from diffbio.operators.variant.classifier import (
    VariantClassifier,
    VariantClassifierConfig,
)
from diffbio.operators.variant.pileup import DifferentiablePileup, PileupConfig

__all__ = [
    "DifferentiablePileup",
    "PileupConfig",
    "VariantClassifier",
    "VariantClassifierConfig",
]
