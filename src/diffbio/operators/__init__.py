"""Differentiable bioinformatics operators for DiffBio.

This module provides differentiable operators for common bioinformatics
operations such as quality filtering, sequence alignment, and variant calling.
All operators extend Datarax's OperatorModule for seamless integration.
"""

from diffbio.operators.quality_filter import (
    DifferentiableQualityFilter,
    QualityFilterConfig,
)


__all__ = [
    "DifferentiableQualityFilter",
    "QualityFilterConfig",
]
