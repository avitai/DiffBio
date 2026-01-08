"""Pre-built differentiable pipeline templates.

This module provides ready-to-use pipeline templates for common bioinformatics
workflows such as variant calling and RNA-seq analysis.
"""

from diffbio.pipelines.variant_calling import (
    VariantCallingPipeline,
    VariantCallingPipelineConfig,
    create_variant_calling_pipeline,
)


__all__ = [
    "VariantCallingPipeline",
    "VariantCallingPipelineConfig",
    "create_variant_calling_pipeline",
]
