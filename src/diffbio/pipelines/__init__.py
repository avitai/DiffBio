"""Pre-built differentiable pipeline templates.

This module provides ready-to-use pipeline templates for common bioinformatics
workflows such as variant calling, preprocessing, differential expression,
and single-cell analysis.
"""

from diffbio.pipelines.differential_expression import (
    DEPipelineConfig,
    DifferentialExpressionPipeline,
)
from diffbio.pipelines.preprocessing import (
    PreprocessingPipeline,
    PreprocessingPipelineConfig,
    create_preprocessing_pipeline,
)
from diffbio.pipelines.variant_calling import (
    VariantCallingPipeline,
    VariantCallingPipelineConfig,
    create_cnn_variant_pipeline,
    create_variant_calling_pipeline,
)


__all__ = [
    # Differential Expression
    "DEPipelineConfig",
    "DifferentialExpressionPipeline",
    # Preprocessing
    "PreprocessingPipeline",
    "PreprocessingPipelineConfig",
    "create_preprocessing_pipeline",
    # Variant calling
    "VariantCallingPipeline",
    "VariantCallingPipelineConfig",
    "create_variant_calling_pipeline",
    "create_cnn_variant_pipeline",
]
