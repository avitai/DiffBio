"""Pre-built differentiable pipeline templates.

This module provides ready-to-use pipeline templates for common bioinformatics
workflows such as variant calling, preprocessing, differential expression,
and single-cell analysis.
"""

from diffbio.pipelines.adapters import (
    RenameField,
    RenameFieldConfig,
)
from diffbio.pipelines.differential_expression import (
    DEPipelineConfig,
    DifferentialExpressionPipeline,
)
from diffbio.pipelines.joint_preprocessing import (
    JointPreprocessingPipeline,
    JointPreprocessingPipelineConfig,
)
from diffbio.pipelines.joint_training import (
    JointTrainingConfig,
    JointTrainingResult,
    fit_jointly,
)
from diffbio.pipelines.enhanced_variant_calling import (
    EnhancedVariantCallingPipeline,
    EnhancedVariantCallingPipelineConfig,
    create_enhanced_variant_calling_pipeline,
)
from diffbio.pipelines.preprocessing import (
    PreprocessingPipeline,
    PreprocessingPipelineConfig,
    create_preprocessing_pipeline,
)
from diffbio.pipelines.single_cell import (
    SingleCellPipeline,
    SingleCellPipelineConfig,
    create_single_cell_pipeline,
)
from diffbio.pipelines.perturbation import (
    PerturbationPipeline,
    PerturbationPipelineConfig,
    PerturbationPipelineResult,
)
from diffbio.pipelines.variant_calling import (
    VariantCallingPipeline,
    VariantCallingPipelineConfig,
    create_cnn_variant_pipeline,
    create_variant_calling_pipeline,
)


__all__ = [
    # Composition adapters
    "RenameField",
    "RenameFieldConfig",
    # Differential Expression
    "DEPipelineConfig",
    "DifferentialExpressionPipeline",
    # Joint single-cell preprocessing -> annotation
    "JointPreprocessingPipeline",
    "JointPreprocessingPipelineConfig",
    "JointTrainingConfig",
    "JointTrainingResult",
    "fit_jointly",
    # Enhanced Variant Calling
    "EnhancedVariantCallingPipeline",
    "EnhancedVariantCallingPipelineConfig",
    "create_enhanced_variant_calling_pipeline",
    # Preprocessing
    "PreprocessingPipeline",
    "PreprocessingPipelineConfig",
    "create_preprocessing_pipeline",
    # Single-Cell Analysis
    "SingleCellPipeline",
    "SingleCellPipelineConfig",
    "create_single_cell_pipeline",
    # Perturbation
    "PerturbationPipeline",
    "PerturbationPipelineConfig",
    "PerturbationPipelineResult",
    # Variant calling
    "VariantCallingPipeline",
    "VariantCallingPipelineConfig",
    "create_variant_calling_pipeline",
    "create_cnn_variant_pipeline",
]
