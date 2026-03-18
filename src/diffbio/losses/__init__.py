"""Loss functions and regularization for bioinformatics pipelines.

This module provides loss functions for training differentiable bioinformatics
pipelines, including biological regularization to prevent adversarial optimization.
"""

from diffbio.losses.alignment_losses import (
    AlignmentConsistencyLoss,
    AlignmentScoreLoss,
    SoftEditDistanceLoss,
)
from diffbio.losses.biological_regularization import (
    BiologicalPlausibilityLoss,
    BiologicalRegularizationConfig,
    GapPatternRegularization,
    GCContentRegularization,
    SequenceComplexityLoss,
)
from diffbio.losses.singlecell_losses import (
    BatchMixingLoss,
    ClusteringCompactnessLoss,
    VelocityConsistencyLoss,
)
from diffbio.losses.metric_losses import DifferentiableAUROC, ExactAUROC
from diffbio.losses.statistical_losses import (
    HMMLikelihoodLoss,
    NegativeBinomialLoss,
    VAELoss,
)

__all__ = [
    # Alignment losses
    "AlignmentConsistencyLoss",
    "AlignmentScoreLoss",
    "SoftEditDistanceLoss",
    # Biological regularization
    "BiologicalPlausibilityLoss",
    "BiologicalRegularizationConfig",
    "GapPatternRegularization",
    "GCContentRegularization",
    "SequenceComplexityLoss",
    # Metric losses
    "DifferentiableAUROC",
    "ExactAUROC",
    # Single-cell losses
    "BatchMixingLoss",
    "ClusteringCompactnessLoss",
    "VelocityConsistencyLoss",
    # Statistical losses
    "HMMLikelihoodLoss",
    "NegativeBinomialLoss",
    "VAELoss",
]
