"""Population genetics operators.

This module provides differentiable operators for population genetics analysis
including ancestry estimation, phasing, and imputation.
"""

from diffbio.operators.population.ancestry_estimation import (
    AncestryEstimatorConfig,
    DifferentiableAncestryEstimator,
    create_ancestry_estimator,
)

__all__ = [
    "AncestryEstimatorConfig",
    "DifferentiableAncestryEstimator",
    "create_ancestry_estimator",
]
