"""Single-cell analysis operators for differentiable scRNA-seq processing.

This module provides differentiable components for single-cell analysis:
- DifferentiableAmbientRemoval: CellBender-style ambient RNA removal
- SoftKMeansClustering: Differentiable soft k-means clustering
- DifferentiableHarmony: Harmony-style batch correction
- DifferentiableVelocity: RNA velocity via Neural ODEs
"""

from diffbio.operators.singlecell.ambient_removal import (
    AmbientRemovalConfig,
    DifferentiableAmbientRemoval,
)
from diffbio.operators.singlecell.batch_correction import (
    BatchCorrectionConfig,
    DifferentiableHarmony,
)
from diffbio.operators.singlecell.soft_clustering import (
    SoftClusteringConfig,
    SoftKMeansClustering,
)
from diffbio.operators.singlecell.velocity import (
    DifferentiableVelocity,
    VelocityConfig,
)

__all__ = [
    # Ambient Removal
    "AmbientRemovalConfig",
    "DifferentiableAmbientRemoval",
    # Clustering
    "SoftClusteringConfig",
    "SoftKMeansClustering",
    # Batch Correction
    "BatchCorrectionConfig",
    "DifferentiableHarmony",
    # Velocity
    "DifferentiableVelocity",
    "VelocityConfig",
]
