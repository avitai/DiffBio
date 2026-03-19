"""Single-cell analysis operators for differentiable scRNA-seq processing.

This module provides differentiable components for single-cell analysis:
- DifferentiableAmbientRemoval: CellBender-style ambient RNA removal
- DifferentiableDiffusionImputer: MAGIC-style diffusion imputation
- DifferentiableDoubletScorer: Scrublet-style doublet detection
- DifferentiableLigandReceptor: Ligand-receptor co-expression scoring
- SoftKMeansClustering: Differentiable soft k-means clustering
- DifferentiableHarmony: Harmony-style batch correction
- DifferentiableSwitchDE: Sigmoidal switch differential expression
- DifferentiableVelocity: RNA velocity via Neural ODEs
"""

from diffbio.operators.singlecell.ambient_removal import (
    AmbientRemovalConfig,
    DifferentiableAmbientRemoval,
)
from diffbio.operators.singlecell.communication import (
    DifferentiableLigandReceptor,
    LRScoringConfig,
)
from diffbio.operators.singlecell.batch_correction import (
    BatchCorrectionConfig,
    DifferentiableHarmony,
)
from diffbio.operators.singlecell.doublet_detection import (
    DifferentiableDoubletScorer,
    DoubletScorerConfig,
)
from diffbio.operators.singlecell.imputation import (
    DifferentiableDiffusionImputer,
    DiffusionImputerConfig,
)
from diffbio.operators.singlecell.soft_clustering import (
    SoftClusteringConfig,
    SoftKMeansClustering,
)
from diffbio.operators.singlecell.switch_de import (
    DifferentiableSwitchDE,
    SwitchDEConfig,
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
    # Communication (L-R scoring)
    "DifferentiableLigandReceptor",
    "LRScoringConfig",
    # Batch Correction
    "BatchCorrectionConfig",
    "DifferentiableHarmony",
    # Doublet Detection
    "DifferentiableDoubletScorer",
    "DoubletScorerConfig",
    # Imputation
    "DifferentiableDiffusionImputer",
    "DiffusionImputerConfig",
    # Switch DE
    "DifferentiableSwitchDE",
    "SwitchDEConfig",
    # Velocity
    "DifferentiableVelocity",
    "VelocityConfig",
]
