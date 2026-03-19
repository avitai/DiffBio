"""Single-cell analysis operators for differentiable scRNA-seq processing.

This module provides differentiable components for single-cell analysis:
- DifferentiableAmbientRemoval: CellBender-style ambient RNA removal
- DifferentiableCellAnnotator: Cell type annotation (celltypist/cellassign/scanvi)
- DifferentiableDiffusionImputer: MAGIC-style diffusion imputation
- DifferentiableDoubletScorer: Scrublet-style doublet detection
- DifferentiableCellCommunication: GNN-based cell-cell communication analysis
- DifferentiableGRN: GATv2-based gene regulatory network inference
- DifferentiableLigandReceptor: Ligand-receptor co-expression scoring
- SoftKMeansClustering: Differentiable soft k-means clustering
- DifferentiableHarmony: Harmony-style batch correction
- DifferentiableMMDBatchCorrection: MMD-regularised autoencoder batch correction
- DifferentiableWGANBatchCorrection: Adversarial (WGAN) batch correction
- DifferentiableSwitchDE: Sigmoidal switch differential expression
- DifferentiableVelocity: RNA velocity via Neural ODEs
- DifferentiablePseudotime: Diffusion-map pseudotime ordering
- DifferentiableFateProbability: Absorption-based fate estimation
- DifferentiableSpatialDomain: STAGATE-style spatial domain identification
- DifferentiablePASTEAlignment: PASTE-style spatial slice alignment
"""

from diffbio.operators.singlecell.ambient_removal import (
    AmbientRemovalConfig,
    DifferentiableAmbientRemoval,
)
from diffbio.operators.singlecell.cell_annotation import (
    CellAnnotatorConfig,
    DifferentiableCellAnnotator,
)
from diffbio.operators.singlecell.communication import (
    CellCommunicationConfig,
    DifferentiableCellCommunication,
    DifferentiableLigandReceptor,
    LRScoringConfig,
)
from diffbio.operators.singlecell.batch_correction import (
    BatchCorrectionConfig,
    DifferentiableHarmony,
)
from diffbio.operators.singlecell.enhanced_batch_correction import (
    DifferentiableMMDBatchCorrection,
    DifferentiableWGANBatchCorrection,
    MMDBatchCorrectionConfig,
    WGANBatchCorrectionConfig,
)
from diffbio.operators.singlecell.grn_inference import (
    DifferentiableGRN,
    GRNInferenceConfig,
)
from diffbio.operators.singlecell.doublet_detection import (
    DifferentiableDoubletScorer,
    DifferentiableSoloDetector,
    DoubletScorerConfig,
    SoloDetectorConfig,
)
from diffbio.operators.singlecell.imputation import (
    DifferentiableDiffusionImputer,
    DifferentiableTransformerDenoiser,
    DiffusionImputerConfig,
    TransformerDenoiserConfig,
)
from diffbio.operators.singlecell.soft_clustering import (
    SoftClusteringConfig,
    SoftKMeansClustering,
)
from diffbio.operators.singlecell.switch_de import (
    DifferentiableSwitchDE,
    SwitchDEConfig,
)
from diffbio.operators.singlecell.trajectory import (
    DifferentiableFateProbability,
    DifferentiablePseudotime,
    FateProbabilityConfig,
    PseudotimeConfig,
)
from diffbio.operators.singlecell.spatial_domains import (
    DifferentiablePASTEAlignment,
    DifferentiableSpatialDomain,
    PASTEAlignmentConfig,
    SpatialDomainConfig,
)
from diffbio.operators.singlecell.velocity import (
    DifferentiableVelocity,
    VelocityConfig,
)

__all__ = [
    # Ambient Removal
    "AmbientRemovalConfig",
    "DifferentiableAmbientRemoval",
    # Cell Annotation
    "CellAnnotatorConfig",
    "DifferentiableCellAnnotator",
    # Clustering
    "SoftClusteringConfig",
    "SoftKMeansClustering",
    # Communication (L-R scoring + GNN-based)
    "CellCommunicationConfig",
    "DifferentiableCellCommunication",
    "DifferentiableLigandReceptor",
    "LRScoringConfig",
    # Batch Correction (Harmony)
    "BatchCorrectionConfig",
    "DifferentiableHarmony",
    # Batch Correction (MMD + WGAN)
    "DifferentiableMMDBatchCorrection",
    "DifferentiableWGANBatchCorrection",
    "MMDBatchCorrectionConfig",
    "WGANBatchCorrectionConfig",
    # GRN Inference
    "DifferentiableGRN",
    "GRNInferenceConfig",
    # Doublet Detection
    "DifferentiableDoubletScorer",
    "DifferentiableSoloDetector",
    "DoubletScorerConfig",
    "SoloDetectorConfig",
    # Imputation
    "DifferentiableDiffusionImputer",
    "DifferentiableTransformerDenoiser",
    "DiffusionImputerConfig",
    "TransformerDenoiserConfig",
    # Switch DE
    "DifferentiableSwitchDE",
    "SwitchDEConfig",
    # Trajectory Inference
    "DifferentiableFateProbability",
    "DifferentiablePseudotime",
    "FateProbabilityConfig",
    "PseudotimeConfig",
    # Spatial Domain Identification
    "DifferentiablePASTEAlignment",
    "DifferentiableSpatialDomain",
    "PASTEAlignmentConfig",
    "SpatialDomainConfig",
    # Velocity
    "DifferentiableVelocity",
    "VelocityConfig",
]
