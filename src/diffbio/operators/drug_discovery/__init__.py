"""Drug discovery operators for differentiable chemoinformatics.

This module provides differentiable operators for molecular property prediction,
fingerprint computation, and similarity scoring.

Operators:
    MolecularPropertyPredictor: ChemProp-style MPNN for property prediction
    DifferentiableMolecularFingerprint: Neural graph fingerprints
    CircularFingerprintOperator: Differentiable ECFP/Morgan fingerprints
    MolecularSimilarityOperator: Differentiable Tanimoto/cosine similarity

Utilities:
    smiles_to_graph: Convert SMILES to molecular graph
    batch_smiles_to_graphs: Batch conversion with padding
"""

from diffbio.operators.drug_discovery.fingerprint import (
    CircularFingerprintConfig,
    CircularFingerprintOperator,
    DifferentiableMolecularFingerprint,
    MolecularFingerprintConfig,
    create_ecfp4_operator,
    create_ecfp6_operator,
    create_fcfp4_operator,
    create_fingerprint_operator,
)
from diffbio.operators.drug_discovery.message_passing import (
    MessagePassingLayer,
    StackedMessagePassing,
)
from diffbio.operators.drug_discovery.primitives import (
    AtomFeatureConfig,
    DEFAULT_ATOM_CONFIG,
    DEFAULT_ATOM_FEATURES,
    batch_smiles_to_graphs,
    smiles_to_graph,
)
from diffbio.operators.drug_discovery.property_predictor import (
    MolecularPropertyConfig,
    MolecularPropertyPredictor,
    create_property_predictor,
)
from diffbio.operators.drug_discovery.similarity import (
    MolecularSimilarityConfig,
    MolecularSimilarityOperator,
    cosine_similarity,
    create_similarity_operator,
    dice_similarity,
    tanimoto_similarity,
)

__all__ = [
    # Primitives
    "smiles_to_graph",
    "batch_smiles_to_graphs",
    "DEFAULT_ATOM_FEATURES",
    "DEFAULT_ATOM_CONFIG",
    "AtomFeatureConfig",
    # Message Passing
    "MessagePassingLayer",
    "StackedMessagePassing",
    # Property Prediction
    "MolecularPropertyConfig",
    "MolecularPropertyPredictor",
    "create_property_predictor",
    # Neural Fingerprints
    "MolecularFingerprintConfig",
    "DifferentiableMolecularFingerprint",
    "create_fingerprint_operator",
    # Circular Fingerprints (ECFP/Morgan)
    "CircularFingerprintConfig",
    "CircularFingerprintOperator",
    "create_ecfp4_operator",
    "create_ecfp6_operator",
    "create_fcfp4_operator",
    # Similarity
    "MolecularSimilarityConfig",
    "MolecularSimilarityOperator",
    "create_similarity_operator",
    "tanimoto_similarity",
    "cosine_similarity",
    "dice_similarity",
]
