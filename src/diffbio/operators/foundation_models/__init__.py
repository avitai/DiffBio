"""Foundation model operators for DiffBio.

This module provides transformer-based sequence encoders and single-cell
foundation model infrastructure.

Operators:
    TransformerSequenceEncoder: BERT-style transformer for sequence embedding
    DifferentiableFoundationModel: Geneformer/scGPT-style foundation model
    GeneTokenizer: Rank-value gene tokenization via soft sorting

Factory Functions:
    create_dna_encoder: Create encoder for DNA sequences
    create_rna_encoder: Create encoder for RNA sequences
"""

from diffbio.operators.foundation_models.foundation_model import (
    DifferentiableFoundationModel,
    FoundationModelConfig,
    GeneTokenizer,
)
from diffbio.operators.foundation_models.transformer_encoder import (
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
    create_dna_encoder,
    create_rna_encoder,
)

__all__ = [
    "DifferentiableFoundationModel",
    "FoundationModelConfig",
    "GeneTokenizer",
    "TransformerSequenceEncoder",
    "TransformerSequenceEncoderConfig",
    "create_dna_encoder",
    "create_rna_encoder",
]
