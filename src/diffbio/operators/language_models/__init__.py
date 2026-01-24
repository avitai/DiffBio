"""DNA/RNA language model operators for DiffBio.

This module provides transformer-based sequence encoders following
DNABERT and RNA-FM architecture patterns for DNA/RNA sequence
embedding and analysis.

Operators:
    TransformerSequenceEncoder: BERT-style transformer for sequence embedding

Factory Functions:
    create_dna_encoder: Create encoder for DNA sequences
    create_rna_encoder: Create encoder for RNA sequences
"""

from diffbio.operators.language_models.transformer_encoder import (
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
    create_dna_encoder,
    create_rna_encoder,
)

__all__ = [
    "TransformerSequenceEncoder",
    "TransformerSequenceEncoderConfig",
    "create_dna_encoder",
    "create_rna_encoder",
]
