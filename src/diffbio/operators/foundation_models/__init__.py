"""Foundation model operators for DiffBio.

This module provides transformer-based sequence encoders and single-cell
foundation model infrastructure plus shared adapter contracts.

Operators:
    TransformerSequenceEncoder: BERT-style transformer for sequence embedding
    DifferentiableFoundationModel: Geneformer/scGPT-style foundation model
    GeneTokenizer: Rank-value gene tokenization via soft sorting

Factory Functions:
    create_dna_encoder: Create encoder for DNA sequences
    create_rna_encoder: Create encoder for RNA sequences
    create_foundation_model: Create registered foundation-model operator
"""

from diffbio.operators.foundation_models.contracts import (
    AdapterMode,
    FOUNDATION_BENCHMARK_COMPARISON_AXES,
    FoundationArtifactSpec,
    FoundationEmbeddingMixin,
    FoundationEmbeddingOperatorConfig,
    FoundationModelKind,
    PoolingStrategy,
    build_foundation_benchmark_metadata,
    build_foundation_model_metadata,
    create_foundation_model,
    decode_foundation_model_metadata,
    decode_foundation_text,
    encode_foundation_text,
    get_foundation_model_cls,
    register_foundation_model,
)
from diffbio.operators.foundation_models.adapters import (
    FoundationBenchmarkAdapter,
    SequenceFoundationAdapter,
    create_foundation_adapter,
    get_foundation_adapter_cls,
    register_foundation_adapter,
)
from diffbio.operators.foundation_models.embedding_probe import (
    EmbeddingProbeConfig,
    LinearEmbeddingProbe,
)
from diffbio.operators.foundation_models.foundation_model import (
    DifferentiableFoundationModel,
    FoundationModelConfig,
    GeneTokenizer,
)
from diffbio.operators.foundation_models.precomputed import (
    DNABERT2PrecomputedAdapter,
    GeneformerPrecomputedAdapter,
    NucleotideTransformerPrecomputedAdapter,
    ScGPTPrecomputedAdapter,
    SequencePrecomputedAdapter,
    SingleCellPrecomputedAdapter,
)
from diffbio.operators.foundation_models.frozen import FrozenSequenceEncoderAdapter
from diffbio.operators.foundation_models.transformer_encoder import (
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
    create_dna_encoder,
    create_rna_encoder,
)

__all__ = [
    "DifferentiableFoundationModel",
    "DNABERT2PrecomputedAdapter",
    "EmbeddingProbeConfig",
    "FoundationBenchmarkAdapter",
    "FoundationModelConfig",
    "GeneTokenizer",
    "GeneformerPrecomputedAdapter",
    "FrozenSequenceEncoderAdapter",
    "LinearEmbeddingProbe",
    "NucleotideTransformerPrecomputedAdapter",
    "ScGPTPrecomputedAdapter",
    "SequenceFoundationAdapter",
    "SequencePrecomputedAdapter",
    "AdapterMode",
    "FOUNDATION_BENCHMARK_COMPARISON_AXES",
    "FoundationArtifactSpec",
    "FoundationEmbeddingMixin",
    "FoundationEmbeddingOperatorConfig",
    "FoundationModelKind",
    "PoolingStrategy",
    "SingleCellPrecomputedAdapter",
    "TransformerSequenceEncoder",
    "TransformerSequenceEncoderConfig",
    "build_foundation_benchmark_metadata",
    "build_foundation_model_metadata",
    "create_foundation_adapter",
    "create_foundation_model",
    "create_dna_encoder",
    "decode_foundation_model_metadata",
    "decode_foundation_text",
    "encode_foundation_text",
    "get_foundation_adapter_cls",
    "get_foundation_model_cls",
    "create_rna_encoder",
    "register_foundation_adapter",
    "register_foundation_model",
]
