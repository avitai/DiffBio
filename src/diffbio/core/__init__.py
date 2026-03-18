"""Core module for DiffBio.

This module provides foundational components for building differentiable
bioinformatics operators:

- **base_operators**: Domain-specific base classes (TemperatureOperator,
  SequenceOperator, EncoderDecoderOperator, GraphOperator, HMMOperator)
- **differentiable_ops**: Soft approximations of discrete operations
  (soft_argmax, soft_sort, gumbel_softmax, etc.)
- **neural_components**: DiffBio-specific neural network modules
  (GumbelSoftmaxModule, GraphMessagePassing)
- **gnn_components**: Graph attention modules
  (GraphAttentionLayer, GraphAttentionBlock, GATv2Layer, GATv2Block)
- **optimal_transport**: Optimal transport solvers (SinkhornLayer)
- **data_types**: Type aliases and protocols for type safety

Usage:
    from diffbio.core import TemperatureOperator, logsumexp_smooth_max
    from diffbio.core import GraphAttentionLayer, GATv2Layer
    from diffbio.core import SinkhornLayer
    from diffbio.core.data_types import SequenceData
"""

import diffbio.core.base_operators as _base_operators
import diffbio.core.differentiable_ops as _differentiable_ops
import diffbio.core.gnn_components as _gnn_components
import diffbio.core.graph_utils as _graph_utils
import diffbio.core.neural_components as _neural_components
import diffbio.core.optimal_transport as _optimal_transport

TemperatureOperator = _base_operators.TemperatureOperator
SequenceOperator = _base_operators.SequenceOperator
EncoderDecoderOperator = _base_operators.EncoderDecoderOperator
GraphOperator = _base_operators.GraphOperator
HMMOperator = _base_operators.HMMOperator

soft_argmax = _differentiable_ops.soft_argmax
soft_sort = _differentiable_ops.soft_sort
soft_threshold = _differentiable_ops.soft_threshold
logsumexp_smooth_max = _differentiable_ops.logsumexp_smooth_max
segment_softmax = _differentiable_ops.segment_softmax
gumbel_softmax = _differentiable_ops.gumbel_softmax
differentiable_scan = _differentiable_ops.differentiable_scan
soft_one_hot = _differentiable_ops.soft_one_hot
soft_attention_weights = _differentiable_ops.soft_attention_weights
differentiable_topk = _differentiable_ops.differentiable_topk

GumbelSoftmaxModule = _neural_components.GumbelSoftmaxModule
GraphMessagePassing = _neural_components.GraphMessagePassing
PositionalEncoding = _neural_components.PositionalEncoding
SinusoidalPositionalEncoding = _neural_components.SinusoidalPositionalEncoding
RoPE = _neural_components.RoPE
ResidualBlock1D = _neural_components.ResidualBlock1D
ResidualBlock2D = _neural_components.ResidualBlock2D

GraphAttentionLayer = _gnn_components.GraphAttentionLayer
GraphAttentionBlock = _gnn_components.GraphAttentionBlock
GATv2Layer = _gnn_components.GATv2Layer
GATv2Block = _gnn_components.GATv2Block

SinkhornLayer = _optimal_transport.SinkhornLayer

compute_pairwise_distances = _graph_utils.compute_pairwise_distances
compute_knn_graph = _graph_utils.compute_knn_graph
compute_fuzzy_membership = _graph_utils.compute_fuzzy_membership
symmetrize_graph = _graph_utils.symmetrize_graph

# Data types
from diffbio.core.data_types import (
    AlignmentResultData,
    BatchArray,
    DifferentiableOperator,
    GraphData,
    LatentData,
    LossFunction,
    MetadataDict,
    OperatorOutput,
    PositionWeightMatrix,
    Probability,
    ProbabilityArray,
    Regularizer,
    ScoreMatrix,
    SequenceArray,
    SequenceData,
    SequenceEncoder,
    StateDict,
    Temperature,
    VariantData,
)

__all__ = [  # pyright: ignore[reportUnsupportedDunderAll]
    *_base_operators.__all__,
    *_differentiable_ops.__all__,
    *_gnn_components.__all__,
    *_graph_utils.__all__,
    *_neural_components.__all__,
    *_optimal_transport.__all__,
    "soft_one_hot",
    "soft_attention_weights",
    "differentiable_topk",
    # Data types
    "SequenceData",
    "AlignmentResultData",
    "VariantData",
    "LatentData",
    "GraphData",
    "StateDict",
    "MetadataDict",
    "OperatorOutput",
    "DifferentiableOperator",
    "SequenceEncoder",
    "LossFunction",
    "Regularizer",
    "Temperature",
    "Probability",
    "SequenceArray",
    "BatchArray",
    "ProbabilityArray",
    "ScoreMatrix",
    "PositionWeightMatrix",
]
