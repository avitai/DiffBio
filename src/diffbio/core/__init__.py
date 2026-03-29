"""Core module for DiffBio.

This module provides foundational components for building differentiable
bioinformatics operators:

- **soft_ops**: Differentiable soft operations (soft sorting, argmax,
  comparisons, logical ops, selection, quantile, straight-through estimators)
- **base_operators**: Domain-specific base classes (TemperatureOperator,
  SequenceOperator, EncoderDecoderOperator, GraphOperator, HMMOperator)
- **neural_components**: DiffBio-specific neural network modules
  (GumbelSoftmaxModule, GraphMessagePassing)
- **gnn_components**: Graph attention modules
  (GraphAttentionLayer, GraphAttentionBlock, GATv2Layer, GATv2Block)
- **optimal_transport**: Optimal transport solvers (SinkhornLayer)
- **data_types**: Type aliases and protocols for type safety

Usage::

    from diffbio.core import soft_ops
    from diffbio.core import TemperatureOperator
    from diffbio.core import GraphAttentionLayer, GATv2Layer
    from diffbio.core import SinkhornLayer
    from diffbio.core.data_types import SequenceData
    from diffbio.core.soft_ops import SoftBool, SoftIndex
"""

import diffbio.core.base_operators as _base_operators
import diffbio.core.gnn_components as _gnn_components
import diffbio.core.graph_utils as _graph_utils
import diffbio.core.neural_components as _neural_components
import diffbio.core.optimal_transport as _optimal_transport
import diffbio.core.soft_ops as soft_ops  # noqa: F401 -- public submodule

# Soft operation types (re-exported for convenience)
SoftBool = soft_ops.SoftBool
SoftIndex = soft_ops.SoftIndex

# Base operators
TemperatureOperator = _base_operators.TemperatureOperator
SequenceOperator = _base_operators.SequenceOperator
EncoderDecoderOperator = _base_operators.EncoderDecoderOperator
GraphOperator = _base_operators.GraphOperator
HMMOperator = _base_operators.HMMOperator

# Neural components
GumbelSoftmaxModule = _neural_components.GumbelSoftmaxModule
GraphMessagePassing = _neural_components.GraphMessagePassing
PositionalEncoding = _neural_components.PositionalEncoding
SinusoidalPositionalEncoding = _neural_components.SinusoidalPositionalEncoding
RoPE = _neural_components.RoPE
ResidualBlock1D = _neural_components.ResidualBlock1D
ResidualBlock2D = _neural_components.ResidualBlock2D

# GNN components
GraphAttentionLayer = _gnn_components.GraphAttentionLayer
GraphAttentionBlock = _gnn_components.GraphAttentionBlock
GATv2Layer = _gnn_components.GATv2Layer
GATv2Block = _gnn_components.GATv2Block

# Optimal transport
SinkhornLayer = _optimal_transport.SinkhornLayer

# Graph utilities
compute_pairwise_distances = _graph_utils.compute_pairwise_distances
compute_knn_graph = _graph_utils.compute_knn_graph
compute_fuzzy_membership = _graph_utils.compute_fuzzy_membership
symmetrize_graph = _graph_utils.symmetrize_graph

# Data types (including SoftBool and SoftIndex from soft_ops)
from diffbio.core.data_types import (  # noqa: E402
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
    # Soft operations (full module)
    "soft_ops",
    "SoftBool",
    "SoftIndex",
    # Base operators
    *_base_operators.__all__,
    # GNN components
    *_gnn_components.__all__,
    # Graph utilities
    *_graph_utils.__all__,
    # Neural components
    *_neural_components.__all__,
    # Optimal transport
    *_optimal_transport.__all__,
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
