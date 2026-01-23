"""Core module for DiffBio.

This module provides foundational components for building differentiable
bioinformatics operators:

- **base_operators**: Domain-specific base classes (TemperatureOperator,
  SequenceOperator, EncoderDecoderOperator, GraphOperator, HMMOperator)
- **differentiable_ops**: Soft approximations of discrete operations
  (soft_argmax, soft_sort, gumbel_softmax, etc.)
- **neural_components**: DiffBio-specific neural network modules
  (GumbelSoftmaxModule, GraphMessagePassing)
- **data_types**: Type aliases and protocols for type safety

Usage:
    from diffbio.core import TemperatureOperator, logsumexp_smooth_max
    from diffbio.core.data_types import SequenceData
"""

# Base operators
from diffbio.core.base_operators import (
    EncoderDecoderOperator,
    GraphOperator,
    HMMOperator,
    SequenceOperator,
    TemperatureOperator,
)

# Differentiable operations
from diffbio.core.differentiable_ops import (
    differentiable_scan,
    differentiable_topk,
    gumbel_softmax,
    logsumexp_smooth_max,
    segment_softmax,
    soft_argmax,
    soft_attention_weights,
    soft_one_hot,
    soft_sort,
    soft_threshold,
)

# Neural components (DiffBio-specific)
from diffbio.core.neural_components import (
    GraphMessagePassing,
    GumbelSoftmaxModule,
)

# Re-export from artifex (required dependency)
from artifex.generative_models.core.layers.positional import (
    PositionalEncoding,
    RotaryPositionalEncoding as RoPE,
    SinusoidalPositionalEncoding,
)
from artifex.generative_models.core.layers.residual import (
    Conv1DResidualBlock as ResidualBlock1D,
    Conv2DResidualBlock as ResidualBlock2D,
)

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

__all__ = [
    # Base operators
    "TemperatureOperator",
    "SequenceOperator",
    "EncoderDecoderOperator",
    "GraphOperator",
    "HMMOperator",
    # Differentiable ops
    "soft_argmax",
    "soft_sort",
    "soft_threshold",
    "logsumexp_smooth_max",
    "segment_softmax",
    "gumbel_softmax",
    "differentiable_scan",
    "soft_one_hot",
    "soft_attention_weights",
    "differentiable_topk",
    # Neural components (DiffBio-specific)
    "GumbelSoftmaxModule",
    "GraphMessagePassing",
    # Re-exported from artifex
    "PositionalEncoding",
    "SinusoidalPositionalEncoding",
    "RoPE",
    "ResidualBlock1D",
    "ResidualBlock2D",
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
