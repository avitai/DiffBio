"""Mapping operators for differentiable read alignment.

This module provides neural network-based approaches to read mapping
that enable gradient flow through the mapping process.

- NeuralReadMapper: Cross-attention based soft read mapping
"""

from diffbio.operators.mapping.neural_mapper import (
    NeuralReadMapper,
    NeuralReadMapperConfig,
)

__all__ = [
    "NeuralReadMapper",
    "NeuralReadMapperConfig",
]
