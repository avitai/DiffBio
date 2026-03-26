"""Base configuration classes for DiffBio operators.

This module provides base configuration classes that reduce duplication
across operator configs by providing common fields with sensible defaults.

Note: The `stochastic` and `stream_name` fields are already defined in
datarax.core.config.OperatorConfig, so we don't re-declare them here.
DiffBio configs inherit these fields automatically.
"""

from dataclasses import dataclass

from datarax.core.config import OperatorConfig

from diffbio.constants import (
    DEFAULT_DROPOUT_RATE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_NUM_CLASSES,
    DEFAULT_NUM_LAYERS,
    DEFAULT_TEMPERATURE,
)

# Re-export OperatorConfig as DiffBioOperatorConfig for backward compatibility
# and to provide a clear base class for DiffBio operators.
# Note: stochastic and stream_name are inherited from OperatorConfig.
DiffBioOperatorConfig = OperatorConfig


@dataclass(frozen=True)
class TemperatureConfig(OperatorConfig):
    """Configuration for operators with temperature parameter.

    Use this base class for operators that use temperature-based
    smoothing (logsumexp relaxation, soft thresholding, etc.).

    Attributes:
        temperature: Temperature for smooth operations.
            Lower = sharper (closer to hard operations).
            Higher = smoother (more gradient flow).
        learnable_temperature: Whether temperature is a learnable parameter.
            If True, temperature will be an nnx.Param that receives gradients.
    """

    temperature: float = DEFAULT_TEMPERATURE
    learnable_temperature: bool = False


@dataclass(frozen=True)
class ClassifierConfig(OperatorConfig):
    """Base configuration for classifier operators.

    Provides common fields for neural network classifiers.

    Attributes:
        num_classes: Number of output classes.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of hidden layers.
        dropout_rate: Dropout rate for regularization.
    """

    num_classes: int = DEFAULT_NUM_CLASSES
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout_rate: float = DEFAULT_DROPOUT_RATE
