"""Utility functions for DiffBio.

This module provides utility functions for I/O, encoding, training,
neural network building, and other common operations in bioinformatics pipelines.
"""

from diffbio.utils.nn_utils import (
    build_mlp_layers,
    ensure_rngs,
    get_rng_key,
    init_learnable_param,
    safe_divide,
    safe_log,
    sigmoid_blend,
    soft_threshold,
    temperature_scaled_softmax,
)
from diffbio.utils.training import (
    Trainer,
    TrainingConfig,
    TrainingState,
    create_optax_optimizer,
    create_optimizer,
    create_synthetic_training_data,
    cross_entropy_loss,
    data_iterator,
)

__all__ = [
    # Training utilities
    "Trainer",
    "TrainingConfig",
    "TrainingState",
    "create_optax_optimizer",
    "create_optimizer",
    "create_synthetic_training_data",
    "cross_entropy_loss",
    "data_iterator",
    # Neural network utilities
    "build_mlp_layers",
    "ensure_rngs",
    "get_rng_key",
    "init_learnable_param",
    "safe_divide",
    "safe_log",
    "sigmoid_blend",
    "soft_threshold",
    "temperature_scaled_softmax",
]
