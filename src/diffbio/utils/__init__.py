"""Utility functions for DiffBio.

This module provides utility functions for I/O, encoding, training,
and other common operations in bioinformatics pipelines.
"""

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
    "Trainer",
    "TrainingConfig",
    "TrainingState",
    "create_optax_optimizer",
    "create_optimizer",
    "create_synthetic_training_data",
    "cross_entropy_loss",
    "data_iterator",
]
