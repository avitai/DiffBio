"""Utility functions for DiffBio.

This module provides utility functions for I/O, encoding, training,
neural network building, and other common operations in bioinformatics pipelines.
"""

from diffbio.utils.dependency_runtime import (
    ECOSYSTEM_PACKAGES,
    DependencyRuntimeRecord,
    FNOConstructorContract,
    collect_dependency_runtime,
    inspect_fno_constructor,
    verify_canonical_dependency_runtime,
)
from diffbio.utils.quality import apply_quality_filter
from diffbio.utils.nn_utils import (
    ensure_rngs,
    extract_windows_1d,
    get_rng_key,
    init_learnable_param,
)
from diffbio.utils.training import (
    Trainer,
    TrainingConfig,
    TrainingState,
    create_optax_optimizer,
    create_synthetic_training_data,
    cross_entropy_loss,
    data_iterator,
)

__all__ = [
    # Dependency runtime utilities
    "ECOSYSTEM_PACKAGES",
    "DependencyRuntimeRecord",
    "FNOConstructorContract",
    "collect_dependency_runtime",
    "inspect_fno_constructor",
    "verify_canonical_dependency_runtime",
    # Training utilities
    "Trainer",
    "TrainingConfig",
    "TrainingState",
    "create_optax_optimizer",
    "create_synthetic_training_data",
    "cross_entropy_loss",
    "data_iterator",
    # Quality utilities
    "apply_quality_filter",
    # Neural network utilities
    "ensure_rngs",
    "extract_windows_1d",
    "get_rng_key",
    "init_learnable_param",
]
