"""Differentiable preprocessing operators for bioinformatics sequences.

This module provides preprocessing operators that maintain gradient flow
for end-to-end trainable pipelines.

Operators:
    SoftAdapterRemoval: Differentiable adapter trimming using soft alignment
    DifferentiableDuplicateWeighting: Probabilistic duplicate weighting
    SoftErrorCorrection: Neural network-based error correction
"""

from diffbio.operators.preprocessing.adapter_removal import (
    AdapterRemovalConfig,
    SoftAdapterRemoval,
)
from diffbio.operators.preprocessing.duplicate_filter import (
    DifferentiableDuplicateWeighting,
    DuplicateWeightingConfig,
)
from diffbio.operators.preprocessing.error_correction import (
    ErrorCorrectionConfig,
    SoftErrorCorrection,
)

__all__ = [
    "AdapterRemovalConfig",
    "SoftAdapterRemoval",
    "DuplicateWeightingConfig",
    "DifferentiableDuplicateWeighting",
    "ErrorCorrectionConfig",
    "SoftErrorCorrection",
]
