"""Differentiable preprocessing operators for bioinformatics sequences.

This module provides preprocessing operators that maintain gradient flow
for end-to-end trainable pipelines.

Operators:
    SoftAdapterRemoval: Differentiable adapter trimming using soft alignment
    DifferentiableDuplicateWeighting: Probabilistic duplicate weighting
    SoftErrorCorrection: Neural network-based error correction

Factories:
    wrap_probabilistic: Wrap any preprocessing operator in a datarax
        ProbabilisticOperator for random augmentation during training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from datarax.core.operator import OperatorModule
    from datarax.operators import ProbabilisticOperator


def wrap_probabilistic(
    operator: OperatorModule,
    probability: float = 0.5,
) -> ProbabilisticOperator:
    """Wrap a preprocessing operator in a ProbabilisticOperator.

    The wrapped operator is applied with the given probability during
    each forward pass, enabling random augmentation during training.
    When not applied, the input data passes through unchanged.

    Args:
        operator: A preprocessing operator to wrap.
        probability: Probability of applying the operator (0.0 to 1.0).

    Returns:
        A ProbabilisticOperator wrapping the given operator.

    Example:
        >>> adapter_remover = SoftAdapterRemoval(AdapterRemovalConfig(), rngs=rngs)
        >>> prob_remover = wrap_probabilistic(adapter_remover, probability=0.8)
    """
    from datarax.operators import (  # noqa: PLC0415
        ProbabilisticOperator,
        ProbabilisticOperatorConfig,
    )

    config = ProbabilisticOperatorConfig(operator=operator, probability=probability)
    return ProbabilisticOperator(config)


__all__ = [
    "AdapterRemovalConfig",
    "DifferentiableDuplicateWeighting",
    "DuplicateWeightingConfig",
    "ErrorCorrectionConfig",
    "SoftAdapterRemoval",
    "SoftErrorCorrection",
    "wrap_probabilistic",
]
