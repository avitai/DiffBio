"""Opifex-owned optimizer helpers for benchmark training loops."""

from __future__ import annotations

import optax
from opifex.core.training.optimizers import OptimizerConfig, create_optimizer

BENCHMARK_OPTIMIZER_SUBSTRATE = {
    "optimizer_factory": "opifex.core.training.optimizers.create_optimizer",
    "optimizer_config": "opifex.core.training.optimizers.OptimizerConfig",
}


def create_benchmark_optimizer(
    *,
    learning_rate: float,
    optimizer_type: str = "adam",
    gradient_clip: float | None = None,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Create the benchmark optimizer through the Opifex training substrate.

    ``weight_decay`` is exposed because ``OptimizerConfig`` defaults it to 0.0 while
    ``optax.adamw`` defaults it to 1e-4; a caller moving off a direct ``optax.adamw``
    call needs to be able to say which of the two it meant.
    """
    return create_optimizer(
        OptimizerConfig(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            gradient_clip=gradient_clip,
            weight_decay=weight_decay,
        )
    )
