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
) -> optax.GradientTransformation:
    """Create the benchmark optimizer through the Opifex training substrate."""
    return create_optimizer(
        OptimizerConfig(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            gradient_clip=gradient_clip,
        )
    )
