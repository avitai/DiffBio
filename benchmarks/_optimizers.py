"""The one place the benchmark training loops build their optimizer, through substrax."""

from __future__ import annotations

from typing import TYPE_CHECKING

from substrax.optim import create_transformation, OptimizerConfig


if TYPE_CHECKING:
    import optax
    from flax import nnx

BENCHMARK_OPTIMIZER_SUBSTRATE = {
    "optimizer_factory": "substrax.optim.create_transformation",
    "optimizer_config": "substrax.optim.OptimizerConfig",
}


def create_benchmark_optimizer(
    model: nnx.Module,
    *,
    learning_rate: float,
    optimizer_type: str = "adam",
    gradient_clip: float | None = None,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Create the benchmark optimizer for ``model`` through substrax.

    ``weight_decay`` is exposed because ``OptimizerConfig`` defaults it to 0.0 while
    ``optax.adamw`` defaults it to 1e-4; a caller moving off a direct ``optax.adamw``
    call needs to be able to say which of the two it meant. substrax refuses a decay
    on an optimizer without decoupled decay.
    """
    return create_transformation(
        model,
        OptimizerConfig(
            optimizer_type=optimizer_type,  # type: ignore[arg-type]
            learning_rate=learning_rate,
            gradient_clip_norm=gradient_clip,
            weight_decay=weight_decay,
        ),
    )
