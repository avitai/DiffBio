"""Shared scalar-loss balancing helpers for DiffBio operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flax import nnx
from jaxtyping import Array, Float
from opifex.core.physics.gradnorm import GradNormBalancer

from diffbio.utils.nn_utils import ensure_rngs


def combine_scalar_losses(
    losses: Mapping[str, Float[Array, ""]],
    *,
    use_gradnorm: bool,
    rngs: nnx.Rngs | None = None,
) -> Float[Array, ""]:
    """Combine scalar losses with optional GradNorm-based balancing.

    Args:
        losses: Named scalar losses to combine.
        use_gradnorm: Whether to balance losses with ``GradNormBalancer``.
        rngs: Optional random generators used when constructing GradNorm.

    Returns:
        Combined scalar loss.

    Raises:
        ValueError: If *losses* is empty.
    """
    if not losses:
        msg = "losses must contain at least one scalar loss"
        raise ValueError(msg)

    loss_values = list(losses.values())
    if use_gradnorm:
        balancer = GradNormBalancer(
            num_losses=len(loss_values),
            rngs=ensure_rngs(rngs),
        )
        return balancer(loss_values)

    total_loss = loss_values[0]
    for loss_value in loss_values[1:]:
        total_loss = total_loss + loss_value
    return total_loss


class LossBalancingMixin:
    """Reusable operator mixin exposing ``compute_balanced_loss``."""

    config: Any

    def compute_balanced_loss(
        self,
        losses: Mapping[str, Float[Array, ""]],
    ) -> Float[Array, ""]:
        """Combine operator loss terms using the config's GradNorm flag."""
        return combine_scalar_losses(
            losses,
            use_gradnorm=bool(getattr(self.config, "use_gradnorm", False)),
        )
