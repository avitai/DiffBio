"""Shared scalar-loss balancing helpers for DiffBio operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from opifex.core.physics.gradnorm import GradNormBalancer


def combine_scalar_losses(
    losses: Mapping[str, Float[Array, ""]],
    *,
    balancer: GradNormBalancer | None,
) -> Float[Array, ""]:
    """Combine scalar losses, weighted by ``balancer`` when one is given.

    Args:
        losses: Named scalar losses to combine.
        balancer: The ``GradNormBalancer`` whose weights combine the losses, or ``None``
            to sum them.

    Returns:
        Combined scalar loss.

    Raises:
        ValueError: If *losses* is empty.
    """
    if not losses:
        msg = "losses must contain at least one scalar loss"
        raise ValueError(msg)
    loss_values = list(losses.values())
    if balancer is not None:
        return balancer.compute_weighted_loss(jnp.stack(loss_values))
    total_loss = loss_values[0]
    for loss_value in loss_values[1:]:
        total_loss = total_loss + loss_value
    return total_loss


class LossBalancingMixin:
    """Reusable operator mixin exposing ``compute_balanced_loss``.

    The mixin is a stateless combiner: with ``config.use_gradnorm`` it builds a fresh
    ``GradNormBalancer`` from the operator's ``rngs`` on every call and never updates its
    weights, so it weights the losses equally; a training loop that wants adaptive GradNorm
    composes the balancer itself across steps, as ``diffbio.pipelines.joint_training`` does.
    """

    config: Any
    rngs: nnx.Rngs

    def compute_balanced_loss(
        self,
        losses: Mapping[str, Float[Array, ""]],
    ) -> Float[Array, ""]:
        """Combine operator loss terms using the config's GradNorm flag."""
        balancer = (
            GradNormBalancer(num_losses=len(losses), rngs=self.rngs)
            if getattr(self.config, "use_gradnorm", False)
            else None
        )
        return combine_scalar_losses(losses, balancer=balancer)
