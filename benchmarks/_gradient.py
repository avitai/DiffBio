"""Gradient flow verification for DiffBio operators.

This is the only DiffBio-specific benchmark infrastructure. All other
infrastructure comes from calibrax (profiling, results, analysis) and
datarax (data loading).

Verifying that gradients flow through operators is unique to
differentiable bioinformatics -- calibrax has no equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from flax import nnx


@dataclass(frozen=True, kw_only=True)
class GradientFlowResult:
    """Result of a gradient flow check.

    Attributes:
        gradient_norm: L2 norm of all parameter gradients.
        gradient_nonzero: Whether gradients are non-trivially nonzero.
    """

    gradient_norm: float
    gradient_nonzero: bool


def check_gradient_flow(
    loss_fn: Any,
    model: nnx.Module,
    *args: Any,
) -> GradientFlowResult:
    """Verify gradient flow through a model's learnable parameters.

    Computes gradients of ``loss_fn(model, *args)`` with respect to
    the model's ``nnx.Param`` leaves and reports the total L2 norm.

    Args:
        loss_fn: Callable ``(model, *args) -> scalar``.
        model: Flax NNX module to differentiate.
        *args: Additional positional arguments forwarded to *loss_fn*.

    Returns:
        :class:`GradientFlowResult` with gradient norm and nonzero flag.
    """
    grad_fn = nnx.grad(loss_fn)
    grads = grad_fn(model, *args)

    total_sq = 0.0
    for _, param in nnx.iter_graph(grads):
        if hasattr(param, "value") and isinstance(param.value, jnp.ndarray):
            total_sq += float(jnp.sum(param.value ** 2))

    norm = total_sq ** 0.5
    return GradientFlowResult(
        gradient_norm=norm,
        gradient_nonzero=norm > 1e-8,
    )
