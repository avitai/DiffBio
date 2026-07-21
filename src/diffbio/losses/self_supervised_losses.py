"""Self-supervised masked-value prediction losses.

Label-free objective for training preprocessing end-to-end: mask a fraction of the input
values, predict them from the rest, and score the reconstruction only at the masked
positions (the masked-language-modelling / masked-autoencoder objective used by scGPT,
Geneformer and DNABERT). The masking itself is provided by the shared masked-gene
infrastructure (``operators._masked_gene_transformer``); this module supplies only the loss
term, which the ``core.losses`` ledger flags as the one genuinely new B6 piece.
"""

from __future__ import annotations

import jax
from flax import nnx

_EPSILON = 1.0e-8
_VALID_LOSS_TYPES = ("mse", "mae")


def masked_value_loss(
    predictions: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
    *,
    loss_type: str = "mse",
    epsilon: float = _EPSILON,
) -> jax.Array:
    """Reconstruction loss averaged over masked positions only.

    Args:
        predictions: Predicted values, any shape broadcastable with ``mask``.
        targets: Ground-truth values, same shape as ``predictions``.
        mask: ``1`` at positions to reconstruct (the masked-out values), ``0`` elsewhere.
        loss_type: ``"mse"`` (squared error) or ``"mae"`` (absolute error).
        epsilon: Stabilizer so an all-zero mask returns ``0`` rather than ``0 / 0``.

    Returns:
        The mask-weighted mean reconstruction error at the masked positions.

    Raises:
        ValueError: If ``loss_type`` is not one of ``("mse", "mae")``.
    """
    if loss_type not in _VALID_LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {_VALID_LOSS_TYPES}, got {loss_type!r}")
    residual = predictions - targets
    error = residual**2 if loss_type == "mse" else abs(residual)
    return (mask * error).sum() / (mask.sum() + epsilon)


class MaskedValueLoss(nnx.Module):
    """Masked-value reconstruction loss as a reusable module.

    Wraps :func:`masked_value_loss` so it composes with other ``nnx`` loss modules. The
    loss holds no parameters; it is a module for interface consistency with the rest of
    :mod:`diffbio.losses`.
    """

    def __init__(self, *, loss_type: str = "mse", rngs: nnx.Rngs | None = None) -> None:
        """Store the reconstruction loss type.

        Args:
            loss_type: ``"mse"`` or ``"mae"``.
            rngs: Optional RNG state (unused; kept for interface consistency).

        Raises:
            ValueError: If ``loss_type`` is not one of ``("mse", "mae")``.
        """
        del rngs
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {_VALID_LOSS_TYPES}, got {loss_type!r}")
        self.loss_type = loss_type

    def __call__(
        self,
        predictions: jax.Array,
        targets: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        """Return the masked-value reconstruction loss for one batch."""
        return masked_value_loss(predictions, targets, mask, loss_type=self.loss_type)
