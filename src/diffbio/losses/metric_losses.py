"""Differentiable metric-based loss functions.

This module provides differentiable approximations of evaluation metrics
so they can be used as training objectives with gradient-based optimizers,
plus exact evaluation metrics backed by calibrax.

Includes:
- DifferentiableAUROC: Sigmoid-approximated Area Under the ROC Curve (training)
- ExactAUROC: Trapezoidal-rule AUROC via calibrax (evaluation)
"""

import jax
import jax.numpy as jnp
from calibrax.metrics.functional.classification import roc_auc
from flax import nnx
from jaxtyping import Array, Float


class DifferentiableAUROC(nnx.Module):
    """Differentiable approximation of the Area Under the ROC Curve.

    This is a smooth training surrogate.  For exact AUROC evaluation use
    :class:`ExactAUROC`, which delegates to calibrax's trapezoidal-rule
    implementation.

    Approximates AUROC by replacing the hard indicator in the Wilcoxon-Mann-Whitney
    statistic with a sigmoid function, making it fully differentiable and
    JIT-compatible.

    For every (positive, negative) pair the hard AUROC checks whether the
    positive score exceeds the negative score.  This module replaces that
    indicator with ``sigmoid((pos - neg) / temperature)``, yielding a smooth
    surrogate whose gradient can drive optimisation.

    Args:
        temperature: Controls sharpness of the sigmoid approximation.
            Lower values approach the hard indicator; higher values give
            smoother gradients.  Default 1.0.

    Example:
        ```python
        auroc_loss = DifferentiableAUROC(temperature=1.0)
        predictions = jnp.array([0.9, 0.8, 0.1, 0.2])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])
        value = auroc_loss(predictions, labels)
        ```
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialise the differentiable AUROC loss.

        Args:
            temperature: Sigmoid temperature.  Lower values produce a
                sharper (closer to hard) approximation.
        """
        super().__init__()
        self.temperature = nnx.Param(jnp.array(temperature))

    def __call__(
        self,
        predictions: Float[Array, " n"],
        labels: Float[Array, " n"],
    ) -> Float[Array, ""]:
        """Compute the differentiable AUROC approximation.

        Args:
            predictions: Model output scores, shape ``(n,)``.
            labels: Binary ground-truth labels (0 or 1), shape ``(n,)``.

        Returns:
            Scalar AUROC approximation in ``[0, 1]``.
        """
        temp = self.temperature[...]

        pos_mask = labels == 1  # (n,)
        neg_mask = labels == 0  # (n,)

        n_pos = jnp.sum(pos_mask)
        n_neg = jnp.sum(neg_mask)

        # Pairwise differences: pos_scores[:, None] - neg_scores[None, :]
        # Built via broadcasting with masks to stay JIT-compatible.
        # pos_vals[i] = predictions[i] where label==1, else 0
        pos_vals = jnp.where(pos_mask, predictions, 0.0)  # (n,)
        neg_vals = jnp.where(neg_mask, predictions, 0.0)  # (n,)

        # Outer difference over all (i, j) pairs
        diffs = pos_vals[:, None] - neg_vals[None, :]  # (n, n)

        # Mask to select only valid (positive_i, negative_j) pairs
        pair_mask = pos_mask[:, None] & neg_mask[None, :]  # (n, n)

        sigmoid_diffs = jax.nn.sigmoid(diffs / temp)  # (n, n)

        # Mean over valid pairs
        auroc = jnp.sum(sigmoid_diffs * pair_mask) / jnp.maximum(n_pos * n_neg, 1.0)

        return auroc


class ExactAUROC(nnx.Module):
    """Exact AUROC metric using calibrax's trapezoidal-rule implementation.

    Delegates to :func:`calibrax.metrics.functional.classification.roc_auc`
    to compute the exact Area Under the ROC Curve via threshold-sweep and
    the trapezoidal rule.

    Use this for evaluation; use :class:`DifferentiableAUROC` for training
    (the sorting-based trapezoidal rule has zero gradients w.r.t. predictions
    because ``argsort`` is not differentiable).

    Example:
        ```python
        exact = ExactAUROC()
        predictions = jnp.array([0.9, 0.8, 0.1, 0.2])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])
        value = exact(predictions, labels)  # 1.0
        ```
    """

    def __init__(self) -> None:
        """Initialise the exact AUROC metric (no learnable parameters)."""
        super().__init__()

    def __call__(
        self,
        predictions: Float[Array, " n"],
        labels: Float[Array, " n"],
    ) -> Float[Array, ""]:
        """Compute the exact AUROC via calibrax.

        Args:
            predictions: Model output scores, shape ``(n,)``.
            labels: Binary ground-truth labels (0 or 1), shape ``(n,)``.

        Returns:
            Scalar AUROC in ``[0, 1]``.
        """
        return roc_auc(predictions, labels)
