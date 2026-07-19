"""Differentiable per-gene standardization (z-score) operator.

Standardizes each gene to zero mean and unit variance across cells and clips the
result to a symmetric bound, reproducing the ``scale`` step of the standard
single-cell preprocessing stack. The transform is differentiable, so gradients
flow through it to upstream operators during joint optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jax.typing import ArrayLike

_DEFAULT_CLIP = 10.0


def standardize_features(features: ArrayLike, *, clip: float = _DEFAULT_CLIP) -> jnp.ndarray:
    """Z-score each gene across cells and clip to ``[-clip, clip]``.

    The mean and standard deviation are reduced over the cell axis (axis 0), so a
    gene with zero variance (a constant column) uses a unit denominator and maps
    to zero rather than ``0 / 0``.

    Args:
        features: A ``(n_cells, n_genes)`` matrix.
        clip: Symmetric clipping bound applied after standardization.

    Returns:
        The standardized, clipped matrix, same shape as ``features``.
    """
    features = jnp.asarray(features)
    mean = jnp.mean(features, axis=0, keepdims=True)
    variance = jnp.var(features, axis=0, keepdims=True)
    # Guard the zero-variance case *before* the square root: ``sqrt(0)`` has an
    # infinite gradient, so a constant (e.g. HVG-gated) gene would otherwise emit a
    # NaN gradient that contaminates the whole covariance downstream. Replacing the
    # variance by 1 before the root keeps both the value and the gradient finite.
    std = jnp.sqrt(jnp.where(variance == 0.0, 1.0, variance))
    scaled = (features - mean) / std
    return jnp.clip(scaled, -clip, clip)


@dataclass(frozen=True)
class ScalerConfig(OperatorConfig):
    """Configuration for :class:`DifferentiableScaler`.

    Attributes:
        clip: Symmetric bound applied to the standardized values.
    """

    clip: float = _DEFAULT_CLIP

    def __post_init__(self) -> None:
        """Validate the configuration at construction, failing fast on bad values.

        Raises:
            ValueError: If ``clip`` is not strictly positive.
        """
        super().__post_init__()
        if self.clip <= 0.0:
            raise ValueError(f"clip must be strictly positive, got {self.clip}")


class DifferentiableScaler(OperatorModule):
    """Per-gene standardization with clipping for the preprocessing stack.

    This is a whole-matrix (cross-cell) operator like :class:`DifferentiablePCA`:
    one element is the full ``(n_cells, n_genes)`` matrix, and ``apply`` reduces
    over the cell axis to standardize each gene. Only ``apply`` is implemented, per
    the ``OperatorModule`` contract; the framework's ``apply_batch`` maps it over a
    batch of such matrices (``apply`` is vmap-safe). The operator holds no learnable
    parameters but stays differentiable so gradients reach upstream operators.
    """

    def __init__(
        self,
        config: ScalerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the operator.

        Args:
            config: Scaling configuration.
            rngs: Optional RNG state (unused; kept for interface compatibility).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Standardize ``data["features"]`` in place.

        Args:
            data: Dictionary containing ``"features"`` ``(n_cells, n_genes)``.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where ``output_data``
            replaces ``"features"`` with the standardized, clipped matrix.
        """
        del random_params, stats
        config: ScalerConfig = self.config
        scaled = standardize_features(data["features"], clip=config.clip)
        return {**data, "features": scaled}, state, metadata
