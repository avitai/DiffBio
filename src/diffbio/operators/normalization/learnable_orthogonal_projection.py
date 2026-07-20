"""Learnable orthonormal projection on the Stiefel manifold (eigensolver-free).

Like :class:`LearnableProjection`, this learns a task-aware reduction initialized at the
PCA loadings and trained jointly with the downstream model. It differs in one respect:
the projection is kept orthonormal throughout training by a QR retraction of a
PCA-anchored residual, ``Q(V + Delta)`` with the columns re-orthonormalized every forward
pass, so the learned directions stay decorrelated (a genuine Stiefel-manifold point) rather
than drifting into a correlated, unconstrained basis.

This is the ``learn-directions'' arm the differentiable-PCA literature recommends when one
wants a trainable orthonormal basis without differentiating an eigensolver: because the
retraction is a QR, not an eigen/SVD decomposition, the notorious ``1 / (lambda_i -
lambda_j)`` eigengap term never appears, so gradients stay finite even where a
differentiated eigendecomposition would diverge (Li et al., ICLR 2020, Cayley/Stiefel
optimization; here realized with the simpler QR retraction). The delta is initialized to
zero, so training starts exactly at the (orthonormal) PCA baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jax.typing import ArrayLike


@dataclass(frozen=True)
class LearnableOrthogonalProjectionConfig(OperatorConfig):
    """Configuration for :class:`LearnableOrthogonalProjection`.

    Attributes:
        n_features: Input feature dimension.
        n_components: Output (projected) dimension.
    """

    n_features: int = 2000
    n_components: int = 50

    def __post_init__(self) -> None:
        """Validate the configuration, failing fast on non-positive sizes.

        Raises:
            ValueError: If ``n_features`` or ``n_components`` is not strictly positive.
        """
        super().__post_init__()
        if self.n_features <= 0:
            raise ValueError(f"n_features must be strictly positive, got {self.n_features}")
        if self.n_components <= 0:
            raise ValueError(f"n_components must be strictly positive, got {self.n_components}")


class LearnableOrthogonalProjection(OperatorModule):
    """Jointly-trainable orthonormal projection of ``(n_samples, n_features)`` features.

    The projection is ``orthonormalize(anchor + delta)`` where ``anchor`` is a fixed
    orthonormal frame (e.g. PCA loadings) and ``delta`` a learnable residual initialized to
    zero, so the basis starts exactly at the anchor and stays on the Stiefel manifold as
    ``delta`` moves. Only ``apply`` is implemented, per the ``OperatorModule`` contract; it
    is vmap-safe so the framework batches it over a stack of matrices.
    """

    def __init__(
        self,
        config: LearnableOrthogonalProjectionConfig,
        *,
        init_loadings: ArrayLike,
        rngs: nnx.Rngs,
        name: str | None = None,
    ) -> None:
        """Initialize the projection at an orthonormal anchor.

        Args:
            config: Projection configuration.
            init_loadings: ``(n_features, n_components)`` orthonormal anchor (e.g. PCA
                loadings); the operator learns a zero-initialized residual on top.
            rngs: RNG state (kept for interface compatibility).
            name: Optional module name.

        Raises:
            ValueError: If ``init_loadings`` does not have shape
                ``(n_features, n_components)``.
        """
        super().__init__(config, rngs=rngs, name=name)
        shape = (config.n_features, config.n_components)
        anchor = jnp.asarray(init_loadings, dtype=jnp.float32)
        if anchor.shape != shape:
            raise ValueError(f"init_loadings must have shape {shape}, got {anchor.shape}")
        self.anchor = nnx.Variable(anchor)
        self.delta = nnx.Param(jnp.zeros(shape, dtype=jnp.float32))
        self.projection_bias = nnx.Param(jnp.zeros(config.n_components, dtype=jnp.float32))

    def orthonormal_basis(self) -> jnp.ndarray:
        """Return the current orthonormal projection basis via a sign-fixed QR retraction."""
        matrix = self.anchor[...] + self.delta[...]
        orthonormal, upper = jnp.linalg.qr(matrix)
        signs = jnp.sign(jnp.diagonal(upper))
        signs = jnp.where(signs == 0.0, 1.0, signs)
        return orthonormal * signs[None, :]

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Project ``data["features"]`` onto the learnable orthonormal subspace.

        Args:
            data: Dictionary containing ``"features"`` ``(n_samples, n_features)``.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where ``output_data`` adds
            ``"projection"`` ``(n_samples, n_components)``.
        """
        del random_params, stats
        embedded = jnp.asarray(data["features"]) @ self.orthonormal_basis()
        return {**data, "projection": embedded + self.projection_bias[...]}, state, metadata
