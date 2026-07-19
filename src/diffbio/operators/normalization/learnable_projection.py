"""Learnable linear projection operator (task-aware dimensionality reduction).

A differentiable, jointly-trainable alternative to the fixed :class:`DifferentiablePCA`
projection. Where PCA picks the maximum-variance (task-agnostic) subspace, this
operator learns the projection end-to-end against a downstream loss, so it can keep
the task-relevant directions instead. In its residual form the projection is
initialized from PCA loadings and learns a correction on top -- so it starts exactly
at the PCA baseline and cannot do worse before training, then adapts.

This is the single-cell analogue of a learnable feature frontend (e.g. LEAF for
audio): jointly optimizing it with a complex head beats a frozen projection once the
dataset is large and the reduction is aggressive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jax.typing import ArrayLike


@dataclass(frozen=True)
class LearnableProjectionConfig(OperatorConfig):
    """Configuration for :class:`LearnableProjection`.

    Attributes:
        n_genes: Input feature dimension (genes).
        n_components: Output (projected) dimension.
    """

    n_genes: int = 2000
    n_components: int = 50

    def __post_init__(self) -> None:
        """Validate the configuration, failing fast on non-positive sizes.

        Raises:
            ValueError: If ``n_genes`` or ``n_components`` is not strictly positive.
        """
        super().__post_init__()
        if self.n_genes <= 0:
            raise ValueError(f"n_genes must be strictly positive, got {self.n_genes}")
        if self.n_components <= 0:
            raise ValueError(f"n_components must be strictly positive, got {self.n_components}")


class LearnableProjection(OperatorModule):
    """Jointly-trainable linear projection of ``(n_cells, n_genes)`` features.

    With ``init_loadings`` (e.g. PCA components) the projection is the residual
    ``basis + delta`` where ``basis`` is a fixed anchor and ``delta`` a learnable
    correction initialized to zero -- so the projection starts exactly at the anchor.
    Without it, the projection is a learnable matrix initialized with a small random
    scale. Only ``apply`` is implemented, per the ``OperatorModule`` contract; it is
    vmap-safe so the framework batches it over a stack of matrices.
    """

    def __init__(
        self,
        config: LearnableProjectionConfig,
        *,
        init_loadings: ArrayLike | None = None,
        rngs: nnx.Rngs,
        name: str | None = None,
    ) -> None:
        """Initialize the projection.

        Args:
            config: Projection configuration.
            init_loadings: Optional ``(n_genes, n_components)`` anchor (e.g. PCA
                loadings); when given the operator learns a zero-initialized residual.
            rngs: RNG state used to initialize a random projection when no anchor is
                supplied.
            name: Optional module name.

        Raises:
            ValueError: If ``init_loadings`` does not have shape
                ``(n_genes, n_components)``.
        """
        super().__init__(config, rngs=rngs, name=name)
        shape = (config.n_genes, config.n_components)
        if init_loadings is not None:
            anchor = jnp.asarray(init_loadings, dtype=jnp.float32)
            if anchor.shape != shape:
                raise ValueError(f"init_loadings must have shape {shape}, got {anchor.shape}")
            self.basis = nnx.Variable(anchor)
            self.delta = nnx.Param(jnp.zeros(shape, dtype=jnp.float32))
        else:
            key = rngs.params() if "params" in rngs else jax.random.key(0)
            self.basis = nnx.Variable(jnp.zeros(shape, dtype=jnp.float32))
            self.delta = nnx.Param(jax.random.normal(key, shape) / math.sqrt(config.n_genes))
        self.projection_bias = nnx.Param(jnp.zeros(config.n_components, dtype=jnp.float32))

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Project ``data["features"]`` onto the learnable subspace.

        Args:
            data: Dictionary containing ``"features"`` ``(n_cells, n_genes)``.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where ``output_data`` adds
            ``"projection"`` ``(n_cells, n_components)``.
        """
        del random_params, stats
        projection = self.basis[...] + self.delta[...]
        embedded = jnp.asarray(data["features"]) @ projection + self.projection_bias[...]
        return {**data, "projection": embedded}, state, metadata
