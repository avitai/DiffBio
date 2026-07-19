"""Learnable count-normalization operator.

Differentiable depth normalization followed by a log transform with a learnable
pseudocount and a learnable depth-normalization exponent, so a downstream loss
can tune the normalization end-to-end during joint optimization. With its default
parameter values -- pseudocount 1 and depth exponent 1 -- it reproduces scanpy
``normalize_total(target_sum)`` followed by ``log1p``.

The pseudocount is parameterized through ``softplus`` so it stays strictly
positive under gradient updates, keeping ``log(x + pseudocount)`` finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jax.typing import ArrayLike

_DEFAULT_TARGET_SUM = 1.0e4
_DEFAULT_PSEUDOCOUNT = 1.0


def normalize_counts(
    counts: ArrayLike,
    *,
    pseudocount: ArrayLike = _DEFAULT_PSEUDOCOUNT,
    depth_exponent: ArrayLike = 1.0,
    target_sum: float = _DEFAULT_TARGET_SUM,
) -> jnp.ndarray:
    """Depth-normalize counts and apply a log transform.

    The library size is reduced over the last axis, so this works on a single
    cell ``(n_genes,)`` or a batch ``(n_cells, n_genes)``. With ``pseudocount=1``
    and ``depth_exponent=1`` it equals scanpy ``normalize_total(target_sum)`` +
    ``log1p``.

    Args:
        counts: Raw counts, library size taken over the last axis.
        pseudocount: Strictly-positive offset inside the logarithm.
        depth_exponent: Strength of depth normalization; ``0`` disables it and
            ``1`` applies full ``target_sum`` normalization.
        target_sum: Per-cell target library size.

    Returns:
        The log-normalized expression, same shape as ``counts``.
    """
    counts = jnp.asarray(counts)
    # Reduce in at least float32 so a low-precision (bf16/fp16) input does not lose
    # library-size accuracy, while preserving float64 when the caller opts into it.
    reduce_dtype = jnp.promote_types(counts.dtype, jnp.float32)
    library = jnp.sum(counts, axis=-1, keepdims=True, dtype=reduce_dtype)
    library = jnp.maximum(library, 1.0)
    scale = (target_sum / library) ** depth_exponent
    return jnp.log(counts * scale + pseudocount)


def _inverse_softplus(value: ArrayLike) -> jnp.ndarray:
    """Return the ``softplus`` pre-image; applying ``softplus`` to it recovers ``value``."""
    return jnp.log(jnp.expm1(jnp.asarray(value)))


@dataclass(frozen=True)
class LearnableNormalizationConfig(OperatorConfig):
    """Configuration for :class:`LearnableNormalization`.

    Attributes:
        target_sum: Per-cell target library size for depth normalization.
        pseudocount_init: Initial pseudocount; must be strictly positive so the
            ``softplus`` pre-image is finite. ``1`` gives ``log1p``.
        depth_exponent_init: Initial depth-normalization exponent; ``1`` applies
            full ``target_sum`` normalization and ``0`` disables it.
    """

    target_sum: float = _DEFAULT_TARGET_SUM
    pseudocount_init: float = _DEFAULT_PSEUDOCOUNT
    depth_exponent_init: float = 1.0

    def __post_init__(self) -> None:
        """Validate the configuration at construction, failing fast on bad values.

        Raises:
            ValueError: If ``pseudocount_init`` or ``target_sum`` is not strictly
                positive.
        """
        super().__post_init__()
        if self.pseudocount_init <= 0.0:
            raise ValueError(
                f"pseudocount_init must be strictly positive, got {self.pseudocount_init}"
            )
        if self.target_sum <= 0.0:
            raise ValueError(f"target_sum must be strictly positive, got {self.target_sum}")


class LearnableNormalization(OperatorModule):
    """Per-cell count normalization with a learnable pseudocount and depth exponent.

    This is a per-cell operator (like ``VAENormalizer``): ``apply`` normalizes one
    cell's ``(n_genes,)`` counts, and the framework's ``apply_batch`` vmaps it over
    cells. The learnable ``pseudocount`` and ``depth_exponent`` are shared across
    cells and receive gradients from a downstream loss.
    """

    def __init__(
        self,
        config: LearnableNormalizationConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the operator with learnable normalization parameters.

        Args:
            config: Normalization configuration.
            rngs: Optional RNG state (unused; kept for interface compatibility).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        self.raw_pseudocount = nnx.Param(
            _inverse_softplus(jnp.asarray(config.pseudocount_init, dtype=jnp.float32))
        )
        self.depth_exponent = nnx.Param(jnp.asarray(config.depth_exponent_init, dtype=jnp.float32))

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Normalize ``data["counts"]`` and add ``"normalized"`` to the output.

        Args:
            data: Dictionary containing ``"counts"`` ``(n_genes,)`` for one cell.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where ``output_data`` adds
            the ``"normalized"`` log-normalized expression.
        """
        del random_params, stats
        config: LearnableNormalizationConfig = self.config
        pseudocount = jax.nn.softplus(self.raw_pseudocount[...])
        normalized = normalize_counts(
            data["counts"],
            pseudocount=pseudocount,
            depth_exponent=self.depth_exponent[...],
            target_sum=config.target_sum,
        )
        output_data = {**data, "normalized": normalized}
        return output_data, state, metadata
