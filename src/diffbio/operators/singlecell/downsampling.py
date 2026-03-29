"""Differentiable read-depth downsampling for count data.

Provides a differentiable approximation to binomial read downsampling,
enabling gradient flow through the downsampling operation via a
straight-through estimator.

References:
    - cell-load/src/cell_load/dataset/_perturbation.py (downsampling logic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownsamplingConfig(OperatorConfig):
    """Configuration for ReadDownsampler.

    Attributes:
        mode: Downsampling mode. ``"fraction"`` scales by a fixed fraction.
            ``"target_depth"`` computes per-cell fraction from target read depth.
        fraction: Fraction of reads to keep when ``mode="fraction"``.
        target_depth: Target total reads per cell when ``mode="target_depth"``.
        apply_log1p: Whether to apply log1p to the downsampled counts.
        is_log1p_input: Whether the input counts are already log1p-transformed.
    """

    mode: str = "fraction"
    fraction: float = 1.0
    target_depth: int | None = None
    apply_log1p: bool = True
    is_log1p_input: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Downsampling is stochastic
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "downsample")
        super().__post_init__()


class ReadDownsampler(OperatorModule):
    """Differentiable read-depth downsampler.

    Approximates binomial downsampling using a continuous relaxation that
    allows gradient flow via the straight-through estimator:

    - **Forward:** ``downsampled = floor(counts * fraction)`` plus a
      stochastic rounding of the remainder.
    - **Backward:** Gradients flow through ``counts * fraction`` (the
      expected value).

    Handles log1p domain: if input is log1p-transformed, applies expm1
    before downsampling and log1p after.

    Args:
        config: Downsampling configuration.
        rngs: RNG state for stochastic sampling.
        name: Optional module name.
    """

    def __init__(
        self,
        config: DownsamplingConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(config, rngs=rngs, name=name)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: Any,
        random_params: Any = None,  # noqa: ARG002
        stats: Any = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], dict[str, Any], Any]:
        """Apply differentiable downsampling to count data.

        Args:
            data: Dict with ``"counts"`` key containing expression matrix.
            state: Pipeline state (passed through).
            metadata: Pipeline metadata (passed through).
            random_params: Unused (RNG handled internally).
            stats: Unused.

        Returns:
            Tuple of (updated_data, state, metadata).
        """
        counts = data["counts"]
        config: DownsamplingConfig = self.config  # type: ignore[assignment]

        # Invert log1p if needed
        if config.is_log1p_input:
            counts = jnp.expm1(counts)

        # Compute per-cell fraction
        fraction: float | jnp.ndarray
        if config.mode == "target_depth" and config.target_depth is not None:
            cell_totals = jnp.sum(counts, axis=-1, keepdims=True)
            cell_totals = jnp.maximum(cell_totals, 1.0)  # avoid div by zero
            fraction = jnp.minimum(config.target_depth / cell_totals, 1.0)
        else:
            fraction = config.fraction

        # Straight-through differentiable downsampling
        downsampled = _straight_through_downsample(counts, fraction, self.rngs)

        # Ensure non-negative
        downsampled = jnp.maximum(downsampled, 0.0)

        # Re-apply log1p if configured
        if config.apply_log1p:
            downsampled = jnp.log1p(downsampled)

        return {**data, "counts": downsampled}, state, metadata


def _straight_through_downsample(
    counts: jnp.ndarray,
    fraction: float | jnp.ndarray,
    rngs: nnx.Rngs | None,
) -> jnp.ndarray:
    """Downsample counts with straight-through gradient estimator.

    Forward: ``floor(counts * f) + Bernoulli(counts * f - floor(counts * f))``
    Backward: gradient flows through ``counts * f`` (expected value).

    Args:
        counts: Count matrix.
        fraction: Downsampling fraction (scalar or per-cell array).
        rngs: RNG state for Bernoulli sampling.

    Returns:
        Downsampled count matrix.
    """
    expected = counts * fraction
    floored = jnp.floor(expected)
    remainder = expected - floored

    # Stochastic rounding of remainder
    if rngs is not None and "downsample" in rngs:
        key = rngs.downsample()
    else:
        key = jax.random.key(0)

    uniform = jax.random.uniform(key, shape=remainder.shape)
    rounded = jnp.where(uniform < remainder, 1.0, 0.0)

    # Forward: discrete value. Backward: gradient through expected value.
    discrete = floored + rounded
    return expected + jax.lax.stop_gradient(discrete - expected)
