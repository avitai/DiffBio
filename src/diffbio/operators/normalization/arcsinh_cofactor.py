"""Learnable per-channel arcsinh cofactor transform for mass-cytometry (CyTOF) data.

CyTOF and flow-cytometry marker intensities are conventionally variance-stabilized with a
fixed ``arcsinh(x / cofactor)`` transform, where the cofactor is a hand-set constant (5 by
convention for CyTOF). This operator makes the cofactor a per-channel learnable parameter --
one value per marker channel, initialized at 5 -- so a downstream gating loss can tune it
end-to-end during joint optimization. At its default initialization it reproduces the frozen
``arcsinh(x / 5)`` transform exactly, giving a clean init-at-frozen baseline for the
frozen-vs-joint study.

The cofactor is parameterized through ``softplus`` so it stays strictly positive under
gradient updates, keeping ``x / cofactor`` finite. Setting ``trainable=False`` freezes the
cofactor by stop-gradienting it, which realizes the frozen arm of the study with the same
operator.
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

_DEFAULT_COFACTOR = 5.0


def arcsinh_transform(intensities: ArrayLike, cofactor: ArrayLike) -> jnp.ndarray:
    """Apply the ``arcsinh(x / cofactor)`` variance-stabilizing transform.

    The cofactor broadcasts over the last (channel) axis, so this works on a single
    cell ``(n_channels,)`` or a batch ``(n_cells, n_channels)`` with a per-channel
    ``(n_channels,)`` cofactor. With a scalar ``cofactor`` of 5 this equals the fixed
    CyTOF ``arcsinh(x / 5)`` convention.

    Args:
        intensities: Marker intensities; the channel axis is the last axis.
        cofactor: Strictly-positive divisor, broadcast over the last axis.

    Returns:
        The transformed intensities, same shape as ``intensities``.
    """
    intensities = jnp.asarray(intensities)
    return jnp.arcsinh(intensities / cofactor)


def _inverse_softplus(value: ArrayLike) -> jnp.ndarray:
    """Return the ``softplus`` pre-image; applying ``softplus`` to it recovers ``value``."""
    return jnp.log(jnp.expm1(jnp.asarray(value)))


@dataclass(frozen=True)
class ArcsinhCofactorConfig(OperatorConfig):
    """Configuration for :class:`ArcsinhCofactor`.

    Attributes:
        num_channels: Number of marker channels; the cofactor holds one parameter per
            channel.
        cofactor_init: Initial cofactor for every channel; must be strictly positive so
            the ``softplus`` pre-image is finite. ``5`` reproduces the CyTOF convention.
        trainable: When ``True`` the cofactor receives gradients (joint arm); when
            ``False`` it is stop-gradiented, freezing the transform (frozen arm).
    """

    num_channels: int = 1
    cofactor_init: float = _DEFAULT_COFACTOR
    trainable: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration at construction, failing fast on bad values.

        Raises:
            ValueError: If ``num_channels`` is not strictly positive or ``cofactor_init``
                is not strictly positive.
        """
        super().__post_init__()
        if self.num_channels <= 0:
            raise ValueError(f"num_channels must be strictly positive, got {self.num_channels}")
        if self.cofactor_init <= 0.0:
            raise ValueError(f"cofactor_init must be strictly positive, got {self.cofactor_init}")


class ArcsinhCofactor(OperatorModule):
    """Per-channel learnable arcsinh cofactor transform for CyTOF/flow intensities.

    This is a per-cell operator (like :class:`LearnableNormalization`): ``apply`` transforms
    one cell's ``(n_channels,)`` intensities, and the framework's ``apply_batch`` vmaps it
    over cells. The learnable ``cofactor`` holds one value per channel, shared across cells,
    and receives gradients from a downstream loss unless ``trainable`` is ``False``.
    """

    def __init__(
        self,
        config: ArcsinhCofactorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the operator with a per-channel learnable cofactor.

        Args:
            config: Arcsinh cofactor configuration.
            rngs: Optional RNG state (unused; kept for interface compatibility).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        raw = _inverse_softplus(
            jnp.full((config.num_channels,), config.cofactor_init, dtype=jnp.float32)
        )
        self.raw_cofactor = nnx.Param(raw)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Transform ``data["intensities"]`` and add ``"transformed"`` to the output.

        Args:
            data: Dictionary containing ``"intensities"`` ``(n_channels,)`` for one cell.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where ``output_data`` adds the
            ``"transformed"`` arcsinh-cofactor intensities.
        """
        del random_params, stats
        config: ArcsinhCofactorConfig = self.config
        cofactor = jax.nn.softplus(self.raw_cofactor[...])
        if not config.trainable:
            cofactor = jax.lax.stop_gradient(cofactor)
        transformed = arcsinh_transform(data["intensities"], cofactor)
        output_data = {**data, "transformed": transformed}
        return output_data, state, metadata
