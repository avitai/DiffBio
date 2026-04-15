"""FNO-based differentiable peak calling for ChIP-seq and ATAC-seq data.

Uses a Fourier Neural Operator (FNO) from opifex to learn the mapping
from coverage signals to peak probabilities. The FNO captures multi-scale
patterns in the frequency domain, making it well-suited for detecting
peaks of varying widths without explicit multi-scale CNN kernels.

This is an alternative to the CNN-based ``DifferentiablePeakCaller``
in ``peak_calling.py``. The FNO approach processes the entire signal
in one pass via spectral convolutions rather than sliding windows.

Reference:
    Li et al. (2021) "Fourier Neural Operator for Parametric Partial
    Differential Equations". ICLR 2021.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx

from diffbio.core import soft_ops
from jaxtyping import PyTree
from opifex.neural.operators import FourierNeuralOperator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FNOPeakCallerConfig(OperatorConfig):
    """Configuration for FNO-based peak caller.

    Attributes:
        hidden_channels: Number of hidden channels in FNO layers.
        modes: Number of Fourier modes to retain (controls frequency resolution).
        num_layers: Number of FNO layers.
        threshold: Initial soft threshold for peak classification.
        temperature: Temperature for sigmoid smoothing.
    """

    hidden_channels: int = 32
    modes: int = 16
    num_layers: int = 4
    threshold: float = 0.5
    temperature: float = 1.0


class FNOPeakCaller(OperatorModule):
    """FNO-based differentiable peak caller.

    Applies a Fourier Neural Operator to map coverage signals to peak
    probability scores. The FNO learns spectral convolution kernels that
    capture peak patterns at multiple frequency scales simultaneously.

    Input data:
        - ``"coverage"``: Coverage signal ``(batch, length)`` or ``(length,)``.

    Output adds:
        - ``"peak_scores"``: Raw FNO output scores ``(batch, length)``.
        - ``"peak_probabilities"``: Sigmoid-transformed scores in [0, 1].

    Args:
        config: FNOPeakCallerConfig with FNO parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = FNOPeakCallerConfig(hidden_channels=32, modes=16)
        >>> op = FNOPeakCaller(config, rngs=nnx.Rngs(42))
        >>> result, _, _ = op.apply({"coverage": coverage}, {}, None)
        >>> peaks = result["peak_probabilities"] > 0.5
    """

    def __init__(
        self,
        config: FNOPeakCallerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize FNO peak caller."""
        super().__init__(config, rngs=rngs, name=name)
        self.config: FNOPeakCallerConfig = config

        if rngs is None:
            rngs = nnx.Rngs(0)

        # FNO: 1 input channel (coverage) -> 1 output channel (peak score)
        self.fno = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=config.hidden_channels,
            modes=config.modes,
            num_layers=config.num_layers,
            # Opifex now defaults FNOs to 2D spectral layers; this operator is 1D.
            spatial_dims=1,
            rngs=rngs,
        )

        self.threshold = nnx.Param(jnp.array(config.threshold))

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply FNO peak detection to coverage signal.

        Args:
            data: Dict with ``"coverage"`` key — shape ``(batch, length)``
                or ``(length,)`` for unbatched.
            state: Element state (passed through).
            metadata: Element metadata (passed through).
            random_params: Unused.
            stats: Unused.

        Returns:
            Tuple of (output_data, state, metadata).
        """
        coverage = data["coverage"]
        unbatched = coverage.ndim == 1
        if unbatched:
            coverage = coverage[None, :]

        # FNO expects (batch, channels, *spatial_dims) — channels-first
        x = coverage[:, None, :]  # (batch, 1, length)
        scores = self.fno(x)  # (batch, 1, length)
        scores = scores[:, 0, :]  # (batch, length)

        # Soft peak classification
        temp = self.config.temperature
        probs = soft_ops.greater(scores, self.threshold[...], softness=temp)

        if unbatched:
            scores = scores.squeeze(0)
            probs = probs.squeeze(0)

        output_data = {
            **data,
            "peak_scores": scores,
            "peak_probabilities": probs,
        }

        return output_data, state, metadata
