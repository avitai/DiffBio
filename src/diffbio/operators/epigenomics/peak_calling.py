"""Differentiable peak calling for ChIP-seq and ATAC-seq data.

This module implements a CNN-based differentiable peak caller that can be
used for ChIP-seq and ATAC-seq analysis with end-to-end gradient flow.
"""

from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule


@dataclass
class PeakCallerConfig(OperatorConfig):
    """Configuration for differentiable peak caller.

    Attributes:
        window_size: Size of the sliding window for peak detection.
        num_filters: Number of CNN filters for peak pattern detection.
        kernel_sizes: List of kernel sizes for multi-scale detection.
        threshold: Initial threshold for peak calling.
        temperature: Temperature for sigmoid smoothing.
        min_peak_width: Minimum width for called peaks.
        stochastic: Whether to use stochastic operations.
        stream_name: Name of the data stream to process.
    """

    window_size: int = 200
    num_filters: int = 32
    kernel_sizes: tuple[int, ...] = (5, 11, 21)
    threshold: float = 0.5
    temperature: float = 1.0
    min_peak_width: int = 50
    stochastic: bool = False
    stream_name: str | None = None


class PeakDetectionCNN(nnx.Module):
    """CNN module for detecting peak patterns in coverage signals.

    Uses multi-scale convolutions to capture peaks of varying widths.
    """

    def __init__(
        self,
        num_filters: int = 32,
        kernel_sizes: tuple[int, ...] = (5, 11, 21),
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the peak detection CNN.

        Args:
            num_filters: Number of filters per convolution layer.
            kernel_sizes: Kernel sizes for multi-scale detection.
            rngs: Random number generators for initialization.
        """
        super().__init__()
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes

        # Multi-scale convolution layers - use nnx.List for proper parameter tracking
        self.conv_layers = nnx.List(
            [
                nnx.Conv(
                    in_features=1,
                    out_features=num_filters,
                    kernel_size=(kernel_size,),
                    padding="SAME",
                    rngs=rngs,
                )
                for kernel_size in kernel_sizes
            ]
        )

        # Combine multi-scale features
        total_features = num_filters * len(kernel_sizes)
        self.combine_conv = nnx.Conv(
            in_features=total_features,
            out_features=num_filters,
            kernel_size=(3,),
            padding="SAME",
            rngs=rngs,
        )

        # Final prediction layer
        self.output_conv = nnx.Conv(
            in_features=num_filters,
            out_features=1,
            kernel_size=(1,),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass for peak detection.

        Args:
            x: Input coverage signal of shape (batch, length, 1) or (batch, length).

        Returns:
            Peak scores of shape (batch, length).
        """
        # Ensure input has channel dimension
        if x.ndim == 2:
            x = x[..., None]

        # Multi-scale convolutions
        features = []
        for conv in self.conv_layers:
            feat = nnx.relu(conv(x))
            features.append(feat)

        # Concatenate multi-scale features
        combined = jnp.concatenate(features, axis=-1)

        # Combine and predict
        x = nnx.relu(self.combine_conv(combined))
        scores = self.output_conv(x)

        return scores.squeeze(-1)


class DifferentiablePeakCaller(OperatorModule):
    """Differentiable peak caller for ChIP-seq and ATAC-seq data.

    This operator uses a CNN-based approach to detect peaks in coverage
    signals, with soft thresholding for end-to-end differentiability.

    The operator processes coverage data and outputs:
    - Peak probabilities at each position
    - Peak boundaries (soft)
    - Peak summits

    Example:
        ```python
        config = PeakCallerConfig(
            window_size=200,
            num_filters=32,
            threshold=0.5,
        )
        peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

        data = {"coverage": coverage_signal}
        result, state, metadata = peak_caller.apply(data, {}, None)
        peak_probs = result["peak_probabilities"]
        ```
    """

    def __init__(self, config: PeakCallerConfig, *, rngs: nnx.Rngs | None = None):
        """Initialize the differentiable peak caller.

        Args:
            config: Configuration for the peak caller.
            rngs: Random number generators for initialization.
        """
        super().__init__(config, rngs=rngs)
        self.config = config

        # Initialize RNGs if not provided
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Learnable threshold parameter
        self.threshold = nnx.Param(jnp.array(config.threshold))

        # Learnable temperature parameter
        self.temperature = nnx.Param(jnp.array(config.temperature))

        # Peak detection CNN
        self.peak_cnn = PeakDetectionCNN(
            num_filters=config.num_filters,
            kernel_sizes=config.kernel_sizes,
            rngs=rngs,
        )

        # Local maximum detection kernel (for summit finding)
        self.summit_kernel_size = config.min_peak_width

    def _soft_local_max(self, scores: jax.Array, window_size: int, temperature: float) -> jax.Array:
        """Compute soft local maximum indicator.

        Args:
            scores: Peak scores of shape (batch, length).
            window_size: Window size for local maximum detection.
            temperature: Temperature for softmax.

        Returns:
            Soft local maximum indicators of shape (batch, length).
        """
        _, length = scores.shape
        half_window = window_size // 2

        # Pad scores for windowed comparison
        padded = jnp.pad(
            scores, ((0, 0), (half_window, half_window)), mode="constant", constant_values=-jnp.inf
        )

        # Extract windows around each position
        def extract_windows(padded_row):
            indices = jnp.arange(length)
            windows = jax.vmap(lambda i: jax.lax.dynamic_slice(padded_row, (i,), (window_size,)))(
                indices
            )
            return windows

        windows = jax.vmap(extract_windows)(padded)  # (batch, length, window_size)

        # Compute softmax over window to find local maximum
        # The center position should have high weight if it's the maximum
        center_idx = half_window
        window_softmax = jax.nn.softmax(windows / temperature, axis=-1)

        # Extract probability of center being the maximum
        local_max_prob = window_softmax[:, :, center_idx]

        return local_max_prob

    def _compute_peak_boundaries(self, peak_probs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute soft peak boundaries.

        Args:
            peak_probs: Peak probabilities of shape (batch, length).

        Returns:
            Tuple of (peak_starts, peak_ends) as soft indicators.
        """
        # Compute gradient of peak probabilities
        # Rising edge indicates start, falling edge indicates end
        grad = jnp.diff(peak_probs, axis=-1, prepend=peak_probs[:, :1])

        # Soft peak starts (positive gradient)
        peak_starts = jax.nn.sigmoid(grad * 10.0)  # Sharp transition

        # Soft peak ends (negative gradient)
        peak_ends = jax.nn.sigmoid(-grad * 10.0)

        return peak_starts, peak_ends

    def apply(
        self,
        data: dict,
        state: dict,
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Apply peak calling to coverage data.

        Args:
            data: Dictionary containing:
                - 'coverage': Coverage signal of shape (batch, length) or (length,)
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of (output_data, state, metadata) where output_data contains:
                - 'coverage': Original coverage signal
                - 'peak_scores': Raw peak detection scores
                - 'peak_probabilities': Soft peak probabilities
                - 'peak_summits': Soft summit indicators
                - 'peak_starts': Soft peak start indicators
                - 'peak_ends': Soft peak end indicators
        """
        del random_params, stats  # Unused

        coverage = data["coverage"]

        # Handle single sequence input
        single_input = coverage.ndim == 1
        if single_input:
            coverage = coverage[None, :]

        # Get peak scores from CNN
        peak_scores = self.peak_cnn(coverage)

        # Apply soft threshold
        temperature = jnp.abs(self.temperature.value) + 1e-6
        peak_probs = jax.nn.sigmoid((peak_scores - self.threshold.value) / temperature)

        # Find soft summits (local maxima)
        summit_probs = self._soft_local_max(
            peak_scores * peak_probs,  # Weight by peak probability
            self.summit_kernel_size,
            temperature,
        )

        # Compute soft peak boundaries
        peak_starts, peak_ends = self._compute_peak_boundaries(peak_probs)

        # Remove batch dimension if input was single
        if single_input:
            peak_scores = peak_scores[0]
            peak_probs = peak_probs[0]
            summit_probs = summit_probs[0]
            peak_starts = peak_starts[0]
            peak_ends = peak_ends[0]
            coverage = coverage[0]

        output_data = {
            **data,
            "peak_scores": peak_scores,
            "peak_probabilities": peak_probs,
            "peak_summits": summit_probs,
            "peak_starts": peak_starts,
            "peak_ends": peak_ends,
        }

        return output_data, state, metadata
