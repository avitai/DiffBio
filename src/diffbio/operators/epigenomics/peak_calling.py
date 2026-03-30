"""Differentiable peak calling for ChIP-seq and ATAC-seq data.

This module implements a CNN-based differentiable peak caller that can be
used for ChIP-seq and ATAC-seq analysis with end-to-end gradient flow.

Optionally includes a VAE-based denoising stage (inspired by SCALE) that
encodes the coverage signal into a latent space and decodes it using a
Poisson decoder before peak detection.

Inherits from TemperatureOperator to get:

- _temperature property for temperature-controlled smoothing
- soft_max() for logsumexp-based smooth maximum
- soft_argmax() for soft position selection
"""

import logging
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from artifex.generative_models.core.losses.divergence import gaussian_kl_divergence
from datarax.core.config import OperatorConfig

from diffbio.core import soft_ops
from diffbio.core.base_operators import TemperatureOperator
from diffbio.utils.nn_utils import ensure_rngs, get_rng_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeakCallerConfig(OperatorConfig):
    """Configuration for differentiable peak caller.

    Attributes:
        window_size: Size of the sliding window for peak detection.
        num_filters: Number of CNN filters for peak pattern detection.
        kernel_sizes: List of kernel sizes for multi-scale detection.
        threshold: Initial threshold for peak calling.
        temperature: Temperature for sigmoid smoothing.
        min_peak_width: Minimum width for called peaks.
        use_vae_denoising: Whether to apply VAE-based denoising before
            peak detection. When enabled, the coverage signal is encoded
            to a latent space and decoded with a Poisson decoder.
        vae_latent_dim: Latent space dimension for VAE denoiser.
        vae_hidden_dim: Hidden layer dimension for VAE encoder/decoder.
        stream_name: Name of the data stream to process.
    """

    window_size: int = 200
    num_filters: int = 32
    kernel_sizes: tuple[int, ...] = (5, 11, 21)
    threshold: float = 0.5
    temperature: float = 1.0
    learnable_temperature: bool = True
    min_peak_width: int = 50
    use_vae_denoising: bool = False
    vae_latent_dim: int = 16
    vae_hidden_dim: int = 64


class SignalVAE(nnx.Module):
    """VAE for denoising coverage signals with Poisson decoder.

    Inspired by SCALE (Single-Cell ATAC-seq Analysis via Latent feature
    Extraction), this module uses a VAE with a Poisson decoder to denoise
    coverage signals. The Poisson distribution naturally models count data
    (read coverage) and the latent space captures the underlying signal.

    Architecture:
        Encoder: coverage -> hidden -> (mean, logvar)
        Reparameterize: z = mean + exp(0.5 * logvar) * epsilon
        Decoder: z -> hidden -> log_rate (Poisson parameter)
    """

    def __init__(
        self,
        signal_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the signal VAE.

        Args:
            signal_dim: Dimension of the input signal (length).
            latent_dim: Dimension of the latent space.
            hidden_dim: Dimension of hidden layers.
            rngs: Random number generators for initialization.
        """
        super().__init__()
        self.signal_dim = signal_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_hidden = nnx.Linear(signal_dim, hidden_dim, rngs=rngs)
        self.enc_mean = nnx.Linear(hidden_dim, latent_dim, rngs=rngs)
        self.enc_logvar = nnx.Linear(hidden_dim, latent_dim, rngs=rngs)

        # Decoder (Poisson: outputs log-rate)
        self.dec_hidden = nnx.Linear(latent_dim, hidden_dim, rngs=rngs)
        self.dec_output = nnx.Linear(hidden_dim, signal_dim, rngs=rngs)

    def encode(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Encode signal to latent distribution parameters.

        Args:
            x: Input signal of shape (..., signal_dim).

        Returns:
            Tuple of (mean, logvar) each of shape (..., latent_dim).
        """
        h = nnx.relu(self.enc_hidden(x))
        mean = self.enc_mean(h)
        logvar = jnp.clip(self.enc_logvar(h), -10.0, 10.0)
        return mean, logvar

    def decode(self, z: jax.Array) -> jax.Array:
        """Decode latent representation to Poisson log-rate.

        Args:
            z: Latent representation of shape (..., latent_dim).

        Returns:
            Log-rate for Poisson decoder of shape (..., signal_dim).
        """
        h = nnx.relu(self.dec_hidden(z))
        log_rate = self.dec_output(h)
        return log_rate

    def __call__(
        self,
        x: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Forward pass: encode, sample, decode.

        Args:
            x: Input signal of shape (..., signal_dim).
            rng_key: JAX PRNG key for reparameterization sampling.

        Returns:
            Tuple of (denoised, mean, logvar, log_rate):
                - denoised: Denoised signal (exp(log_rate)), shape (..., signal_dim).
                - mean: Latent mean, shape (..., latent_dim).
                - logvar: Latent log-variance, shape (..., latent_dim).
                - log_rate: Raw Poisson log-rate, shape (..., signal_dim).
        """
        mean, logvar = self.encode(x)

        # Reparameterization trick
        std = jnp.exp(0.5 * logvar)
        epsilon = jax.random.normal(rng_key, mean.shape)
        z = mean + std * epsilon

        log_rate = self.decode(z)
        # Poisson rate = exp(log_rate), clamp for stability
        denoised = jnp.exp(jnp.clip(log_rate, -10.0, 10.0))

        return denoised, mean, logvar, log_rate


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


class DifferentiablePeakCaller(TemperatureOperator):
    """Differentiable peak caller for ChIP-seq and ATAC-seq data.

    This operator uses a CNN-based approach to detect peaks in coverage
    signals, with soft thresholding for end-to-end differentiability.

    Optionally applies VAE-based denoising before peak detection, using a
    Poisson decoder (per SCALE) to model count data. When VAE denoising
    is enabled, the pipeline is:
        coverage -> VAE encoder -> latent -> Poisson decoder -> denoised -> CNN -> peaks

    The operator processes coverage data and outputs:
    - Peak probabilities at each position
    - Peak boundaries (soft)
    - Peak summits
    - Denoised coverage (when VAE is enabled)

    Example:
        ```python
        config = PeakCallerConfig(
            window_size=200,
            num_filters=32,
            threshold=0.5,
            use_vae_denoising=True,
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

        self._rngs = ensure_rngs(rngs)

        # Learnable threshold parameter
        self.threshold = nnx.Param(jnp.array(config.threshold))

        # Temperature is managed by TemperatureOperator via self._temperature

        # Peak detection CNN
        self.peak_cnn = PeakDetectionCNN(
            num_filters=config.num_filters,
            kernel_sizes=config.kernel_sizes,
            rngs=rngs,
        )

        # Local maximum detection kernel (for summit finding)
        self.summit_kernel_size = config.min_peak_width

        # Optional VAE denoiser
        self._use_vae = config.use_vae_denoising
        if self._use_vae:
            # Signal dim is set per-call since it depends on input length,
            # but we pre-initialize the VAE with a fixed config dim.
            # For position-independent denoising, we use a 1D VAE per position
            # applied across the signal length. This uses a fixed-dim VAE.
            self.vae_encoder = SignalVAE(
                signal_dim=config.window_size,
                latent_dim=config.vae_latent_dim,
                hidden_dim=config.vae_hidden_dim,
                rngs=rngs,
            )

    def _vae_denoise(self, coverage: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Apply VAE-based denoising to coverage signal.

        Processes each sample in the batch independently through the VAE.
        The signal is reshaped into windows of size window_size, denoised
        per-window, and reassembled.

        Args:
            coverage: Coverage signal of shape (batch, length).

        Returns:
            Tuple of (denoised_coverage, kl_loss):
                - denoised_coverage: Denoised signal of shape (batch, length).
                - kl_loss: Scalar KL divergence loss.
        """
        batch_size, length = coverage.shape
        window_size = self.config.window_size

        # Pad signal to be divisible by window_size
        remainder = length % window_size
        if remainder != 0:
            pad_amount = window_size - remainder
            coverage_padded = jnp.pad(coverage, ((0, 0), (0, pad_amount)), mode="edge")
        else:
            pad_amount = 0
            coverage_padded = coverage

        padded_length = coverage_padded.shape[1]
        num_windows = padded_length // window_size

        # Reshape to (batch * num_windows, window_size)
        windows = coverage_padded.reshape(batch_size * num_windows, window_size)

        # Log-transform for VAE input (coverage is count-like data)
        vae_input = jnp.log1p(jnp.abs(windows))

        # Get RNG key for sampling
        rng_key = get_rng_key(self._rngs, "sample", fallback_seed=0)

        # Run VAE forward pass
        denoised_windows, mean, logvar, _ = self.vae_encoder(vae_input, rng_key)

        # KL divergence via artifex
        kl_loss = gaussian_kl_divergence(mean, logvar, reduction="sum")

        # Reassemble signal
        denoised_padded = denoised_windows.reshape(batch_size, padded_length)

        # Remove padding
        denoised = denoised_padded[:, :length]

        return denoised, kl_loss

    def _soft_local_max(
        self, scores: jax.Array, window_size: int, temperature: jax.Array | float
    ) -> jax.Array:
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
        def extract_windows(padded_row: jax.Array) -> jax.Array:
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
        peak_starts = soft_ops.greater(grad, 0.0, softness=0.1)

        # Soft peak ends (negative gradient)
        peak_ends = soft_ops.less(grad, 0.0, softness=0.1)

        return peak_starts, peak_ends

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Apply peak calling to coverage data.

        When VAE denoising is enabled, the coverage signal is first denoised
        through a VAE with Poisson decoder before peak detection.

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
                - 'denoised_coverage': Denoised signal (only when VAE enabled)
                - 'vae_kl_loss': KL divergence loss (only when VAE enabled)
        """
        del random_params, stats  # Unused

        coverage = data["coverage"]

        # Handle single sequence input
        single_input = coverage.ndim == 1
        if single_input:
            coverage = coverage[None, :]

        # Optional VAE denoising
        vae_extras: dict[str, Any] = {}
        if self._use_vae:
            denoised, kl_loss = self._vae_denoise(coverage)
            vae_extras["denoised_coverage"] = denoised
            vae_extras["vae_kl_loss"] = kl_loss
            # Use denoised signal for peak detection
            cnn_input = denoised
        else:
            cnn_input = coverage

        # Get peak scores from CNN
        peak_scores = self.peak_cnn(cnn_input)

        # Apply soft threshold
        temperature = jnp.abs(self._temperature) + 1e-6
        peak_probs = soft_ops.greater(peak_scores, self.threshold[...], softness=temperature)

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
            for key in list(vae_extras.keys()):
                if key == "denoised_coverage":
                    vae_extras[key] = vae_extras[key][0]

        output_data = {
            **data,
            "peak_scores": peak_scores,
            "peak_probabilities": peak_probs,
            "peak_summits": summit_probs,
            "peak_starts": peak_starts,
            "peak_ends": peak_ends,
            **vae_extras,
        }

        return output_data, state, metadata
