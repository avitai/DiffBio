"""Differentiable soft centroiding for profile-mode mass spectra.

Classic centroiding (e.g. XCMS centWave) detects peaks in a profile-mode spectrum with
hand-set knobs -- a signal-to-noise threshold, an expected peak width, and a local-maximum
rule -- and collapses each peak to a single ``(m/z, intensity)`` centroid before any
downstream identification. Those knobs are frozen and the collapse is lossy.

This operator makes the peak picker differentiable and its knobs learnable, so a downstream
identification loss can tune them end-to-end. A profile spectrum is smoothed with a
learnable-width Gaussian kernel, gated by a soft signal-to-noise threshold (``sigmoid`` of the
smoothed intensity above ``snthresh * noise``) and a soft local-maximum weight, and the peak
position is read off as the intensity-weighted centroid (a soft-argmax over m/z that resolves
the peak below the grid spacing). The learnable ``snthresh`` and peak width receive gradients;
freezing them (``trainable=False``) recovers a fixed-knob baseline for a frozen-vs-joint study.
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

_DEFAULT_SNTHRESH = 3.0
_DEFAULT_PEAK_WIDTH = 2.0
_DEFAULT_NOISE_QUANTILE = 0.5
_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_KERNEL_RADIUS = 15
_EPSILON = 1.0e-8


def mz_grid(min_mz: float, max_mz: float, n_points: int) -> jnp.ndarray:
    """Return the ``(n_points,)`` m/z grid spanning ``[min_mz, max_mz]`` inclusive."""
    return jnp.linspace(min_mz, max_mz, n_points)


def _inverse_softplus(value: ArrayLike) -> jnp.ndarray:
    """Return the ``softplus`` pre-image; applying ``softplus`` to it recovers ``value``."""
    return jnp.log(jnp.expm1(jnp.asarray(value)))


def _gaussian_smooth(intensity: jnp.ndarray, width: jnp.ndarray, radius: int) -> jnp.ndarray:
    """Convolve with a fixed-support Gaussian kernel of learnable ``width`` (grid points)."""
    offsets = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    kernel = jnp.exp(-0.5 * (offsets / width) ** 2)
    kernel = kernel / jnp.sum(kernel)
    return jnp.convolve(intensity, kernel, mode="same")


def _soft_local_maximum(smoothed: jnp.ndarray, temperature: float) -> jnp.ndarray:
    """Soft weight (in ``[0, 1]``) for each point exceeding both grid neighbors."""
    previous = jnp.concatenate([smoothed[:1], smoothed[:-1]])
    following = jnp.concatenate([smoothed[1:], smoothed[-1:]])
    higher_than_previous = jax.nn.sigmoid((smoothed - previous) / temperature)
    higher_than_following = jax.nn.sigmoid((smoothed - following) / temperature)
    return higher_than_previous * higher_than_following


@dataclass(frozen=True)
class SoftCentroiderConfig(OperatorConfig):
    """Configuration for :class:`SoftCentroider`.

    Attributes:
        min_mz: Lower bound of the m/z grid.
        max_mz: Upper bound of the m/z grid.
        n_points: Number of profile grid points.
        snthresh_init: Initial signal-to-noise threshold (strictly positive).
        peak_width_init: Initial Gaussian smoothing width in grid points (strictly positive).
        noise_quantile: Quantile of the spectrum used as the noise estimate.
        temperature: Sharpness of the soft threshold and local-maximum gates.
        kernel_radius: Fixed half-width of the smoothing kernel support.
        trainable: When ``False`` the learnable knobs are stop-gradiented (frozen arm).
    """

    min_mz: float = 0.0
    max_mz: float = 1000.0
    n_points: int = 1000
    snthresh_init: float = _DEFAULT_SNTHRESH
    peak_width_init: float = _DEFAULT_PEAK_WIDTH
    noise_quantile: float = _DEFAULT_NOISE_QUANTILE
    temperature: float = _DEFAULT_TEMPERATURE
    kernel_radius: int = _DEFAULT_KERNEL_RADIUS
    trainable: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration at construction, failing fast on bad values.

        Raises:
            ValueError: If the m/z range is empty, sizes are non-positive, the knobs are
                not strictly positive, or ``noise_quantile`` is outside ``[0, 1]``.
        """
        super().__post_init__()
        if self.max_mz <= self.min_mz:
            raise ValueError(f"max_mz must exceed min_mz, got {self.max_mz} <= {self.min_mz}")
        if self.n_points <= 0:
            raise ValueError(f"n_points must be strictly positive, got {self.n_points}")
        if self.snthresh_init <= 0.0:
            raise ValueError(f"snthresh_init must be strictly positive, got {self.snthresh_init}")
        if self.peak_width_init <= 0.0:
            raise ValueError(
                f"peak_width_init must be strictly positive, got {self.peak_width_init}"
            )
        if not 0.0 <= self.noise_quantile <= 1.0:
            raise ValueError(f"noise_quantile must be in [0, 1], got {self.noise_quantile}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive, got {self.temperature}")
        if self.kernel_radius <= 0:
            raise ValueError(f"kernel_radius must be strictly positive, got {self.kernel_radius}")


class SoftCentroider(OperatorModule):
    """Differentiable profile-mode peak picker with learnable S/N threshold and width.

    This is a per-spectrum operator: ``apply`` centroids one spectrum's ``(n_points,)``
    profile intensities, and the framework's ``apply_batch`` vmaps it over spectra. The
    learnable ``snthresh`` and peak width are shared across spectra and receive gradients
    from a downstream loss unless ``trainable`` is ``False``.
    """

    def __init__(
        self,
        config: SoftCentroiderConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the operator with learnable, softplus-positive peak-picking knobs.

        Args:
            config: Centroiding configuration.
            rngs: Optional RNG state (unused; kept for interface compatibility).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        self.raw_snthresh = nnx.Param(
            _inverse_softplus(jnp.asarray(config.snthresh_init, dtype=jnp.float32))
        )
        self.raw_peak_width = nnx.Param(
            _inverse_softplus(jnp.asarray(config.peak_width_init, dtype=jnp.float32))
        )
        self.mz = mz_grid(config.min_mz, config.max_mz, config.n_points)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Soft-centroid ``data["intensity"]`` and add peak weights, profile, and centroid.

        Args:
            data: Dictionary containing ``"intensity"`` ``(n_points,)`` for one spectrum.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where ``output_data`` adds
            ``"peak_weights"`` (soft mask in ``[0, 1]``), ``"centroided"`` (the gated
            profile), and ``"centroid_mz"`` (the intensity-weighted peak position).
        """
        del random_params, stats
        config: SoftCentroiderConfig = self.config
        snthresh = jax.nn.softplus(self.raw_snthresh[...])
        peak_width = jax.nn.softplus(self.raw_peak_width[...])
        if not config.trainable:
            snthresh = jax.lax.stop_gradient(snthresh)
            peak_width = jax.lax.stop_gradient(peak_width)

        intensity = data["intensity"]
        smoothed = _gaussian_smooth(intensity, peak_width, config.kernel_radius)
        noise = jnp.quantile(intensity, config.noise_quantile)
        above_threshold = jax.nn.sigmoid((smoothed - snthresh * noise) / config.temperature)
        local_maximum = _soft_local_maximum(smoothed, config.temperature)
        peak_weights = above_threshold * local_maximum

        centroided = smoothed * peak_weights
        weight = peak_weights * smoothed
        centroid_mz = jnp.sum(self.mz * weight) / (jnp.sum(weight) + _EPSILON)

        output_data = {
            **data,
            "peak_weights": peak_weights,
            "centroided": centroided,
            "centroid_mz": centroid_mz,
        }
        return output_data, state, metadata
