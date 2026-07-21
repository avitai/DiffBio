"""Differentiable soft isotope-envelope deisotoping for mass spectra.

After centroiding, a compound appears as an isotope envelope: a comb of peaks spaced by the
mass difference between neighbouring isotopes divided by the charge (~1.00235 / z m/z),
with intensities following an averagine-like decay. Classic deisotoping collapses each
envelope to its monoisotopic peak and assigns a charge with hand-set mass tolerances.

This operator makes deisotoping a differentiable matched filter. For each candidate charge it
samples the intensity at the expected isotope offsets (linear interpolation resolves offsets
that fall between grid points) and correlates them with a learnable averagine template. The
per-charge scores give a soft charge assignment at every m/z; the deisotoped signal is the
soft-max over charges, which concentrates at monoisotopic positions. The learnable envelope
decay receives gradients from a downstream identification loss; freezing it
(``trainable=False``) recovers a fixed-template baseline for a frozen-vs-joint study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx

from diffbio.operators.metabolomics.soft_centroiding import mz_grid

_DEFAULT_ISOTOPE_SPACING = 1.00235
_DEFAULT_DECAY = 0.6
_DEFAULT_TEMPERATURE = 1.0


def _inverse_sigmoid(value: float) -> float:
    """Return the ``sigmoid`` pre-image of a value in ``(0, 1)``."""
    return float(jnp.log(jnp.asarray(value) / (1.0 - value)))


@dataclass(frozen=True)
class SoftIsotopeEnvelopeConfig(OperatorConfig):
    """Configuration for :class:`SoftIsotopeEnvelope`.

    Attributes:
        min_mz: Lower bound of the m/z grid.
        max_mz: Upper bound of the m/z grid.
        n_points: Number of profile grid points.
        charges: Candidate charge states to score.
        n_isotopes: Number of isotope peaks in the matched-filter template.
        isotope_spacing: Neighbouring-isotope mass spacing in daltons.
        decay_init: Initial averagine intensity decay in ``(0, 1)``.
        temperature: Sharpness of the soft charge assignment.
        trainable: When ``False`` the learnable decay is stop-gradiented (frozen arm).
    """

    min_mz: float = 0.0
    max_mz: float = 1000.0
    n_points: int = 1000
    charges: tuple[int, ...] = field(default_factory=lambda: (1, 2, 3))
    n_isotopes: int = 4
    isotope_spacing: float = _DEFAULT_ISOTOPE_SPACING
    decay_init: float = _DEFAULT_DECAY
    temperature: float = _DEFAULT_TEMPERATURE
    trainable: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration at construction, failing fast on bad values.

        Raises:
            ValueError: If the m/z range is empty, sizes/charges are non-positive, the
                isotope count is not strictly positive, or ``decay_init`` is outside ``(0, 1)``.
        """
        super().__post_init__()
        if self.max_mz <= self.min_mz:
            raise ValueError(f"max_mz must exceed min_mz, got {self.max_mz} <= {self.min_mz}")
        if self.n_points <= 0:
            raise ValueError(f"n_points must be strictly positive, got {self.n_points}")
        if len(self.charges) == 0:
            raise ValueError("charges must be non-empty")
        if any(charge <= 0 for charge in self.charges):
            raise ValueError(f"charges must be strictly positive, got {self.charges}")
        if self.n_isotopes <= 0:
            raise ValueError(f"n_isotopes must be strictly positive, got {self.n_isotopes}")
        if not 0.0 < self.decay_init < 1.0:
            raise ValueError(f"decay_init must be in (0, 1), got {self.decay_init}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive, got {self.temperature}")


class SoftIsotopeEnvelope(OperatorModule):
    """Differentiable deisotoping matched filter with a learnable averagine decay.

    This is a per-spectrum operator: ``apply`` deisotopes one spectrum's ``(n_points,)``
    profile intensities, and the framework's ``apply_batch`` vmaps it over spectra. The
    learnable envelope ``decay`` is shared across spectra and receives gradients from a
    downstream loss unless ``trainable`` is ``False``.
    """

    def __init__(
        self,
        config: SoftIsotopeEnvelopeConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the operator with a learnable, sigmoid-bounded envelope decay.

        Args:
            config: Isotope-envelope configuration.
            rngs: Optional RNG state (unused; kept for interface compatibility).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        self.raw_decay = nnx.Param(
            jnp.asarray(_inverse_sigmoid(config.decay_init), dtype=jnp.float32)
        )
        self.mz = mz_grid(config.min_mz, config.max_mz, config.n_points)
        self.charges = jnp.asarray(config.charges, dtype=jnp.float32)
        self.isotope_indices = jnp.arange(config.n_isotopes, dtype=jnp.float32)

    def _charge_scores(self, intensity: jnp.ndarray, decay: jnp.ndarray) -> jnp.ndarray:
        """Return the matched-filter score ``(n_charges, n_points)`` for each charge."""
        config: SoftIsotopeEnvelopeConfig = self.config
        template = decay**self.isotope_indices
        template = template / jnp.sum(template)

        def score_for_charge(charge: jnp.ndarray) -> jnp.ndarray:
            offsets = self.isotope_indices * (config.isotope_spacing / charge)
            positions = self.mz[:, None] + offsets[None, :]
            sampled = jnp.interp(positions.reshape(-1), self.mz, intensity).reshape(positions.shape)
            return sampled @ template

        return jax.vmap(score_for_charge)(self.charges)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Deisotope ``data["intensity"]`` and add charge scores, charge, and the signal.

        Args:
            data: Dictionary containing ``"intensity"`` ``(n_points,)`` for one spectrum.
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where ``output_data`` adds
            ``"charge_scores"`` ``(n_charges, n_points)``, ``"charge"`` (expected charge per
            position), and ``"deisotoped"`` (the soft-max-over-charges monoisotopic signal).
        """
        del random_params, stats
        config: SoftIsotopeEnvelopeConfig = self.config
        decay = jax.nn.sigmoid(self.raw_decay[...])
        if not config.trainable:
            decay = jax.lax.stop_gradient(decay)

        scores = self._charge_scores(data["intensity"], decay)
        charge_weights = jax.nn.softmax(scores / config.temperature, axis=0)
        deisotoped = jnp.sum(charge_weights * scores, axis=0)
        charge = jnp.sum(charge_weights * self.charges[:, None], axis=0)

        output_data = {
            **data,
            "charge_scores": scores,
            "charge": charge,
            "deisotoped": deisotoped,
        }
        return output_data, state, metadata
