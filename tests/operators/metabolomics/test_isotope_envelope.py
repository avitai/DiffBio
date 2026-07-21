"""Tests for the SoftIsotopeEnvelope operator (B2: differentiable deisotoping)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.operators.metabolomics.isotope_envelope import (
    SoftIsotopeEnvelope,
    SoftIsotopeEnvelopeConfig,
)
from diffbio.operators.metabolomics.soft_centroiding import mz_grid

_MIN_MZ, _MAX_MZ, _N_POINTS = 100.0, 200.0, 2000
_SPACING = 1.00235


def _config(**overrides: object) -> SoftIsotopeEnvelopeConfig:
    params: dict[str, object] = {
        "min_mz": _MIN_MZ,
        "max_mz": _MAX_MZ,
        "n_points": _N_POINTS,
        "charges": (1, 2, 3),
        "n_isotopes": 4,
    }
    params.update(overrides)
    return SoftIsotopeEnvelopeConfig(**params)  # type: ignore[arg-type]


def _envelope(mono_mz: float, charge: int, decay: float = 0.6) -> jnp.ndarray:
    """A synthetic isotope envelope: peaks at mono + i*spacing/charge, decaying intensity."""
    grid = np.asarray(mz_grid(_MIN_MZ, _MAX_MZ, _N_POINTS))
    spectrum = np.zeros(_N_POINTS, dtype=np.float32)
    for i in range(4):
        center = mono_mz + i * _SPACING / charge
        spectrum += (decay**i) * 100.0 * np.exp(-0.5 * ((grid - center) / 0.05) ** 2)
    return jnp.asarray(spectrum)


def _operator(config: SoftIsotopeEnvelopeConfig | None = None) -> SoftIsotopeEnvelope:
    return SoftIsotopeEnvelope(config or _config(), rngs=nnx.Rngs(0))


def _apply(operator: SoftIsotopeEnvelope, intensity: jnp.ndarray) -> dict:
    return operator.apply({"intensity": intensity}, {}, None)[0]


def _grid() -> np.ndarray:
    return np.asarray(mz_grid(_MIN_MZ, _MAX_MZ, _N_POINTS))


# --- Charge inference ------------------------------------------------------------


def test_infers_correct_charge_at_monoisotopic() -> None:
    """The expected charge at the monoisotopic position matches the true charge."""
    mono, charge = 150.0, 2
    out = _apply(_operator(), _envelope(mono, charge))
    mono_index = int(np.argmin(np.abs(_grid() - mono)))
    inferred = float(np.asarray(out["charge"])[mono_index])
    assert abs(inferred - charge) < 0.5


def test_matched_charge_scores_higher_than_wrong_charge() -> None:
    """At the monoisotopic peak the true charge's matched-filter score wins."""
    mono, charge = 150.0, 2
    out = _apply(_operator(), _envelope(mono, charge))
    scores = np.asarray(out["charge_scores"])  # (n_charges, n_points)
    mono_index = int(np.argmin(np.abs(_grid() - mono)))
    charge_index = (1, 2, 3).index(charge)
    assert int(np.argmax(scores[:, mono_index])) == charge_index


# --- Deisotoping -----------------------------------------------------------------


def test_deisotoped_peak_at_monoisotopic() -> None:
    """The deisotoped signal concentrates at the monoisotopic m/z, not a heavier isotope."""
    mono, charge = 150.0, 2
    out = _apply(_operator(), _envelope(mono, charge))
    grid = _grid()
    peak_mz = grid[int(np.argmax(np.asarray(out["deisotoped"])))]
    assert abs(peak_mz - mono) < 0.1


# --- Differentiability -----------------------------------------------------------


def test_gradient_flows_to_decay() -> None:
    signal = _envelope(150.0, 2)

    def loss_fn(op: SoftIsotopeEnvelope) -> jax.Array:
        return op.apply({"intensity": signal}, {}, None)[0]["deisotoped"].sum()

    grad = nnx.grad(loss_fn)(_operator()).raw_decay.value
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0.0)


def test_frozen_arm_has_zero_decay_gradient() -> None:
    signal = _envelope(150.0, 2)

    def loss_fn(op: SoftIsotopeEnvelope) -> jax.Array:
        return op.apply({"intensity": signal}, {}, None)[0]["deisotoped"].sum()

    grad = nnx.grad(loss_fn)(_operator(_config(trainable=False))).raw_decay.value
    np.testing.assert_allclose(np.asarray(grad), 0.0, atol=1e-12)


# --- Config validation -----------------------------------------------------------


def test_empty_charges_raises() -> None:
    with pytest.raises(ValueError, match="charges"):
        _config(charges=())


def test_non_positive_n_isotopes_raises() -> None:
    with pytest.raises(ValueError, match="n_isotopes"):
        _config(n_isotopes=0)
