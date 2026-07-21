"""Tests for the SoftCentroider operator (B2: differentiable mass-spec peak-picking)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.operators.metabolomics.soft_centroiding import (
    SoftCentroider,
    SoftCentroiderConfig,
    mz_grid,
)

_MIN_MZ, _MAX_MZ, _N_POINTS = 100.0, 200.0, 1000


def _config(**overrides: float) -> SoftCentroiderConfig:
    params: dict[str, float] = {
        "min_mz": _MIN_MZ,
        "max_mz": _MAX_MZ,
        "n_points": _N_POINTS,
    }
    params.update(overrides)
    return SoftCentroiderConfig(**params)  # type: ignore[arg-type]


def _gaussian_peak(center_mz: float, sigma_mz: float = 1.0, height: float = 100.0) -> jnp.ndarray:
    grid = np.asarray(mz_grid(_MIN_MZ, _MAX_MZ, _N_POINTS))
    return jnp.asarray(
        (height * np.exp(-0.5 * ((grid - center_mz) / sigma_mz) ** 2)).astype(np.float32)
    )


def _operator(config: SoftCentroiderConfig | None = None) -> SoftCentroider:
    return SoftCentroider(config or _config(), rngs=nnx.Rngs(0))


def _apply(operator: SoftCentroider, intensity: jnp.ndarray) -> dict:
    return operator.apply({"intensity": intensity}, {}, None)[0]


# --- Peak localization -----------------------------------------------------------


def test_recovers_single_peak_centroid_mz() -> None:
    """The intensity-weighted centroid recovers a known peak position sub-grid."""
    true_mz = 150.37
    out = _apply(_operator(), _gaussian_peak(true_mz))
    assert float(out["centroid_mz"]) == pytest.approx(true_mz, abs=0.05)


def test_peak_weights_are_bounded() -> None:
    out = _apply(_operator(), _gaussian_peak(150.0))
    weights = np.asarray(out["peak_weights"])
    assert np.all(weights >= 0.0) and np.all(weights <= 1.0)


def test_peak_weight_concentrates_at_the_peak() -> None:
    """The maximum soft peak-weight sits at the true peak, not in the noise floor."""
    center = 150.0
    out = _apply(_operator(), _gaussian_peak(center))
    grid = np.asarray(mz_grid(_MIN_MZ, _MAX_MZ, _N_POINTS))
    peak_mz = grid[int(np.argmax(np.asarray(out["peak_weights"])))]
    assert abs(peak_mz - center) < 0.5


# --- Noise suppression -----------------------------------------------------------


def test_threshold_suppresses_noise_below_a_real_peak() -> None:
    """A real peak carries far more centroided (intensity-weighted) mass than noise alone."""
    rng = np.random.default_rng(0)
    noise_floor = jnp.asarray(rng.uniform(0.0, 2.0, size=_N_POINTS).astype(np.float32))
    peak = _gaussian_peak(150.0, height=100.0) + noise_floor
    operator = _operator()
    noise_mass = float(jnp.sum(_apply(operator, noise_floor)["centroided"]))
    peak_mass = float(jnp.sum(_apply(operator, peak)["centroided"]))
    assert peak_mass > 5.0 * noise_mass


def test_higher_snthresh_reduces_total_peak_weight() -> None:
    """Raising the S/N threshold suppresses a weak peak sitting on a noise floor."""
    rng = np.random.default_rng(1)
    signal = _gaussian_peak(150.0, height=5.0) + jnp.asarray(
        rng.uniform(0.0, 2.0, size=_N_POINTS).astype(np.float32)
    )
    low = float(jnp.sum(_apply(_operator(_config(snthresh_init=1.0)), signal)["peak_weights"]))
    high = float(jnp.sum(_apply(_operator(_config(snthresh_init=8.0)), signal)["peak_weights"]))
    assert high < low


# --- Differentiability -----------------------------------------------------------


def test_gradient_flows_to_learnable_params() -> None:
    signal = _gaussian_peak(150.0)

    def loss_fn(op: SoftCentroider) -> jax.Array:
        return op.apply({"intensity": signal}, {}, None)[0]["centroided"].sum()

    grads = nnx.grad(loss_fn)(_operator())
    for leaf in jax.tree.leaves(grads):
        assert jnp.all(jnp.isfinite(leaf))
    assert any(jnp.any(leaf != 0.0) for leaf in jax.tree.leaves(grads))


# --- Config validation -----------------------------------------------------------


def test_invalid_mz_range_raises() -> None:
    with pytest.raises(ValueError, match="max_mz"):
        _config(max_mz=_MIN_MZ)


def test_non_positive_snthresh_init_raises() -> None:
    with pytest.raises(ValueError, match="snthresh_init"):
        _config(snthresh_init=0.0)
