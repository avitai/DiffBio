"""Tests for self-supervised masked-value losses (B6)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.losses.self_supervised_losses import (
    MaskedValueLoss,
    masked_value_loss,
)


def _data(seed: int = 0) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    predictions = jnp.asarray(rng.normal(size=(4, 20)).astype(np.float32))
    targets = jnp.asarray(rng.normal(size=(4, 20)).astype(np.float32))
    mask = jnp.asarray((rng.uniform(size=(4, 20)) < 0.3).astype(np.float32))
    return predictions, targets, mask


# --- Masking semantics -----------------------------------------------------------


def test_only_masked_positions_contribute() -> None:
    """Errors at unmasked positions do not affect the loss."""
    predictions = jnp.zeros((1, 4))
    targets = jnp.asarray([[0.0, 0.0, 100.0, 100.0]])  # large error only where unmasked
    mask = jnp.asarray([[1.0, 1.0, 0.0, 0.0]])
    assert float(masked_value_loss(predictions, targets, mask)) == pytest.approx(0.0, abs=1e-6)


def test_zero_loss_when_predictions_match_on_masked() -> None:
    _, targets, mask = _data()
    matched = targets  # perfect predictions everywhere
    assert float(masked_value_loss(matched, targets, mask)) == pytest.approx(0.0, abs=1e-6)


def test_matches_manual_masked_mse() -> None:
    predictions, targets, mask = _data(1)
    expected = float(
        np.sum(np.asarray(mask) * (np.asarray(predictions) - np.asarray(targets)) ** 2)
        / np.sum(np.asarray(mask))
    )
    assert float(masked_value_loss(predictions, targets, mask)) == pytest.approx(expected, rel=1e-5)


def test_mae_loss_type_matches_manual() -> None:
    predictions, targets, mask = _data(2)
    expected = float(
        np.sum(np.asarray(mask) * np.abs(np.asarray(predictions) - np.asarray(targets)))
        / np.sum(np.asarray(mask))
    )
    got = float(masked_value_loss(predictions, targets, mask, loss_type="mae"))
    assert got == pytest.approx(expected, rel=1e-5)


# --- Robustness ------------------------------------------------------------------


def test_empty_mask_returns_finite_zero() -> None:
    """An all-zero mask (nothing masked) yields a finite zero loss, not a NaN."""
    predictions, targets, _ = _data(3)
    mask = jnp.zeros_like(predictions)
    value = float(masked_value_loss(predictions, targets, mask))
    assert np.isfinite(value)
    assert value == pytest.approx(0.0, abs=1e-6)


def test_gradient_flows_to_predictions_only_through_masked() -> None:
    predictions, targets, mask = _data(4)
    grad = jax.grad(lambda p: masked_value_loss(p, targets, mask))(predictions)
    assert jnp.all(jnp.isfinite(grad))
    # Gradient is exactly zero at unmasked positions.
    np.testing.assert_allclose(np.asarray(grad)[np.asarray(mask) == 0.0], 0.0, atol=1e-7)
    assert jnp.any(grad != 0.0)


# --- Module wrapper --------------------------------------------------------------


def test_module_matches_function() -> None:
    predictions, targets, mask = _data(5)
    module = MaskedValueLoss(rngs=nnx.Rngs(0))
    assert float(module(predictions, targets, mask)) == pytest.approx(
        float(masked_value_loss(predictions, targets, mask)), rel=1e-6
    )


def test_invalid_loss_type_raises() -> None:
    with pytest.raises(ValueError, match="loss_type"):
        masked_value_loss(jnp.zeros((2, 2)), jnp.zeros((2, 2)), jnp.ones((2, 2)), loss_type="bogus")
