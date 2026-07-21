"""Tests for the StochasticGateSelector operator (B3: STG hard-concrete L0 gene gate)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.operators.singlecell.stochastic_gate_selector import (
    StochasticGateSelector,
    StochasticGateSelectorConfig,
    l0_penalty,
)

_SIGMA = 0.5


def _features(n_cells: int, n_genes: int, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.gamma(2.0, 5.0, size=(n_cells, n_genes)).astype(np.float32))


def _operator(
    n_genes: int,
    *,
    stochastic: bool = False,
    mu_init: float = 0.5,
    l0_lambda: float = 1.0,
    init_gate: jnp.ndarray | None = None,
    seed: int = 0,
) -> StochasticGateSelector:
    stream = "gate_noise" if stochastic else None
    config = StochasticGateSelectorConfig(
        n_genes=n_genes,
        sigma=_SIGMA,
        l0_lambda=l0_lambda,
        mu_init=mu_init,
        stochastic=stochastic,
        stream_name=stream,
    )
    rngs = nnx.Rngs(seed, gate_noise=seed) if stochastic else nnx.Rngs(seed)
    return StochasticGateSelector(config, init_gate=init_gate, rngs=rngs)


def _apply(operator: StochasticGateSelector, features: jnp.ndarray) -> dict:
    return operator.apply({"features": features}, {}, None)[0]


# --- Deterministic (eval) gate ---------------------------------------------------


def test_eval_gate_is_clipped_mu() -> None:
    """With stochastic=False the gate is the deterministic clamp of mu (0.5 here)."""
    operator = _operator(10, stochastic=False, mu_init=0.5)
    out = _apply(operator, _features(4, 10, 0))
    np.testing.assert_allclose(np.asarray(out["gate"]), 0.5, atol=1e-6)


def test_gate_is_bounded_in_unit_interval() -> None:
    operator = _operator(10, stochastic=False, mu_init=3.0)  # clamps to 1
    out = _apply(operator, _features(4, 10, 1))
    gate = np.asarray(out["gate"])
    assert np.all(gate >= 0.0) and np.all(gate <= 1.0)
    np.testing.assert_allclose(gate, 1.0, atol=1e-6)


def test_features_are_gated() -> None:
    operator = _operator(8, stochastic=False, mu_init=0.5)
    features = _features(5, 8, 2)
    out = _apply(operator, features)
    np.testing.assert_allclose(np.asarray(out["features"]), np.asarray(features) * 0.5, atol=1e-5)


# --- Init-at-frozen from a top-k mask --------------------------------------------


def test_init_gate_reproduces_frozen_mask() -> None:
    """Init from a hard 0/1 mask makes the deterministic gate reproduce it exactly."""
    mask = jnp.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    operator = _operator(6, stochastic=False, init_gate=mask)
    out = _apply(operator, _features(3, 6, 3))
    np.testing.assert_array_equal(np.asarray(out["gate"]), np.asarray(mask))


# --- L0 penalty ------------------------------------------------------------------


def test_l0_penalty_matches_normal_cdf_sum() -> None:
    mu = jnp.asarray([-1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
    expected = float(jnp.sum(jax.scipy.special.ndtr(mu / _SIGMA)))
    assert float(l0_penalty(mu, _SIGMA)) == pytest.approx(expected, abs=1e-5)


def test_l0_penalty_exposed_and_scaled_by_lambda() -> None:
    operator = _operator(6, stochastic=False, mu_init=0.5, l0_lambda=2.0)
    out = _apply(operator, _features(3, 6, 4))
    expected = 2.0 * float(l0_penalty(jnp.full(6, 0.5), _SIGMA))
    assert float(out["l0_penalty"]) == pytest.approx(expected, abs=1e-4)


# --- Differentiability -----------------------------------------------------------


def test_gradient_flows_to_mu_from_features() -> None:
    operator = _operator(12, stochastic=False, mu_init=0.5)
    features = _features(6, 12, 5)

    def loss_fn(op: StochasticGateSelector) -> jax.Array:
        return op.apply({"features": features}, {}, None)[0]["features"].sum()

    grad = nnx.grad(loss_fn)(operator).mu.value
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0.0)


def test_l0_penalty_gradient_pushes_mu_down() -> None:
    """The L0 penalty gradient wrt mu is positive, so minimizing it shrinks the gates."""
    operator = _operator(12, stochastic=False, mu_init=0.5)

    def penalty_fn(op: StochasticGateSelector) -> jax.Array:
        return op.apply({"features": _features(4, 12, 6)}, {}, None)[0]["l0_penalty"]

    grad = nnx.grad(penalty_fn)(operator).mu.value
    assert jnp.all(grad > 0.0)


# --- Stochastic (train) gate -----------------------------------------------------


def test_stochastic_gate_varies_and_stays_bounded() -> None:
    operator = _operator(200, stochastic=True, mu_init=0.5, seed=7)
    features = _features(2, 200, 8)
    gate_a = np.asarray(_apply(operator, features)["gate"])
    gate_b = np.asarray(_apply(operator, features)["gate"])
    assert np.all(gate_a >= 0.0) and np.all(gate_a <= 1.0)
    assert not np.allclose(gate_a, gate_b)  # fresh noise each call


# --- Config validation -----------------------------------------------------------


def test_non_positive_n_genes_raises() -> None:
    with pytest.raises(ValueError, match="n_genes"):
        StochasticGateSelectorConfig(n_genes=0)


def test_non_positive_sigma_raises() -> None:
    with pytest.raises(ValueError, match="sigma"):
        StochasticGateSelectorConfig(n_genes=4, sigma=0.0)


def test_negative_l0_lambda_raises() -> None:
    with pytest.raises(ValueError, match="l0_lambda"):
        StochasticGateSelectorConfig(n_genes=4, l0_lambda=-1.0)
