"""Tests for the SoftComponentSelection (differentiable soft-k) operator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.operators.normalization.soft_pca import (
    SoftComponentSelection,
    SoftComponentSelectionConfig,
)


def _decreasing_eigenvalues(k: int) -> np.ndarray:
    return np.linspace(10.0, 1.0, k).astype(np.float32)


def _operator(
    k: int, *, coverage: float = 0.9, temperature: float = 0.05
) -> SoftComponentSelection:
    return SoftComponentSelection(
        SoftComponentSelectionConfig(
            n_components=k, init_coverage=coverage, temperature=temperature
        ),
        eigenvalues=_decreasing_eigenvalues(k),
        rngs=nnx.Rngs(0),
    )


def _gate(operator: SoftComponentSelection) -> np.ndarray:
    projection = jnp.ones((1, operator.cumulative_variance.shape[0]))
    gated, _, _ = operator.apply({"projection": projection}, {}, None)
    return np.asarray(gated["projection"])[0]


# --- Config validation ----------------------------------------------------------


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="n_components"):
        SoftComponentSelectionConfig(n_components=0)
    with pytest.raises(ValueError, match="init_coverage"):
        SoftComponentSelectionConfig(n_components=10, init_coverage=1.5)
    with pytest.raises(ValueError, match="temperature"):
        SoftComponentSelectionConfig(n_components=10, temperature=0.0)


def test_init_rejects_mismatched_eigenvalues_length() -> None:
    with pytest.raises(ValueError, match="eigenvalues"):
        SoftComponentSelection(
            SoftComponentSelectionConfig(n_components=10),
            eigenvalues=np.ones(8, dtype=np.float32),
            rngs=nnx.Rngs(0),
        )


# --- Monotone soft gate over the eigen-spectrum ---------------------------------


def test_gate_is_monotone_decreasing_over_components() -> None:
    gate = _gate(_operator(12))
    # Cumulative variance increases with component index, so the keep-gate decreases.
    assert np.all(np.diff(gate) <= 1e-6)
    assert gate[0] > gate[-1]


def test_higher_coverage_keeps_more_components() -> None:
    low = float(np.sum(_gate(_operator(12, coverage=0.5))))
    high = float(np.sum(_gate(_operator(12, coverage=0.99))))
    # A larger coverage target keeps a larger effective dimension.
    assert high > low


def test_output_shape_is_preserved() -> None:
    operator = _operator(10)
    projection = jnp.asarray(np.random.default_rng(0).normal(size=(7, 10)).astype(np.float32))
    gated, _, _ = operator.apply({"projection": projection}, {}, None)
    assert gated["projection"].shape == (7, 10)


# --- Hard limit recovers a hard truncation --------------------------------------


def test_small_temperature_recovers_hard_truncation() -> None:
    # With a tiny temperature the soft gate approaches a step: components whose
    # cumulative variance is below the coverage target are kept (~1), the rest ~0.
    operator = _operator(12, coverage=0.6, temperature=1e-4)
    gate = _gate(operator)
    cumulative = np.asarray(operator.cumulative_variance)
    coverage = float(jax.nn.sigmoid(operator.raw_coverage[...]))
    expected = (cumulative < coverage).astype(np.float32)
    np.testing.assert_allclose(gate, expected, atol=1e-3)


# --- Learnability ---------------------------------------------------------------


def test_gradient_flows_to_coverage_only() -> None:
    operator = _operator(10)
    graphdef, params, rest = nnx.split(operator, nnx.Param, ...)

    def loss(param_state: nnx.State) -> jnp.ndarray:
        model = nnx.merge(graphdef, param_state, rest)
        gated, _, _ = model.apply({"projection": jnp.ones((3, 10))}, {}, None)
        return jnp.sum(gated["projection"])

    grads = jax.grad(loss)(params)
    leaves = jax.tree.leaves(grads)
    # Only the coverage scalar is trainable; the eigen-spectrum is a fixed Variable.
    assert len(leaves) == 1
    assert float(jnp.abs(leaves[0])) > 0.0


def test_effective_dimension_reports_soft_component_count() -> None:
    operator = _operator(12, coverage=0.99)
    effective = float(operator.effective_dimension())
    assert 0.0 < effective <= 12.0


# --- Transform compatibility ----------------------------------------------------


def test_operator_is_nnx_jit_compatible() -> None:
    operator = _operator(8)
    projection = jnp.ones((5, 8))

    @nnx.jit
    def run(module: SoftComponentSelection, matrix: jnp.ndarray) -> jnp.ndarray:
        return module.apply({"projection": matrix}, {}, None)[0]["projection"]

    assert run(operator, projection).shape == (5, 8)
