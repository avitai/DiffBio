"""Tests for the SoftHVG operator (ticket 04)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st

from diffbio.operators.singlecell.soft_hvg import (
    HVGFlavor,
    SoftHVG,
    SoftHVGConfig,
    gene_dispersion,
    highly_variable_genes,
    soft_hvg_mask,
)


def _expression(n_cells: int, n_genes: int, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.gamma(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32))


def _baseline_hvg(features: jnp.ndarray, n_top_genes: int) -> set[int]:
    # Mirrors benchmarks/singlecell/frozen_annotation_baseline._select_highly_variable_genes.
    array = np.asarray(features)
    gene_mean = array.mean(axis=0)
    gene_variance = array.var(axis=0)
    safe_mean = np.where(gene_mean == 0.0, 1.0, gene_mean)
    dispersion = gene_variance / safe_mean
    ranked = np.argsort(dispersion)[::-1][:n_top_genes]
    return set(int(i) for i in ranked)


def _config(n_genes: int, n_top_genes: int, **kwargs: Any) -> SoftHVGConfig:
    return SoftHVGConfig(n_genes=n_genes, n_top_genes=n_top_genes, **kwargs)


def _operator(n_genes: int, n_top_genes: int, seed: int = 0, **kwargs: Any) -> SoftHVG:
    return SoftHVG(_config(n_genes, n_top_genes, **kwargs), rngs=nnx.Rngs(seed))


# --- Dispersion parity with the frozen baseline ---------------------------------


def test_gene_dispersion_matches_baseline_formula() -> None:
    features = _expression(200, 30, seed=0)
    dispersion = gene_dispersion(features)
    array = np.asarray(features)
    safe_mean = np.where(array.mean(0) == 0.0, 1.0, array.mean(0))
    expected = array.var(0) / safe_mean
    np.testing.assert_allclose(np.asarray(dispersion), expected, rtol=1e-4)


def test_hard_mask_selects_baseline_top_k() -> None:
    features = _expression(300, 40, seed=1)
    n_top = 12
    mask, _ = soft_hvg_mask(features, n_top, gene_weights=jnp.zeros(40), softness=0.1, hard=True)
    selected = {int(i) for i in np.flatnonzero(np.asarray(mask) > 0.5)}
    assert selected == _baseline_hvg(features, n_top)


def test_operator_frozen_defaults_select_baseline_top_k() -> None:
    features = _expression(300, 40, seed=2)
    operator = _operator(40, 15)
    output, _, _ = operator.apply({"features": features}, {}, None)
    selected = {int(i) for i in np.flatnonzero(np.asarray(output["hvg_weights"]) > 0.5)}
    assert selected == _baseline_hvg(features, 15)


def test_hard_mask_is_exactly_binary_and_sums_to_k() -> None:
    features = _expression(150, 25, seed=3)
    mask, _ = soft_hvg_mask(features, 8, gene_weights=jnp.zeros(25), softness=0.1, hard=True)
    unique = set(float(v) for v in np.unique(np.round(np.asarray(mask), 5)))
    assert unique <= {0.0, 1.0}
    np.testing.assert_allclose(float(jnp.sum(mask)), 8.0, atol=1e-4)


# --- Soft mode ------------------------------------------------------------------


def test_soft_mask_is_bounded_at_any_softness() -> None:
    features = _expression(150, 25, seed=4)
    for softness in (0.3, 0.05):
        mask, _ = soft_hvg_mask(
            features, 8, gene_weights=jnp.zeros(25), softness=softness, hard=False
        )
        # Capped-simplex projection clips exactly onto [0, 1].
        assert float(jnp.min(mask)) >= 0.0
        assert float(jnp.max(mask)) <= 1.0


def test_soft_mask_sums_to_exactly_k() -> None:
    features = _expression(150, 25, seed=4)
    for softness in (0.3, 0.05):
        mask, _ = soft_hvg_mask(
            features, 8, gene_weights=jnp.zeros(25), softness=softness, hard=False
        )
        np.testing.assert_allclose(float(jnp.sum(mask)), 8.0, atol=1e-4)


def test_soft_mask_is_smoother_than_hard() -> None:
    features = _expression(150, 25, seed=5)
    soft, _ = soft_hvg_mask(features, 8, gene_weights=jnp.zeros(25), softness=0.5, hard=False)
    # A genuinely soft mask has fractional entries, unlike the 0/1 hard mask.
    fractional = np.sum((np.asarray(soft) > 0.01) & (np.asarray(soft) < 0.99))
    assert fractional > 0


# --- Gradient flow to gene weights ----------------------------------------------


def test_gradient_flows_to_gene_weights_hard_mode() -> None:
    features = _expression(120, 20, seed=6)

    def loss(weights: jnp.ndarray) -> jnp.ndarray:
        mask, _ = soft_hvg_mask(features, 6, gene_weights=weights, softness=0.5, hard=True)
        return jnp.sum(mask * jnp.arange(20, dtype=jnp.float32))

    grad = jax.grad(loss)(jnp.zeros(20))
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0


def test_operator_parameter_gradients_flow_and_are_finite() -> None:
    features = _expression(120, 20, seed=7)
    operator = _operator(20, 6, hard=False)
    graphdef, params, rest = nnx.split(operator, nnx.Param, ...)

    def loss(param_state: nnx.State) -> jnp.ndarray:
        model = nnx.merge(graphdef, param_state, rest)
        output, _, _ = model.apply({"features": features}, {}, None)
        return jnp.sum(output["hvg_weights"] * jnp.arange(20, dtype=jnp.float32))

    grads = jax.grad(loss)(params)
    leaves = jax.tree.leaves(grads)
    assert len(leaves) == 1
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert all(float(jnp.linalg.norm(leaf)) > 0.0 for leaf in leaves)


def test_learnable_weight_can_promote_a_gene_into_top_k() -> None:
    features = _expression(200, 30, seed=8)
    baseline = _baseline_hvg(features, 5)
    excluded = next(g for g in range(30) if g not in baseline)
    # A large positive weight on an excluded gene should pull it into the selection.
    weights = jnp.zeros(30).at[excluded].set(10.0)
    mask, _ = soft_hvg_mask(features, 5, gene_weights=weights, softness=0.1, hard=True)
    selected = {int(i) for i in np.flatnonzero(np.asarray(mask) > 0.5)}
    assert excluded in selected


# --- Edge / corner cases --------------------------------------------------------


def test_all_genes_selected_when_k_exceeds_n_genes() -> None:
    features = _expression(80, 10, seed=9)
    mask, _ = soft_hvg_mask(features, 25, gene_weights=jnp.zeros(10), softness=0.1, hard=True)
    np.testing.assert_allclose(np.asarray(mask), np.ones(10), atol=1e-4)


def test_single_top_gene() -> None:
    features = _expression(100, 12, seed=10)
    mask, _ = soft_hvg_mask(features, 1, gene_weights=jnp.zeros(12), softness=0.1, hard=True)
    np.testing.assert_allclose(float(jnp.sum(mask)), 1.0, atol=1e-4)
    top = int(np.argmax(np.asarray(mask)))
    assert top in _baseline_hvg(features, 1)


def test_zero_mean_gene_uses_unit_denominator() -> None:
    array = np.array(_expression(50, 8, seed=11))  # writable copy
    array[:, 3] = 0.0  # a dropped gene: mean 0, variance 0.
    dispersion = gene_dispersion(jnp.asarray(array))
    assert bool(jnp.all(jnp.isfinite(dispersion)))
    np.testing.assert_allclose(float(dispersion[3]), 0.0, atol=1e-6)


def test_higher_variance_gene_ranks_higher() -> None:
    rng = np.random.default_rng(12)
    base = rng.normal(5.0, 0.1, size=(200, 6)).astype(np.float32)
    base[:, 2] = rng.normal(5.0, 3.0, size=200).astype(np.float32)  # inflate one gene's variance.
    dispersion = gene_dispersion(jnp.asarray(base))
    assert int(jnp.argmax(dispersion)) == 2


# --- Config validation and API parity -------------------------------------------


def test_config_rejects_non_positive_n_top_genes() -> None:
    for bad in (0, -3):
        with pytest.raises(ValueError, match="n_top_genes"):
            _config(30, bad)


def test_config_rejects_non_positive_n_genes() -> None:
    with pytest.raises(ValueError, match="n_genes"):
        _config(0, 5)


def test_config_rejects_unsupported_flavor() -> None:
    with pytest.raises(NotImplementedError, match="flavor"):
        _config(30, 5, flavor=HVGFlavor.SEURAT_V3)


def test_config_accepts_seurat_flavor() -> None:
    config = _config(30, 5, flavor=HVGFlavor.SEURAT)
    assert config.flavor is HVGFlavor.SEURAT


def test_highly_variable_genes_returns_soft_weights_and_hard_mask() -> None:
    features = _expression(200, 30, seed=13)
    soft_weights, hard_mask = highly_variable_genes(features, n_top_genes=10, softness=0.3)
    assert soft_weights.shape == (30,)
    assert hard_mask.shape == (30,)
    assert hard_mask.dtype == jnp.bool_
    selected = {int(i) for i in np.flatnonzero(np.asarray(hard_mask))}
    assert selected == _baseline_hvg(features, 10)
    # The soft weights are fractional, the hard mask is boolean parity.
    assert float(jnp.max(soft_weights)) <= 1.0 + 1e-4


# --- JAX / Flax NNX transform compatibility -------------------------------------


def test_mask_is_jit_compatible() -> None:
    features = _expression(100, 20, seed=14)
    reference, _ = soft_hvg_mask(features, 6, gene_weights=jnp.zeros(20), softness=0.1, hard=True)
    jitted = jax.jit(soft_hvg_mask, static_argnums=(1,), static_argnames=("softness", "hard"))
    result, _ = jitted(features, 6, gene_weights=jnp.zeros(20), softness=0.1, hard=True)
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference), atol=1e-4)


def test_operator_apply_is_vmap_batchable() -> None:
    batch = jnp.stack([_expression(60, 15, seed=s) for s in (15, 16, 17)])
    operator = _operator(15, 5)

    def one(matrix: jnp.ndarray) -> jnp.ndarray:
        output, _, _ = operator.apply({"features": matrix}, {}, None)
        return output["hvg_weights"]

    out = jax.vmap(one)(batch)
    assert out.shape == (3, 15)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_gradient_finite_under_jit() -> None:
    features = _expression(100, 20, seed=18)

    def loss(weights: jnp.ndarray) -> jnp.ndarray:
        mask, _ = soft_hvg_mask(features, 6, gene_weights=weights, softness=0.5, hard=True)
        return jnp.sum(mask * jnp.arange(20, dtype=jnp.float32))

    grad = jax.jit(jax.grad(loss))(jnp.zeros(20))
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_operator_is_nnx_jit_compatible() -> None:
    features = _expression(80, 18, seed=19)
    operator = _operator(18, 5)

    @nnx.jit
    def run(module: SoftHVG, matrix: jnp.ndarray) -> jnp.ndarray:
        output, _, _ = module.apply({"features": matrix}, {}, None)
        return output["hvg_weights"]

    result = run(operator, features)
    assert result.shape == (18,)
    assert bool(jnp.all(jnp.isfinite(result)))


def test_output_is_deterministic() -> None:
    features = _expression(100, 20, seed=20)
    operator = _operator(20, 6)
    first, _, _ = operator.apply({"features": features}, {}, None)
    second, _, _ = operator.apply({"features": features}, {}, None)
    np.testing.assert_array_equal(
        np.asarray(first["hvg_weights"]), np.asarray(second["hvg_weights"])
    )


@settings(max_examples=25, deadline=None)
@given(
    n_cells=st.integers(min_value=5, max_value=60),
    n_genes=st.integers(min_value=2, max_value=40),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_property_soft_mask_finite_and_bounded(n_cells: int, n_genes: int, seed: int) -> None:
    features = _expression(n_cells, n_genes, seed)
    k = max(1, n_genes // 2)
    mask, _ = soft_hvg_mask(features, k, gene_weights=jnp.zeros(n_genes), softness=0.3, hard=False)
    assert bool(jnp.all(jnp.isfinite(mask)))
    assert float(jnp.min(mask)) >= -1e-4
    assert float(jnp.max(mask)) <= 1.0 + 1e-4


# --- Operator contract ----------------------------------------------------------


def test_operator_apply_gates_features_and_adds_keys() -> None:
    features = _expression(100, 20, seed=21)
    operator = _operator(20, 6)
    output, _, _ = operator.apply({"features": features}, {}, None)
    assert output["hvg_weights"].shape == (20,)
    assert output["hvg_dispersion"].shape == (20,)
    assert output["features"].shape == (100, 20)
    # Non-selected genes are gated to zero in the emitted features.
    mask = np.asarray(output["hvg_weights"]) > 0.5
    gated = np.asarray(output["features"])
    assert np.allclose(gated[:, ~mask], 0.0, atol=1e-4)


def test_operator_exposes_one_learnable_parameter() -> None:
    operator = _operator(20, 6)
    params = nnx.state(operator, nnx.Param)
    assert len(jax.tree.leaves(params)) == 1
