"""Tests for the LearnableNormalization operator (ticket 03)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st

from diffbio.operators.normalization.learnable_normalization import (
    LearnableNormalization,
    LearnableNormalizationConfig,
    normalize_counts,
)

_TARGET_SUM = 1.0e4


def _counts(n_genes: int, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.poisson(3.0, size=(n_genes,)).astype(np.float32))


def _operator(seed: int = 0) -> LearnableNormalization:
    return LearnableNormalization(LearnableNormalizationConfig(), rngs=nnx.Rngs(seed))


def _scale(counts: jnp.ndarray, depth_exponent: float) -> jnp.ndarray:
    library = jnp.maximum(jnp.sum(counts), 1.0)
    return (_TARGET_SUM / library) ** depth_exponent


# --- Frozen parity with scanpy normalize_total + log1p --------------------------


def test_frozen_matches_normalize_total_log1p_formula() -> None:
    counts = _counts(50, seed=0)
    output = normalize_counts(counts, pseudocount=1.0, depth_exponent=1.0, target_sum=_TARGET_SUM)
    expected = jnp.log1p(counts / jnp.sum(counts) * _TARGET_SUM)
    np.testing.assert_allclose(np.asarray(output), np.asarray(expected), atol=1e-5)


def test_operator_frozen_defaults_match_scanpy() -> None:
    ad = pytest.importorskip("anndata")  # optional dep; skip when unavailable (e.g. CI)
    sc = pytest.importorskip("scanpy")

    rng = np.random.default_rng(3)
    counts = rng.poisson(3.0, size=(20, 40)).astype(np.float32)
    adata = ad.AnnData(X=counts.copy())
    sc.pp.normalize_total(adata, target_sum=_TARGET_SUM)
    sc.pp.log1p(adata)

    operator = _operator()
    ours = jax.vmap(lambda cell: operator.apply({"counts": cell}, {}, None)[0]["normalized"])(
        jnp.asarray(counts)
    )
    np.testing.assert_allclose(np.asarray(ours), np.asarray(adata.X), atol=1e-4)


def test_operator_frozen_pseudocount_is_exactly_one() -> None:
    operator = _operator()
    assert float(jax.nn.softplus(operator.raw_pseudocount[...])) == 1.0
    assert float(operator.depth_exponent[...]) == 1.0


# --- Gradient correctness against closed form and finite differences -------------


def test_pseudocount_gradient_matches_closed_form() -> None:
    counts = _counts(40, seed=1)
    pseudocount, depth = 1.7, 1.0

    def total(pc: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(normalize_counts(counts, pseudocount=pc, depth_exponent=depth))

    analytic = jax.grad(total)(jnp.asarray(pseudocount))
    # d/dpc sum(log(x*scale + pc)) = sum(1 / (x*scale + pc)).
    closed_form = jnp.sum(1.0 / (counts * _scale(counts, depth) + pseudocount))
    np.testing.assert_allclose(float(analytic), float(closed_form), rtol=1e-4)


def test_depth_gradient_matches_closed_form() -> None:
    counts = _counts(40, seed=2)
    pseudocount, depth = 1.0, 0.7

    def total(exponent: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(normalize_counts(counts, pseudocount=pseudocount, depth_exponent=exponent))

    analytic = jax.grad(total)(jnp.asarray(depth))
    library = jnp.maximum(jnp.sum(counts), 1.0)
    scale = (_TARGET_SUM / library) ** depth
    # d/ddepth log(x*scale + pc) = x*scale*ln(target/library) / (x*scale + pc).
    closed_form = jnp.sum(
        counts * scale * jnp.log(_TARGET_SUM / library) / (counts * scale + pseudocount)
    )
    np.testing.assert_allclose(float(analytic), float(closed_form), rtol=1e-4)


def test_operator_parameter_gradients_flow_and_are_finite() -> None:
    counts = _counts(40, seed=4)
    operator = _operator()
    graphdef, params, rest = nnx.split(operator, nnx.Param, ...)

    def loss(param_state: nnx.State) -> jnp.ndarray:
        model = nnx.merge(graphdef, param_state, rest)
        output, _, _ = model.apply({"counts": counts}, {}, None)
        return jnp.sum(output["normalized"] ** 2)

    grads = jax.grad(loss)(params)
    leaves = jax.tree.leaves(grads)
    assert len(leaves) == 2
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert all(float(jnp.linalg.norm(leaf)) > 0.0 for leaf in leaves)


# --- Learnable behavior ---------------------------------------------------------


def test_learnable_pseudocount_changes_output() -> None:
    counts = _counts(30, seed=5)
    low = normalize_counts(counts, pseudocount=0.5, depth_exponent=1.0)
    high = normalize_counts(counts, pseudocount=5.0, depth_exponent=1.0)
    assert float(jnp.max(jnp.abs(low - high))) > 1e-3


def test_depth_exponent_zero_disables_depth_normalization() -> None:
    counts = _counts(30, seed=6)
    output = normalize_counts(counts, pseudocount=1.0, depth_exponent=0.0)
    np.testing.assert_allclose(np.asarray(output), np.asarray(jnp.log1p(counts)), atol=1e-5)


def test_depth_exponent_changes_output() -> None:
    counts = _counts(30, seed=7)
    full = normalize_counts(counts, pseudocount=1.0, depth_exponent=1.0)
    half = normalize_counts(counts, pseudocount=1.0, depth_exponent=0.5)
    assert float(jnp.max(jnp.abs(full - half))) > 1e-3


def test_pseudocount_stays_positive_under_extreme_raw_value() -> None:
    counts = _counts(30, seed=8)
    output = normalize_counts(counts, pseudocount=jax.nn.softplus(-50.0), depth_exponent=1.0)
    assert bool(jnp.all(jnp.isfinite(output)))


# --- Edge / corner cases --------------------------------------------------------


def test_zero_count_cell_is_finite() -> None:
    counts = jnp.zeros((20,), dtype=jnp.float32)
    output = normalize_counts(counts, pseudocount=1.0, depth_exponent=1.0)
    assert bool(jnp.all(jnp.isfinite(output)))
    np.testing.assert_allclose(np.asarray(output), np.zeros(20), atol=1e-6)


def test_single_gene() -> None:
    counts = jnp.asarray([7.0], dtype=jnp.float32)
    output = normalize_counts(counts, pseudocount=1.0, depth_exponent=1.0)
    assert output.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(output)))


def test_large_counts_are_finite() -> None:
    counts = jnp.full((30,), 1.0e6, dtype=jnp.float32)
    output = normalize_counts(counts, pseudocount=1.0, depth_exponent=1.0)
    assert bool(jnp.all(jnp.isfinite(output)))


def test_gradient_finite_when_library_equals_target() -> None:
    # ln(target/library) == 0, so the depth gradient is exactly zero, not NaN.
    counts = jnp.full((10,), _TARGET_SUM / 10.0, dtype=jnp.float32)

    def total(exponent: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(normalize_counts(counts, pseudocount=1.0, depth_exponent=exponent))

    grad = jax.grad(total)(jnp.asarray(1.0))
    assert bool(jnp.isfinite(grad))
    np.testing.assert_allclose(float(grad), 0.0, atol=1e-5)


def test_negative_depth_exponent_is_finite() -> None:
    counts = _counts(20, seed=9)
    output = normalize_counts(counts, pseudocount=1.0, depth_exponent=-0.5)
    assert bool(jnp.all(jnp.isfinite(output)))


def test_output_is_monotonic_in_counts() -> None:
    lower = jnp.asarray(np.arange(1, 21, dtype=np.float32))
    higher = lower.at[5].add(50.0)
    out_lower = normalize_counts(lower, pseudocount=1.0, depth_exponent=0.0)
    out_higher = normalize_counts(higher, pseudocount=1.0, depth_exponent=0.0)
    assert float(out_higher[5]) > float(out_lower[5])


# --- JAX / Flax NNX transform compatibility -------------------------------------


def test_normalize_counts_is_jit_compatible() -> None:
    counts = _counts(25, seed=10)
    reference = normalize_counts(counts, pseudocount=1.0, depth_exponent=1.0)
    jitted = jax.jit(normalize_counts, static_argnames=("target_sum",))
    result = jitted(counts, pseudocount=1.0, depth_exponent=1.0, target_sum=_TARGET_SUM)
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference), atol=1e-5)


def test_operator_apply_is_vmap_batchable() -> None:
    cells = jnp.asarray(np.random.default_rng(11).poisson(3.0, size=(8, 25)).astype(np.float32))
    operator = _operator()

    def one(cell: jnp.ndarray) -> jnp.ndarray:
        output, _, _ = operator.apply({"counts": cell}, {}, None)
        return output["normalized"]

    out = jax.vmap(one)(cells)
    assert out.shape == (8, 25)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_gradient_finite_under_jit() -> None:
    counts = _counts(30, seed=12)

    def total(pc: jnp.ndarray, exponent: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(normalize_counts(counts, pseudocount=pc, depth_exponent=exponent))

    grad = jax.jit(jax.grad(total, argnums=(0, 1)))(jnp.asarray(1.0), jnp.asarray(1.0))
    assert all(bool(jnp.isfinite(component)) for component in grad)


def test_gradient_through_vmap_is_finite() -> None:
    cells = jnp.asarray(np.random.default_rng(13).poisson(3.0, size=(6, 20)).astype(np.float32))

    def total(pc: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(jax.vmap(partial(normalize_counts, pseudocount=pc))(cells))

    grad = jax.grad(total)(jnp.asarray(1.0))
    assert bool(jnp.isfinite(grad))


def test_operator_is_nnx_jit_compatible() -> None:
    counts = _counts(25, seed=14)
    operator = _operator()

    @nnx.jit
    def run(module: LearnableNormalization, cell: jnp.ndarray) -> jnp.ndarray:
        output, _, _ = module.apply({"counts": cell}, {}, None)
        return output["normalized"]

    result = run(operator, counts)
    assert result.shape == (25,)
    assert bool(jnp.all(jnp.isfinite(result)))


def test_output_is_deterministic() -> None:
    counts = _counts(25, seed=15)
    first = normalize_counts(counts, pseudocount=1.2, depth_exponent=0.9)
    second = normalize_counts(counts, pseudocount=1.2, depth_exponent=0.9)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


@settings(max_examples=25, deadline=None)
@given(
    n_genes=st.integers(min_value=1, max_value=40),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_property_output_finite(n_genes: int, seed: int) -> None:
    counts = _counts(n_genes, seed)
    output = normalize_counts(counts, pseudocount=1.0, depth_exponent=1.0)
    assert bool(jnp.all(jnp.isfinite(output)))


# --- Operator contract ----------------------------------------------------------


def test_operator_apply_adds_normalized_and_passes_through() -> None:
    counts = _counts(30, seed=16)
    operator = _operator()
    output, _, _ = operator.apply({"counts": counts}, {}, None)
    assert output["normalized"].shape == (30,)
    assert "counts" in output
    assert bool(jnp.all(jnp.isfinite(output["normalized"])))


def test_operator_exposes_two_learnable_parameters() -> None:
    operator = _operator()
    params = nnx.state(operator, nnx.Param)
    assert len(jax.tree.leaves(params)) == 2


# --- Config validation and configurability --------------------------------------


def test_config_rejects_non_positive_pseudocount_init() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="pseudocount_init"):
            LearnableNormalizationConfig(pseudocount_init=bad)


def test_config_rejects_non_positive_target_sum() -> None:
    for bad in (0.0, -100.0):
        with pytest.raises(ValueError, match="target_sum"):
            LearnableNormalizationConfig(target_sum=bad)


def test_depth_exponent_init_is_configurable() -> None:
    config = LearnableNormalizationConfig(depth_exponent_init=0.5)
    operator = LearnableNormalization(config, rngs=nnx.Rngs(0))
    assert float(operator.depth_exponent[...]) == 0.5


def test_pseudocount_init_flows_into_operator() -> None:
    config = LearnableNormalizationConfig(pseudocount_init=2.5)
    operator = LearnableNormalization(config, rngs=nnx.Rngs(0))
    np.testing.assert_allclose(
        float(jax.nn.softplus(operator.raw_pseudocount[...])), 2.5, rtol=1e-5
    )


def test_library_reduction_is_fp32_under_bf16_counts() -> None:
    # Reductions must stay fp32 under mixed precision (per the DoD / jax guide):
    # a bf16 sum of many moderate counts loses precision; fp32 accumulation keeps it.
    counts_bf16 = jnp.full((4096,), 3.0, dtype=jnp.bfloat16)
    output = normalize_counts(counts_bf16, pseudocount=1.0, depth_exponent=1.0)
    counts_fp32 = counts_bf16.astype(jnp.float32)
    reference = normalize_counts(counts_fp32, pseudocount=1.0, depth_exponent=1.0)
    np.testing.assert_allclose(
        np.asarray(output, dtype=np.float32), np.asarray(reference), rtol=1e-2
    )
