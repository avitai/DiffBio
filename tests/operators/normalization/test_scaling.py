"""Tests for the DifferentiableScaler operator (ticket 05)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st

from diffbio.operators.normalization.scaling import (
    DifferentiableScaler,
    ScalerConfig,
    standardize_features,
)

_CLIP = 10.0


def _features(n_cells: int, n_genes: int, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(2.0, 3.0, size=(n_cells, n_genes)).astype(np.float32))


def _operator(clip: float = _CLIP) -> DifferentiableScaler:
    return DifferentiableScaler(ScalerConfig(clip=clip), rngs=nnx.Rngs(0))


# --- Parity with the frozen baseline scaling ------------------------------------


def test_matches_baseline_zscore_clip() -> None:
    features = _features(200, 30, seed=0)
    array = np.asarray(features)
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    expected = np.clip((array - mean) / std, -_CLIP, _CLIP)
    output = standardize_features(features, clip=_CLIP)
    np.testing.assert_allclose(np.asarray(output), expected, atol=1e-4)


def test_standardized_columns_are_zero_mean_unit_variance() -> None:
    # Without clipping active, each gene column is standardized across cells.
    features = _features(300, 12, seed=1)
    output = standardize_features(features, clip=_CLIP)
    np.testing.assert_allclose(np.asarray(jnp.mean(output, axis=0)), np.zeros(12), atol=1e-4)
    np.testing.assert_allclose(np.asarray(jnp.std(output, axis=0)), np.ones(12), atol=1e-3)


# --- Edge cases -----------------------------------------------------------------


def test_constant_gene_uses_unit_std_and_is_zero() -> None:
    array = np.array(_features(50, 6, seed=2))
    array[:, 3] = 4.0  # constant column: std == 0.
    output = standardize_features(jnp.asarray(array), clip=_CLIP)
    assert bool(jnp.all(jnp.isfinite(output)))
    np.testing.assert_allclose(np.asarray(output[:, 3]), np.zeros(50), atol=1e-6)


def test_output_is_clipped_to_bound() -> None:
    array = np.array(_features(100, 8, seed=3))
    array[0, 0] = 1.0e6  # extreme outlier -> z-score far beyond the clip.
    output = standardize_features(jnp.asarray(array), clip=_CLIP)
    assert float(jnp.max(output)) <= _CLIP
    assert float(jnp.min(output)) >= -_CLIP


def test_custom_clip_bound() -> None:
    array = np.array(_features(100, 8, seed=4))
    array[0, 0] = 1.0e6
    output = standardize_features(jnp.asarray(array), clip=3.0)
    assert float(jnp.max(output)) <= 3.0


# --- Differentiability ----------------------------------------------------------


def test_gradient_flows_through_scaler() -> None:
    features = _features(60, 10, seed=5)

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(standardize_features(x, clip=_CLIP) ** 2)

    grad = jax.grad(loss)(features)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0


def test_gradient_finite_for_zero_variance_gene() -> None:
    # A constant (e.g. HVG-gated) gene has variance 0; the standardization must
    # guard before the square root so it does not emit a NaN gradient.
    array = np.array(_features(40, 6, seed=13))
    array[:, 2] = 5.0  # constant column: variance 0.
    features = jnp.asarray(array)

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(standardize_features(x, clip=_CLIP) ** 2)

    grad = jax.grad(loss)(features)
    assert bool(jnp.all(jnp.isfinite(grad)))


# --- JAX / Flax NNX transform compatibility -------------------------------------


def test_is_jit_compatible() -> None:
    features = _features(80, 15, seed=6)
    reference = standardize_features(features, clip=_CLIP)
    result = jax.jit(partial(standardize_features, clip=_CLIP))(features)
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference), atol=1e-5)


def test_operator_apply_is_vmap_batchable() -> None:
    batch = jnp.stack([_features(40, 10, seed=s) for s in (7, 8, 9)])
    operator = _operator()

    def one(matrix: jnp.ndarray) -> jnp.ndarray:
        output, _, _ = operator.apply({"features": matrix}, {}, None)
        return output["features"]

    out = jax.vmap(one)(batch)
    assert out.shape == (3, 40, 10)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_operator_is_nnx_jit_compatible() -> None:
    features = _features(50, 12, seed=10)
    operator = _operator()

    @nnx.jit
    def run(module: DifferentiableScaler, matrix: jnp.ndarray) -> jnp.ndarray:
        output, _, _ = module.apply({"features": matrix}, {}, None)
        return output["features"]

    result = run(operator, features)
    assert result.shape == (50, 12)
    assert bool(jnp.all(jnp.isfinite(result)))


def test_output_is_deterministic() -> None:
    features = _features(50, 10, seed=11)
    first = standardize_features(features, clip=_CLIP)
    second = standardize_features(features, clip=_CLIP)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


@settings(max_examples=25, deadline=None)
@given(
    n_cells=st.integers(min_value=2, max_value=60),
    n_genes=st.integers(min_value=1, max_value=40),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_property_output_finite_and_bounded(n_cells: int, n_genes: int, seed: int) -> None:
    features = _features(n_cells, n_genes, seed)
    output = standardize_features(features, clip=_CLIP)
    assert bool(jnp.all(jnp.isfinite(output)))
    assert float(jnp.max(jnp.abs(output))) <= _CLIP + 1e-4


# --- Config validation and operator contract ------------------------------------


def test_config_rejects_non_positive_clip() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="clip"):
            ScalerConfig(clip=bad)


def test_operator_apply_overwrites_features_and_passes_through() -> None:
    features = _features(40, 10, seed=12)
    operator = _operator()
    output, _, _ = operator.apply({"features": features, "labels": jnp.arange(40)}, {}, None)
    assert output["features"].shape == (40, 10)
    assert "labels" in output
    assert float(jnp.max(jnp.abs(output["features"]))) <= _CLIP + 1e-4


def test_operator_has_no_learnable_parameters() -> None:
    operator = _operator()
    params = nnx.state(operator, nnx.Param)
    assert len(jax.tree.leaves(params)) == 0
