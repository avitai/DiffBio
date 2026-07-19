"""Tests for the LearnableProjection operator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.operators.normalization.learnable_projection import (
    LearnableProjection,
    LearnableProjectionConfig,
)


def _features(n_cells: int, n_genes: int, seed: int) -> jnp.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(n_cells, n_genes)).astype(np.float32)
    return jnp.asarray(matrix)


def _operator(n_genes: int, k: int, *, init=None, seed: int = 0) -> LearnableProjection:
    return LearnableProjection(
        LearnableProjectionConfig(n_genes=n_genes, n_components=k),
        init_loadings=init,
        rngs=nnx.Rngs(seed),
    )


# --- Residual init: starts exactly at the anchor --------------------------------


def test_residual_init_equals_anchor_projection() -> None:
    n_genes, k = 40, 8
    anchor = np.random.default_rng(0).normal(size=(n_genes, k)).astype(np.float32)
    features = _features(20, n_genes, seed=1)
    operator = _operator(n_genes, k, init=anchor)
    output, _, _ = operator.apply({"features": features}, {}, None)
    # delta and bias are zero at init -> projection is exactly features @ anchor
    # (compared in JAX so the matmul precision matches the operator's).
    expected = features @ jnp.asarray(anchor)
    np.testing.assert_allclose(np.asarray(output["projection"]), np.asarray(expected), atol=1e-4)


def test_residual_delta_starts_zero() -> None:
    operator = _operator(30, 6, init=np.zeros((30, 6), dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(operator.delta[...]), np.zeros((30, 6)))


# --- Output shape / contract ----------------------------------------------------


def test_apply_adds_projection_of_right_shape() -> None:
    features = _features(25, 40, seed=2)
    operator = _operator(40, 10)
    output, _, _ = operator.apply({"features": features, "labels": jnp.arange(25)}, {}, None)
    assert output["projection"].shape == (25, 10)
    assert "labels" in output
    assert bool(jnp.all(jnp.isfinite(output["projection"])))


# --- Learnability / gradients ---------------------------------------------------


def test_gradient_flows_to_projection_parameters() -> None:
    features = _features(30, 40, seed=3)
    anchor = np.random.default_rng(0).normal(size=(40, 10)).astype(np.float32)
    operator = _operator(40, 10, init=anchor)
    graphdef, params, rest = nnx.split(operator, nnx.Param, ...)

    def loss(param_state: nnx.State) -> jnp.ndarray:
        model = nnx.merge(graphdef, param_state, rest)
        output, _, _ = model.apply({"features": features}, {}, None)
        return jnp.sum(output["projection"] ** 2)

    grads = jax.grad(loss)(params)
    leaves = jax.tree.leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    # delta and bias both receive a non-zero gradient.
    assert sum(float(jnp.linalg.norm(leaf)) > 0.0 for leaf in leaves) == 2


def test_anchor_is_not_a_trainable_parameter() -> None:
    operator = _operator(30, 6, init=np.ones((30, 6), dtype=np.float32))
    params = nnx.state(operator, nnx.Param)
    # delta + bias are the only Params; the PCA anchor is a fixed Variable.
    assert len(jax.tree.leaves(params)) == 2


# --- Pure-learnable (no anchor) mode --------------------------------------------


def test_random_init_projection_is_finite_and_nonzero() -> None:
    features = _features(20, 50, seed=4)
    operator = _operator(50, 10, seed=7)  # no init_loadings -> random projection
    output, _, _ = operator.apply({"features": features}, {}, None)
    assert output["projection"].shape == (20, 10)
    assert float(jnp.linalg.norm(output["projection"])) > 0.0


# --- Transform compatibility ----------------------------------------------------


def test_apply_is_vmap_batchable() -> None:
    batch = jnp.stack([_features(15, 30, seed=s) for s in (5, 6, 7)])
    operator = _operator(30, 8)

    def one(matrix: jnp.ndarray) -> jnp.ndarray:
        return operator.apply({"features": matrix}, {}, None)[0]["projection"]

    out = jax.vmap(one)(batch)
    assert out.shape == (3, 15, 8)


def test_operator_is_nnx_jit_compatible() -> None:
    features = _features(20, 30, seed=8)
    operator = _operator(30, 6)

    @nnx.jit
    def run(module: LearnableProjection, matrix: jnp.ndarray) -> jnp.ndarray:
        return module.apply({"features": matrix}, {}, None)[0]["projection"]

    assert run(operator, features).shape == (20, 6)


# --- Config validation ----------------------------------------------------------


def test_config_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError, match="n_genes"):
        LearnableProjectionConfig(n_genes=0)
    with pytest.raises(ValueError, match="n_components"):
        LearnableProjectionConfig(n_components=-1)


def test_init_rejects_mismatched_anchor_shape() -> None:
    with pytest.raises(ValueError, match="init_loadings"):
        LearnableProjection(
            LearnableProjectionConfig(n_genes=30, n_components=6),
            init_loadings=np.zeros((30, 5), dtype=np.float32),
            rngs=nnx.Rngs(0),
        )
