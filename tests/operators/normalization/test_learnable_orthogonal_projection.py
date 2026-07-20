"""Tests for the learnable orthonormal (Stiefel-manifold) projection operator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.operators.normalization.learnable_orthogonal_projection import (
    LearnableOrthogonalProjection,
    LearnableOrthogonalProjectionConfig,
)


def _orthonormal(n_features: int, k: int, seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(n_features, k))
    basis, _ = np.linalg.qr(matrix)
    return basis.astype(np.float32)


def _features(n_samples: int, n_features: int, seed: int) -> jnp.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(n_samples, n_features)).astype(np.float32)
    return jnp.asarray(matrix)


def _operator(n_features: int, k: int, *, seed: int = 0) -> LearnableOrthogonalProjection:
    return LearnableOrthogonalProjection(
        LearnableOrthogonalProjectionConfig(n_features=n_features, n_components=k),
        init_loadings=_orthonormal(n_features, k, seed),
        rngs=nnx.Rngs(seed),
    )


# --- Init-at-frozen and orthonormality (the point of the Stiefel retraction) -----


def test_init_projection_equals_anchor_projection() -> None:
    n_features, k = 40, 6
    anchor = _orthonormal(n_features, k, seed=0)
    features = _features(30, n_features, seed=1)
    operator = LearnableOrthogonalProjection(
        LearnableOrthogonalProjectionConfig(n_features=n_features, n_components=k),
        init_loadings=anchor,
        rngs=nnx.Rngs(0),
    )
    output, _, _ = operator.apply({"features": features}, {}, None)
    expected = features @ jnp.asarray(anchor)
    np.testing.assert_allclose(np.asarray(output["projection"]), np.asarray(expected), atol=1e-4)


def test_basis_is_orthonormal_at_init() -> None:
    operator = _operator(50, 8)
    basis = operator.orthonormal_basis()
    gram = np.asarray(basis.T @ basis)
    np.testing.assert_allclose(gram, np.eye(8), atol=1e-3)


def test_basis_stays_orthonormal_after_gradient_step() -> None:
    features = _features(60, 40, seed=2)
    operator = _operator(40, 6)
    graphdef, params, rest = nnx.split(operator, nnx.Param, ...)

    def loss(param_state: nnx.State) -> jnp.ndarray:
        model = nnx.merge(graphdef, param_state, rest)
        return jnp.sum(model.apply({"features": features}, {}, None)[0]["projection"] ** 2)

    grads = jax.grad(loss)(params)
    updated = jax.tree.map(lambda p, g: p + 0.5 * g, params, grads)
    moved = nnx.merge(graphdef, updated, rest)
    gram = np.asarray(moved.orthonormal_basis().T @ moved.orthonormal_basis())
    # the QR retraction keeps the columns orthonormal off the anchor, unlike an
    # unconstrained learnable projection.
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-3)


# --- Learnability --------------------------------------------------------------


def test_gradient_flows_to_delta_and_bias() -> None:
    features = _features(40, 30, seed=3)
    operator = _operator(30, 5)
    graphdef, params, rest = nnx.split(operator, nnx.Param, ...)

    def loss(param_state: nnx.State) -> jnp.ndarray:
        model = nnx.merge(graphdef, param_state, rest)
        return jnp.sum(model.apply({"features": features}, {}, None)[0]["projection"] ** 2)

    grads = jax.grad(loss)(params)
    leaves = jax.tree.leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert sum(float(jnp.linalg.norm(leaf)) > 0.0 for leaf in leaves) == 2


def test_anchor_is_not_a_trainable_parameter() -> None:
    operator = _operator(30, 5)
    params = nnx.state(operator, nnx.Param)
    # delta + bias are the only Params; the orthonormal anchor is a fixed Variable.
    assert len(jax.tree.leaves(params)) == 2


# --- Output contract / transforms ----------------------------------------------


def test_apply_adds_projection_of_right_shape() -> None:
    features = _features(25, 40, seed=4)
    operator = _operator(40, 10)
    output, _, _ = operator.apply({"features": features, "labels": jnp.arange(25)}, {}, None)
    assert output["projection"].shape == (25, 10)
    assert "labels" in output


def test_operator_is_jit_compatible() -> None:
    features = _features(20, 30, seed=5)
    operator = _operator(30, 6)

    @nnx.jit
    def run(module: LearnableOrthogonalProjection, matrix: jnp.ndarray) -> jnp.ndarray:
        return module.apply({"features": matrix}, {}, None)[0]["projection"]

    assert run(operator, features).shape == (20, 6)


def test_apply_is_vmap_batchable() -> None:
    batch = jnp.stack([_features(15, 30, seed=s) for s in (5, 6, 7)])
    operator = _operator(30, 8)

    def one(matrix: jnp.ndarray) -> jnp.ndarray:
        return operator.apply({"features": matrix}, {}, None)[0]["projection"]

    assert jax.vmap(one)(batch).shape == (3, 15, 8)


# --- Config validation ----------------------------------------------------------


def test_config_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError, match="n_features"):
        LearnableOrthogonalProjectionConfig(n_features=0)
    with pytest.raises(ValueError, match="n_components"):
        LearnableOrthogonalProjectionConfig(n_components=-1)


def test_init_rejects_mismatched_anchor_shape() -> None:
    with pytest.raises(ValueError, match="init_loadings"):
        LearnableOrthogonalProjection(
            LearnableOrthogonalProjectionConfig(n_features=30, n_components=6),
            init_loadings=np.zeros((30, 5), dtype=np.float32),
            rngs=nnx.Rngs(0),
        )
