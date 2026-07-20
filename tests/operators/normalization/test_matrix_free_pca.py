"""Tests for the matrix-free (differentiable subspace-/power-iteration) PCA operator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.operators.normalization.matrix_free_pca import (
    MatrixFreePCA,
    MatrixFreePCAConfig,
    matfree_pca,
)


def _data(n_samples: int, n_features: int, seed: int) -> jnp.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(n_samples, n_features)).astype(np.float32)
    return jnp.asarray(matrix)


def _structured_data(n_samples: int, n_features: int, rank: int, seed: int) -> jnp.ndarray:
    """Low-rank-plus-noise data with a decaying spectrum (well-separated top components).

    This is the regime PCA is used in and where the subspace-iteration solver's extremal
    singular directions converge; a flat (isotropic Gaussian) spectrum has no spectral gap
    and is ill-posed for any partial eigensolver.
    """
    rng = np.random.default_rng(seed)
    variances = np.linspace(6.0, 2.0, rank)
    factors = rng.normal(size=(n_samples, rank)) * variances
    loadings = rng.normal(size=(rank, n_features))
    signal = factors @ loadings + 0.1 * rng.normal(size=(n_samples, n_features))
    return jnp.asarray(signal.astype(np.float32))


def _sklearn_pca(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    model = PCA(n_components=k, svd_solver="full").fit(features)
    return model.transform(features), model.explained_variance_


# --- Correctness against a reference PCA -----------------------------------------


def test_eigenvalues_match_full_pca() -> None:
    features = _structured_data(500, 40, rank=8, seed=0)
    _, _, explained = matfree_pca(features, n_components=5)
    _, ref_var = _sklearn_pca(np.asarray(features), 5)
    # Subspace iteration converges the top eigenvalues; explained variance matches sklearn's.
    np.testing.assert_allclose(np.asarray(explained), ref_var, rtol=1e-3)


def test_scores_match_full_pca_up_to_sign() -> None:
    features = _structured_data(500, 40, rank=8, seed=1)
    scores, _, _ = matfree_pca(features, n_components=5)
    ref_scores, _ = _sklearn_pca(np.asarray(features), 5)
    # svd_flip makes both deterministic; compare absolute values to ignore residual sign.
    np.testing.assert_allclose(np.abs(np.asarray(scores)), np.abs(ref_scores), rtol=1e-2, atol=1e-2)


def test_shapes_and_svd_flip_sign_convention() -> None:
    features = _data(200, 24, seed=2)
    scores, components, explained = matfree_pca(features, n_components=6)
    assert scores.shape == (200, 6)
    assert components.shape == (6, 24)
    assert explained.shape == (6,)
    # svd_flip: the largest-magnitude entry of each loading is positive.
    loadings = np.asarray(components.T)
    max_abs = np.argmax(np.abs(loadings), axis=0)
    assert np.all(loadings[max_abs, np.arange(6)] > 0.0)


# --- Differentiability and stability (the point of the matrix-free solver) -------


def test_gradient_flows_to_input_features() -> None:
    features = _data(150, 20, seed=3)

    def loss(matrix: jnp.ndarray) -> jnp.ndarray:
        scores, _, _ = matfree_pca(matrix, n_components=4)
        return jnp.sum(scores**2)

    grad = jax.grad(loss)(features)
    assert grad.shape == features.shape
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0


def test_gradient_finite_on_rank_deficient_data() -> None:
    # Rank-deficient (duplicated-column) data gives the covariance many zero eigenvalues;
    # the matrix-free subspace iteration flows gradients through matmuls and a QR and stays
    # finite, since the top components it keeps are the well-separated non-zero directions.
    rng = np.random.default_rng(4)
    base = rng.normal(size=(300, 12)).astype(np.float32)
    features = jnp.asarray(np.hstack([base, base]))

    def loss(matrix: jnp.ndarray) -> jnp.ndarray:
        scores, _, _ = matfree_pca(matrix, n_components=4)
        return jnp.sum(scores**2)

    grad = jax.grad(loss)(features)
    assert bool(jnp.all(jnp.isfinite(grad)))


# --- Operator wrapper ------------------------------------------------------------


def test_operator_adds_expected_keys() -> None:
    features = _data(120, 20, seed=5)
    operator = MatrixFreePCA(MatrixFreePCAConfig(n_components=5))
    output, _, _ = operator.apply({"features": features, "labels": jnp.arange(120)}, {}, None)
    assert output["pca"].shape == (120, 5)
    assert output["pca_components"].shape == (5, 20)
    assert output["explained_variance"].shape == (5,)
    assert "labels" in output


def test_operator_has_no_trainable_parameters() -> None:
    # The basis is computed from the data, not stored -- there is no fixed anchor.
    operator = MatrixFreePCA(MatrixFreePCAConfig(n_components=5))
    assert len(jax.tree.leaves(nnx.state(operator, nnx.Param))) == 0


def test_operator_is_jit_compatible() -> None:
    features = _data(120, 20, seed=6)
    operator = MatrixFreePCA(MatrixFreePCAConfig(n_components=5))

    @nnx.jit
    def run(module: MatrixFreePCA, matrix: jnp.ndarray) -> jnp.ndarray:
        return module.apply({"features": matrix}, {}, None)[0]["pca"]

    assert run(operator, features).shape == (120, 5)


def test_default_config_runs_with_larger_k() -> None:
    features = _data(200, 60, seed=7)
    # Default num_iterations/oversampling; k=10 with n_features=60 leaves room for the sketch.
    operator = MatrixFreePCA(MatrixFreePCAConfig(n_components=10))
    output, _, _ = operator.apply({"features": features}, {}, None)
    assert output["pca"].shape == (200, 10)


# --- Config validation -----------------------------------------------------------


def test_config_rejects_non_positive_components() -> None:
    with pytest.raises(ValueError, match="n_components"):
        MatrixFreePCAConfig(n_components=0)


def test_config_rejects_negative_iterations_and_oversampling() -> None:
    with pytest.raises(ValueError, match="num_iterations"):
        MatrixFreePCAConfig(n_components=5, num_iterations=-1)
    with pytest.raises(ValueError, match="oversampling"):
        MatrixFreePCAConfig(n_components=5, oversampling=-1)
