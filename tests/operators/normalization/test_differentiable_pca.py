"""Tests for the robust DifferentiablePCA operator (ticket 02)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.decomposition import PCA

from diffbio.operators.normalization.differentiable_pca import (
    DifferentiablePCA,
    DifferentiablePCAConfig,
    robust_pca,
    safe_eigh,
)


# --- Fixtures -------------------------------------------------------------------


def _well_separated_features(n_cells: int, n_genes: int, seed: int) -> np.ndarray:
    """Features with a clearly separated (geometrically decaying) spectrum."""
    rng = np.random.default_rng(seed)
    rank = min(n_genes, 6)
    scales = np.array([10.0 * (0.5**index) for index in range(rank)])
    latent = rng.normal(size=(n_cells, rank)) * scales
    loadings = np.linalg.qr(rng.normal(size=(n_genes, rank)))[0]
    dense = latent @ loadings.T + rng.normal(0.0, 0.01, size=(n_cells, n_genes))
    return dense.astype(np.float32)


def _near_tied_features(n_cells: int, n_genes: int, n_types: int, seed: int) -> np.ndarray:
    """Features whose covariance has near-tied top eigenvalues (equal blocks)."""
    rng = np.random.default_rng(seed)
    block = n_genes // n_types
    per = n_cells // n_types
    rows: list[np.ndarray] = []
    for group in range(n_types):
        base = rng.normal(0.0, 0.3, size=(per, n_genes))
        base[:, group * block : (group + 1) * block] += 4.0
        rows.append(base)
    return np.concatenate(rows).astype(np.float32)


def _symmetric_with_spectrum(eigenvalues: np.ndarray, seed: int) -> jnp.ndarray:
    """Build a symmetric matrix with a prescribed eigenvalue spectrum."""
    rng = np.random.default_rng(seed)
    dim = eigenvalues.shape[0]
    basis, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    matrix = basis @ np.diag(eigenvalues) @ basis.T
    return jnp.asarray(0.5 * (matrix + matrix.T), dtype=jnp.float32)


def _projector(components: jnp.ndarray) -> jnp.ndarray:
    """Rank-k orthogonal projector V V^T from components of shape (k, p)."""
    basis = components.T
    return basis @ basis.T


def _covariance(features: jnp.ndarray) -> jnp.ndarray:
    centered = features - jnp.mean(features, axis=0, keepdims=True)
    return (centered.T @ centered) / (features.shape[0] - 1)


# --- Parity with sklearn (well-separated spectrum) ------------------------------


def test_frozen_parity_matches_sklearn() -> None:
    features = _well_separated_features(120, 40, seed=0)
    n_components = 6

    _, components, explained_variance = robust_pca(features, n_components)

    reference = PCA(n_components=n_components, random_state=0).fit(features)
    our_ratio = np.asarray(explained_variance / explained_variance.sum())
    np.testing.assert_allclose(our_ratio, reference.explained_variance_ratio_, atol=1e-4)

    projector_gap = jnp.linalg.norm(
        _projector(components) - _projector(jnp.asarray(reference.components_))
    )
    assert float(projector_gap) < 1e-3


def test_orthogonal_rotation_invariance() -> None:
    features = _well_separated_features(80, 30, seed=1)
    rng = np.random.default_rng(7)
    rotation, _ = np.linalg.qr(rng.normal(size=(30, 30)))
    rotated = (features @ rotation).astype(np.float32)

    _, _, ev_original = robust_pca(features, 5)
    _, _, ev_rotated = robust_pca(rotated, 5)
    np.testing.assert_allclose(np.asarray(ev_original), np.asarray(ev_rotated), atol=1e-3)


def test_svd_flip_sign_convention_is_deterministic() -> None:
    features = _well_separated_features(60, 20, seed=5)
    _, components_first, _ = robust_pca(features, 5)
    _, components_second, _ = robust_pca(features, 5)
    np.testing.assert_array_equal(np.asarray(components_first), np.asarray(components_second))

    loadings = np.asarray(components_first)
    max_abs_per_component = loadings[np.arange(5), np.argmax(np.abs(loadings), axis=1)]
    assert bool(np.all(max_abs_per_component > 0.0))


# --- Gradient correctness ------------------------------------------------------


def test_eigenvector_gradient_matches_builtin_eigh() -> None:
    """On a well-separated spectrum the regularized reciprocal equals 1/gap, so the
    custom eigenvector VJP must match jnp.linalg.eigh's exact VJP (catches sign or
    convention errors that finiteness tests miss)."""
    matrix = _symmetric_with_spectrum(np.array([10.0, 6.0, 3.0, 1.0]), seed=8)
    ramp = jnp.arange(1, 5, dtype=jnp.float32)[:, None]

    def loss_safe(a: jnp.ndarray) -> jnp.ndarray:
        _, eigenvectors = safe_eigh(a)
        return jnp.sum(ramp * eigenvectors)

    def loss_builtin(a: jnp.ndarray) -> jnp.ndarray:
        _, eigenvectors = jnp.linalg.eigh(a)
        return jnp.sum(ramp * eigenvectors)

    grad_safe = jax.grad(loss_safe)(matrix)
    grad_builtin = jax.grad(loss_builtin)(matrix)
    np.testing.assert_allclose(np.asarray(grad_safe), np.asarray(grad_builtin), atol=1e-4)


# --- Gradient correctness on the exact (eigenvalue) path ------------------------


def test_eigenvalue_gradient_matches_trace_identity() -> None:
    """sum(eigenvalues) == trace(cov); their gradients must agree exactly."""
    features = jnp.asarray(_well_separated_features(50, 16, seed=3))

    def via_eigh(x: jnp.ndarray) -> jnp.ndarray:
        eigenvalues, _ = safe_eigh(_covariance(x))
        return eigenvalues.sum()

    def via_trace(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.trace(_covariance(x))

    grad_custom = jax.grad(via_eigh)(features)
    grad_true = jax.grad(via_trace)(features)
    np.testing.assert_allclose(np.asarray(grad_custom), np.asarray(grad_true), atol=1e-4)


# --- PRIMARY GATE: robust gradients under degenerate / near-tied eigenvalues ----


def test_gradients_finite_on_isotropic_features() -> None:
    rng = np.random.default_rng(11)
    features = jnp.asarray(rng.normal(size=(64, 16)).astype(np.float32))

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        scores, _, _ = robust_pca(x, 8)
        return jnp.sum(scores**2)

    grad = jax.grad(loss)(features)
    assert bool(jnp.all(jnp.isfinite(grad)))


@pytest.mark.parametrize("gap", [1e-3, 1e-6, 1e-9, 1e-12, 0.0])
def test_gradients_bounded_across_eigenvalue_gap_sweep(gap: float) -> None:
    eigenvalues = np.array([3.0, 2.0 + gap, 2.0, 1.0], dtype=np.float64)
    matrix = _symmetric_with_spectrum(eigenvalues, seed=5)

    def loss(a: jnp.ndarray) -> jnp.ndarray:
        eigvals, eigvecs = safe_eigh(a)
        return jnp.sum(eigvals) + jnp.sum(eigvecs**2)

    grad = jax.grad(loss)(matrix)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) < 1e4


def test_gradients_finite_on_exactly_repeated_spectrum() -> None:
    """An exact isotropic block (repeated eigenvalues) must not NaN."""
    matrix = _symmetric_with_spectrum(np.array([3.0, 1.0, 1.0, 1.0]), seed=9)

    def loss(a: jnp.ndarray) -> jnp.ndarray:
        _, eigvecs = safe_eigh(a)
        return jnp.sum(eigvecs**2)

    grad = jax.grad(loss)(matrix)
    assert bool(jnp.all(jnp.isfinite(grad)))


@settings(max_examples=25, deadline=None)
@given(
    n_cells=st.integers(min_value=20, max_value=40),
    n_genes=st.integers(min_value=6, max_value=16),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_property_gradients_always_finite(n_cells: int, n_genes: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    features = jnp.asarray(rng.normal(size=(n_cells, n_genes)).astype(np.float32))

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        scores, _, explained_variance = robust_pca(x, min(5, n_genes))
        return jnp.sum(scores**2) + explained_variance.sum()

    grad = jax.grad(loss)(features)
    assert bool(jnp.all(jnp.isfinite(grad)))


# --- Edge / corner cases --------------------------------------------------------


def test_single_component() -> None:
    features = jnp.asarray(_well_separated_features(40, 12, seed=4))
    scores, components, explained_variance = robust_pca(features, 1)
    assert scores.shape == (40, 1)
    assert components.shape == (1, 12)
    assert explained_variance.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(scores)))


def test_single_feature() -> None:
    features = jnp.asarray(np.random.default_rng(0).normal(size=(30, 1)).astype(np.float32))
    scores, components, explained_variance = robust_pca(features, 5)
    assert scores.shape == (30, 1)
    assert components.shape == (1, 1)
    assert explained_variance.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(scores)))


def test_two_cells() -> None:
    features = jnp.asarray(np.random.default_rng(1).normal(size=(2, 8)).astype(np.float32))
    scores, _, _ = robust_pca(features, 5)
    assert scores.shape[0] == 2
    assert bool(jnp.all(jnp.isfinite(scores)))


def test_zero_variance_feature_is_finite() -> None:
    base = _well_separated_features(40, 10, seed=6)
    base[:, 3] = 2.0  # constant (zero-variance) gene
    features = jnp.asarray(base)

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        scores, _, _ = robust_pca(x, 5)
        return jnp.sum(scores**2)

    grad = jax.grad(loss)(features)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_all_zero_features_is_finite() -> None:
    features = jnp.zeros((20, 8), dtype=jnp.float32)

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        scores, _, explained_variance = robust_pca(x, 4)
        return jnp.sum(scores**2) + explained_variance.sum()

    value = loss(features)
    grad = jax.grad(loss)(features)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_rank_deficient_more_genes_than_cells_is_finite() -> None:
    features = jnp.asarray(np.random.default_rng(2).normal(size=(8, 20)).astype(np.float32))

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        scores, _, _ = robust_pca(x, 6)
        return jnp.sum(scores**2)

    scores, _, _ = robust_pca(features, 6)
    grad = jax.grad(loss)(features)
    assert bool(jnp.all(jnp.isfinite(scores)))
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_components_are_clamped_to_available_rank() -> None:
    features = _near_tied_features(12, 6, 2, seed=0)
    scores, components, explained_variance = robust_pca(features, n_components=50)
    assert scores.shape[1] <= 6
    assert components.shape[0] == scores.shape[1]
    assert explained_variance.shape[0] == scores.shape[1]


# --- JAX / Flax NNX transform compatibility -------------------------------------


def test_robust_pca_is_jit_compatible() -> None:
    features = jnp.asarray(_well_separated_features(40, 12, seed=0))
    reference, _, _ = robust_pca(features, 5)
    jitted = jax.jit(partial(robust_pca, n_components=5))
    scores, _, _ = jitted(features)
    np.testing.assert_allclose(np.asarray(scores), np.asarray(reference), atol=1e-4)


def test_gradient_is_finite_under_jit() -> None:
    features = jnp.asarray(_well_separated_features(40, 12, seed=1))

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        scores, _, explained_variance = robust_pca(x, 5)
        return jnp.sum(scores**2) + explained_variance.sum()

    grad = jax.jit(jax.grad(loss))(features)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_robust_pca_is_vmap_compatible() -> None:
    batch = jnp.asarray(
        np.stack([_well_separated_features(40, 12, seed=index) for index in range(4)])
    )
    batched = jax.vmap(partial(robust_pca, n_components=5))
    scores, components, explained_variance = batched(batch)
    assert scores.shape == (4, 40, 5)
    assert components.shape == (4, 5, 12)
    assert explained_variance.shape == (4, 5)
    assert bool(jnp.all(jnp.isfinite(scores)))


def test_gradient_through_vmap_is_finite() -> None:
    batch = jnp.asarray(
        np.stack([_well_separated_features(30, 10, seed=index) for index in range(3)])
    )

    def loss(features_batch: jnp.ndarray) -> jnp.ndarray:
        scores = jax.vmap(partial(robust_pca, n_components=4))(features_batch)[0]
        return jnp.sum(scores**2)

    grad = jax.grad(loss)(batch)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_safe_eigh_is_vmap_compatible() -> None:
    matrices = jnp.stack(
        [_symmetric_with_spectrum(np.array([3.0, 2.0, 1.0, 0.5]), seed=index) for index in range(3)]
    )
    eigenvalues, eigenvectors = jax.vmap(safe_eigh)(matrices)
    assert eigenvalues.shape == (3, 4)
    assert eigenvectors.shape == (3, 4, 4)


def test_operator_apply_is_vmap_batchable() -> None:
    """apply must vmap over a batch of matrices, as OperatorModule.apply_batch does."""
    batch = jnp.asarray(
        np.stack([_well_separated_features(30, 12, seed=index) for index in range(3)])
    )
    operator = DifferentiablePCA(DifferentiablePCAConfig(n_components=5), rngs=nnx.Rngs(0))

    def one(features: jnp.ndarray) -> jnp.ndarray:
        output, _, _ = operator.apply({"features": features}, {}, None)
        return output["pca"]

    scores = jax.vmap(one)(batch)
    assert scores.shape == (3, 30, 5)
    assert bool(jnp.all(jnp.isfinite(scores)))


def test_operator_is_nnx_jit_compatible() -> None:
    features = jnp.asarray(_well_separated_features(30, 12, seed=2))
    operator = DifferentiablePCA(DifferentiablePCAConfig(n_components=5), rngs=nnx.Rngs(0))

    @nnx.jit
    def run(module: DifferentiablePCA, feats: jnp.ndarray) -> jnp.ndarray:
        output, _, _ = module.apply({"features": feats}, {}, None)
        return output["pca"]

    scores = run(operator, features)
    assert scores.shape == (30, 5)
    assert bool(jnp.all(jnp.isfinite(scores)))


# --- Operator contract ----------------------------------------------------------


def test_operator_apply_returns_scores_and_flows_gradients() -> None:
    features = jnp.asarray(_well_separated_features(50, 20, seed=2))
    operator = DifferentiablePCA(DifferentiablePCAConfig(n_components=8), rngs=nnx.Rngs(0))

    output, _, _ = operator.apply({"features": features}, {}, None)
    assert output["pca"].shape == (50, 8)
    assert "explained_variance" in output
    assert "pca_components" in output

    def loss(feats: jnp.ndarray) -> jnp.ndarray:
        result, _, _ = operator.apply({"features": feats}, {}, None)
        return jnp.sum(result["pca"] ** 2)

    grad = jax.grad(loss)(features)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0
