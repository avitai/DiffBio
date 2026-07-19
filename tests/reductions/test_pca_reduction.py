"""Tests for the modality-agnostic frozen PCA reduction."""

from __future__ import annotations

import numpy as np

from diffbio.reductions import PCAReduction, fit_pca_reduction


def _blobs(n_samples: int, n_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_features)).astype(np.float32)


def test_fitted_state_shapes_and_ordering() -> None:
    features = _blobs(200, 40, seed=0)
    reduction = fit_pca_reduction(features, n_components=10)
    assert isinstance(reduction, PCAReduction)
    assert reduction.mean.shape == (40,)
    assert reduction.std.shape == (40,)
    assert reduction.loadings.shape == (40, 10)
    assert reduction.eigenvalues.shape == (10,)
    assert np.all(np.diff(reduction.eigenvalues) <= 1e-6)  # descending


def test_transform_matches_scaled_projection() -> None:
    features = _blobs(150, 30, seed=1)
    reduction = fit_pca_reduction(features, n_components=8)
    expected = (reduction.scaled(features) - reduction.pca_mean) @ reduction.loadings
    np.testing.assert_allclose(reduction.transform(features), expected, atol=1e-5)


def test_transform_matches_sklearn_pca() -> None:
    from sklearn.decomposition import PCA

    features = _blobs(200, 25, seed=2)
    reduction = fit_pca_reduction(features, n_components=6)
    scaled = reduction.scaled(features)
    reference = PCA(n_components=6, svd_solver="full").fit_transform(scaled)
    # Sign-invariant column-space agreement.
    np.testing.assert_allclose(np.abs(reduction.transform(features)), np.abs(reference), atol=1e-3)


def test_scaled_clips_and_standardizes() -> None:
    features = _blobs(300, 20, seed=3)
    reduction = fit_pca_reduction(features, n_components=5)
    scaled = reduction.scaled(features)
    assert scaled.max() <= 10.0 + 1e-6
    assert scaled.min() >= -10.0 - 1e-6
    # Standardized training features are ~zero-mean, ~unit-variance per feature.
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-2)


def test_output_dimension_is_capped() -> None:
    features = _blobs(12, 40, seed=4)
    # n_output = min(n_components, n_samples, n_features): capped by n_samples (12).
    reduction = fit_pca_reduction(features, n_components=30)
    assert reduction.loadings.shape[1] == 12


def test_zero_variance_feature_does_not_divide_by_zero() -> None:
    features = _blobs(100, 10, seed=5)
    features[:, 3] = 2.0  # constant feature -> std 0 -> replaced by 1
    reduction = fit_pca_reduction(features, n_components=5)
    assert bool(np.isfinite(reduction.scaled(features)).all())
    assert reduction.std[3] == np.float32(1.0)


def test_transform_applies_to_new_samples() -> None:
    train = _blobs(200, 30, seed=6)
    test = _blobs(50, 30, seed=7)
    reduction = fit_pca_reduction(train, n_components=8)
    assert reduction.transform(test).shape == (50, 8)
    assert reduction.transform(test).dtype == np.float32
