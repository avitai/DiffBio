"""Tests for the fitted FrozenTransform and its parity with frozen_preprocess."""

from __future__ import annotations

import numpy as np

from benchmarks.singlecell.frozen_annotation_baseline import (
    FrozenTransform,
    fit_frozen_preprocess,
    frozen_preprocess,
    supervised_hvg_indices,
)


def _structured_counts(
    n_cells: int, n_genes: int, n_classes: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Counts whose class is encoded in a block of marker genes."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n_cells)
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    markers = max(1, n_genes // (n_classes * 2))
    for class_index in range(n_classes):
        block = slice(class_index * markers, (class_index + 1) * markers)
        counts[labels == class_index, block] += 20.0
    return counts, labels


# --- Parity: fit-then-transform on the same data == frozen_preprocess -----------


def test_fit_transform_matches_frozen_preprocess() -> None:
    counts, _ = _structured_counts(200, 60, 4, seed=0)
    transform = fit_frozen_preprocess(counts, n_top_genes=30, n_components=10)
    replicated = transform.transform(counts)
    reference = frozen_preprocess(counts, n_top_genes=30, n_components=10)
    # Same subspace up to the exact fit_transform scores; must match closely.
    np.testing.assert_allclose(replicated, reference, atol=1e-4)


def test_frozen_preprocess_still_delegates_unchanged() -> None:
    # frozen_preprocess must keep its exact output for its existing callers.
    counts, _ = _structured_counts(150, 40, 3, seed=1)
    features = frozen_preprocess(counts, n_top_genes=20, n_components=8)
    assert features.shape == (150, 8)
    assert features.dtype == np.float32


# --- Fitted-state contract ------------------------------------------------------


def test_fitted_state_shapes_and_ordering() -> None:
    counts, _ = _structured_counts(200, 60, 4, seed=2)
    transform = fit_frozen_preprocess(counts, n_top_genes=30, n_components=10)
    assert isinstance(transform, FrozenTransform)
    assert transform.hvg_indices.shape == (30,)
    assert transform.mean.shape == (30,)
    assert transform.std.shape == (30,)
    assert transform.loadings.shape == (30, 10)
    assert transform.eigenvalues.shape == (10,)
    # Eigenvalues (explained variance) are returned in descending order.
    assert np.all(np.diff(transform.eigenvalues) <= 1e-6)


def test_output_dimension_is_capped() -> None:
    counts, _ = _structured_counts(40, 60, 4, seed=3)
    # n_output = min(n_components, n_cells, n_hvg): here capped by n_cells (40).
    transform = fit_frozen_preprocess(counts, n_top_genes=50, n_components=45)
    assert transform.loadings.shape[1] == 40
    assert transform.transform(counts).shape == (40, 40)


# --- Fit on train, apply to a held-out split (no leakage) -----------------------


def test_transform_applies_train_fit_to_new_cells() -> None:
    train_counts, _ = _structured_counts(200, 60, 4, seed=4)
    test_counts, _ = _structured_counts(80, 60, 4, seed=5)
    transform = fit_frozen_preprocess(train_counts, n_top_genes=30, n_components=10)
    test_features = transform.transform(test_counts)
    assert test_features.shape == (80, 10)
    # The transform uses the train-fitted genes/mean/std, not the test data's:
    # refitting on the test data gives materially different features.
    refit = fit_frozen_preprocess(test_counts, n_top_genes=30, n_components=10)
    assert not np.allclose(test_features, refit.transform(test_counts), atol=1e-3)


# --- Scaled (pre-PCA) features for the learnable-projection arm ------------------


def test_scaled_features_feed_the_projection_consistently() -> None:
    counts, _ = _structured_counts(200, 60, 4, seed=6)
    transform = fit_frozen_preprocess(counts, n_top_genes=30, n_components=10)
    scaled = transform.scaled(counts)
    assert scaled.shape == (200, 30)
    # transform == (scaled - pca_mean) @ loadings, so the projection arm can
    # start exactly at the PCA features by matmul against the loadings.
    projected = (scaled - transform.pca_mean) @ transform.loadings
    np.testing.assert_allclose(projected, transform.transform(counts), atol=1e-4)


# --- Supervised (Wilcoxon) HVG selection ----------------------------------------


def _log_normalized(counts: np.ndarray) -> np.ndarray:
    library = counts.sum(axis=1, keepdims=True)
    library = np.where(library == 0.0, 1.0, library)
    return np.log1p(counts / library * 1.0e4)


def test_supervised_hvg_returns_sorted_unique_indices() -> None:
    counts, labels = _structured_counts(300, 80, 4, seed=0)
    indices = supervised_hvg_indices(_log_normalized(counts), labels, n_top=30)
    assert indices.shape == (30,)
    assert np.all(np.diff(indices) > 0)  # sorted, unique
    assert indices.min() >= 0 and indices.max() < 80


def test_supervised_hvg_prefers_class_discriminative_markers() -> None:
    # Marker genes (the first n_classes*markers columns) carry the class signal, so
    # a supervised selection must recover them far better than picking at random.
    n_genes, n_classes = 80, 4
    counts, labels = _structured_counts(400, n_genes, n_classes, seed=1)
    markers = max(1, n_genes // (n_classes * 2))
    marker_genes = set(range(n_classes * markers))
    indices = supervised_hvg_indices(_log_normalized(counts), labels, n_top=len(marker_genes))
    recovered = len(marker_genes & set(indices.tolist()))
    # Nearly all marker genes should be selected within a budget of their own size.
    assert recovered >= len(marker_genes) - 2


def test_supervised_and_dispersion_hvg_differ() -> None:
    counts, labels = _structured_counts(300, 80, 4, seed=2)
    supervised = fit_frozen_preprocess(
        counts, n_top_genes=30, n_components=10, hvg_method="supervised", labels=labels
    )
    dispersion = fit_frozen_preprocess(counts, n_top_genes=30, n_components=10)
    assert not np.array_equal(supervised.hvg_indices, dispersion.hvg_indices)


def test_supervised_hvg_requires_labels() -> None:
    import pytest

    counts, _ = _structured_counts(100, 40, 3, seed=3)
    with pytest.raises(ValueError, match="labels"):
        fit_frozen_preprocess(counts, n_top_genes=20, n_components=8, hvg_method="supervised")
