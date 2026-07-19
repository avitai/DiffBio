"""Tests for the Gate-2 arm runners (frozen PCA vs learnable projection)."""

from __future__ import annotations

import numpy as np

from benchmarks.singlecell._gate2_arms import (
    ArmResult,
    SoftDimensionResult,
    per_class_f1,
    run_frozen_pca_arm,
    run_learnable_projection_arm,
    run_soft_dimension_arm,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig


def _structured_counts(
    n_cells: int, n_genes: int, n_classes: int, seed: int, signal: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """Counts whose class is encoded in a block of marker genes."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n_cells).astype(np.int32)
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    markers = max(1, n_genes // (n_classes * 2))
    for class_index in range(n_classes):
        block = slice(class_index * markers, (class_index + 1) * markers)
        counts[labels == class_index, block] += signal
    return counts, labels


def _config() -> MiniBatchConfig:
    return MiniBatchConfig(batch_size=64, n_epochs=25, learning_rate=5e-2, seed=0)


# --- per-class F1 helper --------------------------------------------------------


def test_per_class_f1_matches_hand_computation() -> None:
    predictions = np.array([0, 1, 2, 1, 0, 2])
    targets = np.array([0, 1, 1, 1, 0, 0])
    scores = per_class_f1(predictions, targets, n_classes=3)
    assert scores.shape == (3,)
    # class 0: tp=2, fp=0, fn=1 -> f1 = 2*2/(2*2+0+1) = 0.8
    assert scores[0] == np.float32(2 * 2 / (2 * 2 + 0 + 1))
    # class 1: tp=2, fp=0, fn=1 -> 0.8 ; class 2: tp=0 -> 0.0
    assert scores[1] == np.float32(2 * 2 / (2 * 2 + 0 + 1))
    assert scores[2] == np.float32(0.0)


# --- Both arms learn a separable task -------------------------------------------


def test_frozen_pca_arm_learns_and_reports_metrics() -> None:
    train_counts, train_labels = _structured_counts(300, 60, 4, seed=0)
    test_counts, test_labels = _structured_counts(120, 60, 4, seed=1)
    result = run_frozen_pca_arm(
        train_counts,
        train_labels,
        test_counts,
        test_labels,
        n_classes=4,
        n_top_genes=30,
        n_components=10,
        hidden_dim=32,
        rare_classes=np.array([], dtype=np.int64),
        config=_config(),
    )
    assert isinstance(result, ArmResult)
    assert result.macro_f1 > 0.8
    assert 0.0 <= result.balanced_accuracy <= 1.0


def test_learnable_projection_arm_learns_and_reports_metrics() -> None:
    train_counts, train_labels = _structured_counts(300, 60, 4, seed=2)
    test_counts, test_labels = _structured_counts(120, 60, 4, seed=3)
    result = run_learnable_projection_arm(
        train_counts,
        train_labels,
        test_counts,
        test_labels,
        n_classes=4,
        n_top_genes=30,
        n_components=10,
        hidden_dim=32,
        rare_classes=np.array([], dtype=np.int64),
        config=_config(),
    )
    assert isinstance(result, ArmResult)
    assert result.macro_f1 > 0.8


# --- Learnable projection starts at the PCA baseline (residual init) -------------


def test_learnable_projection_untrained_matches_frozen_pca_features() -> None:
    # The learnable-projection arm feeds mean-centered scaled features to a residual
    # projection whose delta starts at zero, so its untrained embedding must equal the
    # frozen PCA embedding exactly -- the arm starts at the PCA baseline.
    import jax.numpy as jnp
    from flax import nnx

    from benchmarks.singlecell.frozen_annotation_baseline import fit_frozen_preprocess
    from diffbio.operators.normalization.learnable_projection import (
        LearnableProjection,
        LearnableProjectionConfig,
    )

    train_counts, _ = _structured_counts(200, 60, 4, seed=4)
    test_counts, _ = _structured_counts(80, 60, 4, seed=5)
    transform = fit_frozen_preprocess(train_counts, n_top_genes=30, n_components=10)
    frozen_features = transform.transform(test_counts)

    n_hvg, n_output = transform.loadings.shape
    projection = LearnableProjection(
        LearnableProjectionConfig(n_genes=n_hvg, n_components=n_output),
        init_loadings=transform.loadings,
        rngs=nnx.Rngs(0),
    )
    centered = jnp.asarray(transform.scaled(test_counts) - transform.pca_mean)
    untrained = projection.apply({"features": centered}, {}, None)[0]["projection"]
    np.testing.assert_allclose(np.asarray(untrained), frozen_features, atol=1e-4)


# --- Rare-class breakdown -------------------------------------------------------


def test_soft_dimension_arm_learns_and_reports_effective_dimension() -> None:
    train_counts, train_labels = _structured_counts(300, 60, 4, seed=8)
    test_counts, test_labels = _structured_counts(120, 60, 4, seed=9)
    result = run_soft_dimension_arm(
        train_counts,
        train_labels,
        test_counts,
        test_labels,
        n_classes=4,
        n_top_genes=30,
        n_components=10,
        hidden_dim=32,
        rare_classes=np.array([], dtype=np.int64),
        config=_config(),
    )
    assert isinstance(result, SoftDimensionResult)
    assert result.metrics.macro_f1 > 0.8
    assert 0.0 < result.effective_dimension <= 10.0
    assert 0.0 < result.coverage < 1.0


def test_rare_class_macro_f1_is_reported() -> None:
    train_counts, train_labels = _structured_counts(300, 60, 4, seed=6)
    test_counts, test_labels = _structured_counts(120, 60, 4, seed=7)
    result = run_frozen_pca_arm(
        train_counts,
        train_labels,
        test_counts,
        test_labels,
        n_classes=4,
        n_top_genes=30,
        n_components=10,
        hidden_dim=32,
        rare_classes=np.array([0, 3], dtype=np.int64),
        config=_config(),
    )
    assert np.isfinite(result.rare_macro_f1)
    assert 0.0 <= result.rare_macro_f1 <= 1.0
