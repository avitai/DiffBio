"""Tests for the frozen classic-preprocessing cell-annotation baseline (T01)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.singlecell.frozen_annotation_baseline import (
    annotation_baseline,
    frozen_preprocess,
)


def _make_separable_counts(
    n_per_type: int,
    n_types: int,
    n_genes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise counts with a distinct marker-gene block per cell type."""
    rng = np.random.default_rng(seed)
    genes_per_type = n_genes // n_types
    counts_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for type_index in range(n_types):
        block = rng.poisson(1.0, size=(n_per_type, n_genes)).astype(np.float32)
        low = type_index * genes_per_type
        high = low + genes_per_type
        block[:, low:high] += rng.poisson(20.0, size=(n_per_type, high - low))
        counts_blocks.append(block)
        label_blocks.append(np.full(n_per_type, type_index, dtype=np.int32))
    return np.concatenate(counts_blocks), np.concatenate(label_blocks)


def test_frozen_preprocess_returns_requested_shape() -> None:
    counts, _ = _make_separable_counts(50, 4, 100, seed=0)
    features = frozen_preprocess(counts, n_top_genes=40, n_components=10)
    assert features.shape == (200, 10)
    assert bool(np.all(np.isfinite(features)))


def test_frozen_preprocess_clamps_components_to_available_dimensions() -> None:
    counts, _ = _make_separable_counts(10, 2, 20, seed=0)
    features = frozen_preprocess(counts, n_top_genes=8, n_components=50)
    assert features.shape[0] == 20
    assert features.shape[1] <= 8


def test_annotation_baseline_separates_well_separated_types() -> None:
    counts, labels = _make_separable_counts(50, 4, 100, seed=0)
    features = frozen_preprocess(counts, n_top_genes=40, n_components=10)
    metrics = annotation_baseline(features, labels, n_classes=4, seed=42, n_train_steps=300)
    assert metrics["macro_f1"] > 0.8
    assert metrics["balanced_accuracy"] > 0.8


def test_annotation_baseline_is_deterministic() -> None:
    counts, labels = _make_separable_counts(40, 3, 90, seed=1)
    features = frozen_preprocess(counts, n_top_genes=30, n_components=8)
    first = annotation_baseline(features, labels, n_classes=3, seed=7, n_train_steps=100)
    second = annotation_baseline(features, labels, n_classes=3, seed=7, n_train_steps=100)
    assert first == second


def test_annotation_baseline_rejects_mismatched_labels() -> None:
    counts, labels = _make_separable_counts(20, 2, 40, seed=2)
    features = frozen_preprocess(counts, n_top_genes=20, n_components=6)
    with pytest.raises(ValueError, match="labels"):
        annotation_baseline(features[:-1], labels, n_classes=2, seed=0, n_train_steps=10)
