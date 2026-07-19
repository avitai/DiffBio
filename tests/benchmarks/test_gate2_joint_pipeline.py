"""Tests for the Gate-2 benchmark harness (frozen PCA vs learnable projection)."""

from __future__ import annotations

import numpy as np

from benchmarks.singlecell.gate2_joint_pipeline import (
    Gate2Comparison,
    gate2_comparison,
    rare_classes_from_counts,
    sweep_frozen_dimensions,
)


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


# --- Rare-class selection -------------------------------------------------------


def test_rare_classes_selects_the_low_frequency_classes() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 2], dtype=np.int32)
    rare = rare_classes_from_counts(labels, n_classes=3, quantile=0.5)
    # counts = [4, 2, 1]; the median is 2, so classes with count <= 2 are rare.
    assert set(rare.tolist()) == {1, 2}


# --- Full comparison ------------------------------------------------------------


def test_gate2_comparison_reports_both_arms() -> None:
    counts, labels = _structured_counts(400, 60, 4, seed=0)
    result = gate2_comparison(
        counts,
        labels,
        n_classes=4,
        seeds=(0, 1),
        n_top_genes=30,
        n_components=10,
        hidden_dim=32,
        batch_size=64,
        n_epochs=20,
        learning_rate=5e-2,
    )
    assert isinstance(result, Gate2Comparison)
    assert len(result.per_seed) == 2
    # Both arms learn the separable task.
    assert result.frozen_macro_f1_mean > 0.7
    assert result.joint_macro_f1_mean > 0.7
    # The reported gain is the joint minus frozen macro-F1.
    assert np.isclose(
        result.macro_f1_gain_mean,
        result.joint_macro_f1_mean - result.frozen_macro_f1_mean,
        atol=1e-6,
    )
    assert np.isfinite(result.macro_f1_gain_std)


def test_gate2_comparison_supervised_hvg_runs() -> None:
    counts, labels = _structured_counts(400, 60, 4, seed=8)
    result = gate2_comparison(
        counts,
        labels,
        n_classes=4,
        seeds=(0,),
        n_top_genes=30,
        n_components=10,
        hidden_dim=32,
        batch_size=64,
        n_epochs=15,
        learning_rate=5e-2,
        hvg_method="supervised",
    )
    # Both arms train on the label-selected genes and report finite metrics.
    assert np.isfinite(result.joint_macro_f1_mean)
    assert result.frozen_macro_f1_mean > 0.6


def test_gate2_comparison_is_deterministic() -> None:
    counts, labels = _structured_counts(300, 50, 3, seed=1)
    kwargs = dict(
        n_classes=3,
        seeds=(0, 1),
        n_top_genes=25,
        n_components=8,
        hidden_dim=None,
        batch_size=48,
        n_epochs=10,
        learning_rate=5e-2,
    )
    first = gate2_comparison(counts, labels, **kwargs)
    second = gate2_comparison(counts, labels, **kwargs)
    assert first.joint_macro_f1_mean == second.joint_macro_f1_mean
    assert first.frozen_macro_f1_mean == second.frozen_macro_f1_mean


def test_gate2_comparison_serializes_to_dict() -> None:
    counts, labels = _structured_counts(300, 50, 3, seed=2)
    result = gate2_comparison(
        counts,
        labels,
        n_classes=3,
        seeds=(0,),
        n_top_genes=25,
        n_components=8,
        hidden_dim=None,
        batch_size=48,
        n_epochs=8,
        learning_rate=5e-2,
    )
    payload = result.to_dict()
    import json

    encoded = json.dumps(payload)  # must be JSON-serializable
    assert "macro_f1_gain_mean" in json.loads(encoded)
    assert "frozen_rare_macro_f1_mean" in payload


# --- Swept dimensionality (Arm 1 knob) ------------------------------------------


def test_sweep_frozen_dimensions_returns_a_curve() -> None:
    counts, labels = _structured_counts(300, 60, 4, seed=3)
    curve = sweep_frozen_dimensions(
        counts,
        labels,
        n_classes=4,
        k_values=(4, 8, 16),
        seeds=(0,),
        n_top_genes=30,
        hidden_dim=None,
        batch_size=48,
        n_epochs=10,
        learning_rate=5e-2,
    )
    assert set(curve.keys()) == {4, 8, 16}
    assert all(np.isfinite(mean) for mean, _ in curve.values())
