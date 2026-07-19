"""Gate 1 tests: frozen pipeline reproduces the baseline annotation F1 (ticket 05)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.singlecell.frozen_annotation_baseline import frozen_preprocess
from benchmarks.singlecell.gate1_frozen_pipeline import (
    gate1_annotation_comparison,
    pipeline_frozen_features,
)

_ATLAS_DATA_DIR = os.environ.get("DIFFBIO_SCIB_DATA_DIR", "/media/mahdi/ssd23/Data/scib")


def _structured_counts(
    n_cells: int, n_genes: int, n_classes: int, seed: int, signal: float
) -> tuple[np.ndarray, np.ndarray]:
    """Counts whose class is encoded (noisily) in a block of marker genes."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n_cells)
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    markers = max(1, n_genes // (n_classes * 2))
    for class_index in range(n_classes):
        block = slice(class_index * markers, (class_index + 1) * markers)
        counts[labels == class_index, block] += signal
    return counts, labels


def _near_degenerate_counts(seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Counts whose scaled PCA spectrum is near-degenerate at the component cutoff.

    Many genes with a weak class signal give a nearly flat top-k eigenvalue
    spectrum, where exact and randomized SVD pick different subspaces -- the case
    that must stay matched for Gate 1.
    """
    rng = np.random.default_rng(seed)
    n_cells, n_genes, n_classes = 250, 400, 4
    labels = rng.integers(0, n_classes, size=n_cells)
    counts = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
    markers = 8
    for class_index in range(n_classes):
        block = slice(class_index * markers, (class_index + 1) * markers)
        counts[labels == class_index, block] += 4.0
    return counts, labels, n_classes


def _subspace_distance(a: np.ndarray, b: np.ndarray) -> float:
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    return float(np.linalg.norm(qa @ qa.T - qb @ qb.T))


# --- Gate 1: macro-F1 parity ----------------------------------------------------


def test_frozen_pipeline_reproduces_baseline_macro_f1() -> None:
    # A moderate signal keeps the task non-trivial (baseline F1 below 1.0), so the
    # zero gap reflects genuine parity rather than a saturated ceiling.
    counts, labels = _structured_counts(400, 80, 4, seed=0, signal=3.0)
    comparison = gate1_annotation_comparison(
        counts, labels, n_classes=4, n_top_genes=40, n_components=15, n_train_steps=150
    )
    assert 0.3 < comparison["baseline_macro_f1"] < 1.0
    # Gate 1: the frozen pipeline reproduces the baseline macro-F1 within noise.
    assert comparison["macro_f1_gap"] < 0.02


def test_gate1_comparison_is_deterministic() -> None:
    counts, labels = _structured_counts(300, 60, 3, seed=1, signal=4.0)
    kwargs = {"n_classes": 3, "n_top_genes": 30, "n_components": 12, "n_train_steps": 100}
    first = gate1_annotation_comparison(counts, labels, **kwargs)
    second = gate1_annotation_comparison(counts, labels, **kwargs)
    assert first["pipeline_macro_f1"] == second["pipeline_macro_f1"]
    assert first["baseline_macro_f1"] == second["baseline_macro_f1"]


# --- Feature parity (transitive basis for the F1 parity) ------------------------


def test_pipeline_features_match_baseline_features_subspace() -> None:
    counts, _ = _structured_counts(300, 60, 3, seed=2, signal=5.0)
    pipeline_features = pipeline_frozen_features(
        counts, n_classes=3, n_top_genes=30, n_components=12
    )
    baseline_features = frozen_preprocess(counts, n_top_genes=30, n_components=12)
    assert pipeline_features.shape == baseline_features.shape
    assert _subspace_distance(pipeline_features, baseline_features) < 1.0e-3


def test_features_match_baseline_on_near_degenerate_spectrum() -> None:
    # Regression guard: when the top-k eigenvalues are nearly tied, exact and
    # randomized SVD diverge. The baseline must use the exact solver (matching the
    # differentiable operator), so the pipeline still reproduces its features here.
    counts, _, n_classes = _near_degenerate_counts(seed=0)
    pipeline_features = pipeline_frozen_features(
        counts, n_classes=n_classes, n_top_genes=200, n_components=25
    )
    baseline_features = frozen_preprocess(counts, n_top_genes=200, n_components=25)
    assert _subspace_distance(pipeline_features, baseline_features) < 1.0e-2


def test_pipeline_frozen_features_are_deterministic() -> None:
    counts, _ = _structured_counts(200, 50, 3, seed=3, signal=5.0)
    kwargs = {"n_classes": 3, "n_top_genes": 25, "n_components": 10}
    first = pipeline_frozen_features(counts, **kwargs)
    second = pipeline_frozen_features(counts, **kwargs)
    np.testing.assert_array_equal(first, second)


# --- Gate 1 on the real immune atlas (exercised when the data is present) --------


@pytest.mark.integration
@pytest.mark.skipif(not Path(_ATLAS_DATA_DIR).exists(), reason="immune_human atlas not available")
def test_gate1_on_immune_atlas() -> None:
    from diffbio.sources.immune_human import ImmuneHumanConfig, ImmuneHumanSource

    source = ImmuneHumanSource(ImmuneHumanConfig(data_dir=_ATLAS_DATA_DIR, subsample=2000))
    data = source.load()
    comparison = gate1_annotation_comparison(
        np.asarray(data["counts"], dtype=np.float32),
        np.asarray(data["cell_type_labels"], dtype=np.int32),
        n_classes=int(data["n_types"]),
        n_top_genes=2000,
        n_components=50,
        n_train_steps=100,
    )
    # Gate 1: the frozen pipeline reproduces the baseline macro-F1 within noise.
    assert comparison["macro_f1_gap"] < 0.05
