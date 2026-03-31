"""Tests for benchmarks._metrics.grn module.

Verifies evaluate_grn precision-recall metrics against known inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks._metrics.grn import evaluate_grn

_EXPECTED_KEYS = frozenset(
    {
        "auprc",
        "precision",
        "recall",
        "random_precision",
        "average_precision",
        "epr",
    }
)


class TestEvaluateGRN:
    """Tests for evaluate_grn metric computation."""

    def test_return_keys(self) -> None:
        """Result dict contains all expected metric keys."""
        pred = np.array([[0.8, 0.1], [0.2, 0.9]])
        truth = np.array([[1, 0], [0, 1]])
        result = evaluate_grn(pred, truth)
        assert set(result.keys()) == _EXPECTED_KEYS

    def test_all_values_are_float(self) -> None:
        """All metric values are Python floats."""
        pred = np.array([[0.8, 0.1], [0.2, 0.9]])
        truth = np.array([[1, 0], [0, 1]])
        result = evaluate_grn(pred, truth)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is {type(value)}, expected float"

    def test_perfect_prediction_high_auprc(self) -> None:
        """Perfect prediction yields high AUPRC and average_precision."""
        n = 10
        truth = np.zeros((n, n))
        truth[0, 1] = 1
        truth[2, 3] = 1
        truth[4, 5] = 1

        # Predicted scores: true edges get high weight, others low
        pred = np.zeros((n, n))
        pred[0, 1] = 0.95
        pred[2, 3] = 0.90
        pred[4, 5] = 0.85

        result = evaluate_grn(pred, truth)
        # average_precision should be perfect (1.0) for this ranking
        assert result["average_precision"] == 1.0
        # AUPRC via trapezoid is lower due to integration method
        assert result["auprc"] > 0.5
        assert result["epr"] > 1.0  # Better than random

    def test_random_prediction_low_auprc(self) -> None:
        """Random prediction yields AUPRC near random_precision."""
        rng = np.random.default_rng(42)
        n = 50
        truth = np.zeros((n, n))
        # Sparse ground truth (5% positive)
        pos_indices = rng.choice(n * n, size=int(0.05 * n * n), replace=False)
        truth.ravel()[pos_indices] = 1

        pred = rng.uniform(size=(n, n))
        result = evaluate_grn(pred, truth)
        # AUPRC should be close to random precision (not high)
        assert result["auprc"] < 0.3

    def test_no_positives_returns_zeros(self) -> None:
        """All-zero ground truth returns all-zero metrics."""
        pred = np.array([[0.5, 0.3], [0.1, 0.7]])
        truth = np.zeros((2, 2))
        result = evaluate_grn(pred, truth)
        assert result["auprc"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["random_precision"] == 0.0
        assert result["average_precision"] == 0.0
        assert result["epr"] == 0.0

    def test_shape_mismatch_raises_value_error(self) -> None:
        """Mismatched shapes raise ValueError."""
        pred = np.ones((3, 3))
        truth = np.ones((2, 2))
        with pytest.raises(ValueError, match="Shape mismatch"):
            evaluate_grn(pred, truth)

    def test_random_precision_correct(self) -> None:
        """random_precision equals n_positives / n_total."""
        truth = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        pred = np.ones_like(truth, dtype=np.float64)
        result = evaluate_grn(pred, truth)
        expected_random = 2.0 / 9.0
        assert abs(result["random_precision"] - expected_random) < 1e-10

    def test_recall_one_when_all_positives_predicted(self) -> None:
        """Recall is 1.0 when all positives have nonzero predictions."""
        truth = np.array([[1, 0], [0, 1]])
        pred = np.array([[1.0, 0.5], [0.3, 0.9]])
        result = evaluate_grn(pred, truth)
        assert result["recall"] == 1.0

    def test_epr_above_one_for_good_ranking(self) -> None:
        """EPR > 1 when top predictions are enriched for true edges."""
        n = 20
        truth = np.zeros((n, n))
        # Place positives at known locations
        truth[0, 1] = 1
        truth[1, 2] = 1
        # Predicted: high score exactly at the positives
        pred = np.full((n, n), 0.01)
        pred[0, 1] = 0.99
        pred[1, 2] = 0.98
        result = evaluate_grn(pred, truth)
        assert result["epr"] > 1.0
