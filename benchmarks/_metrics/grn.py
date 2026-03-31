"""GRN inference evaluation metrics.

Computes precision, recall, AUPRC, and early precision rank (EPR)
for gene regulatory network edge predictions against ground truth.

Reimplements the core logic from benGRN's compute_pr() to avoid
the full benGRN dependency while using the same evaluation protocol.

Reference:
    benGRN: https://github.com/your-org/benGRN
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_grn(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Evaluate a predicted GRN against binary ground truth.

    Computes precision-recall metrics by thresholding the predicted
    edge weights at various levels.

    Args:
        predicted: Weighted adjacency matrix (n_genes, n_genes).
            Higher values = more confident edge prediction.
        ground_truth: Binary adjacency matrix (n_genes, n_genes).
            1 = true edge, 0 = no edge.

    Returns:
        Dict with keys: auprc, precision, recall, random_precision,
        average_precision, epr (early precision rank).
    """
    pred = np.asarray(predicted, dtype=np.float64).ravel()
    truth = np.asarray(ground_truth, dtype=np.float64).ravel()

    if pred.shape != truth.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape} "
            f"vs ground_truth {ground_truth.shape}"
        )

    n_total = len(truth)
    n_positives = int(truth.sum())

    if n_positives == 0:
        logger.warning("No positive edges in ground truth")
        return {
            "auprc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "random_precision": 0.0,
            "average_precision": 0.0,
            "epr": 0.0,
        }

    random_precision = n_positives / n_total

    # Sort predictions descending
    sorted_idx = np.argsort(-pred)
    truth_sorted = truth[sorted_idx]

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(truth_sorted)
    n_predicted = np.arange(1, n_total + 1, dtype=np.float64)

    precision_curve = tp_cumsum / n_predicted
    recall_curve = tp_cumsum / n_positives

    # AUPRC via trapezoidal integration
    # Only at points where recall changes (true positives found)
    tp_mask = truth_sorted > 0
    if tp_mask.sum() > 0:
        average_precision = float(
            np.sum(precision_curve[tp_mask]) / n_positives
        )
    else:
        average_precision = 0.0

    # Simple AUPRC via trapezoid on the full curve
    auprc = float(np.trapezoid(precision_curve, recall_curve))

    # Overall precision/recall at default (all nonzero predictions)
    nonzero_mask = pred[sorted_idx] > 0
    n_nonzero = int(nonzero_mask.sum())
    if n_nonzero > 0:
        tp_at_default = float(tp_cumsum[n_nonzero - 1])
        precision = tp_at_default / n_nonzero
        recall = tp_at_default / n_positives
    else:
        precision = 0.0
        recall = 0.0

    # Early Precision Rank (EPR)
    # How much better than random is the top-K ranking?
    k = min(n_positives, n_total)
    tp_at_k = float(tp_cumsum[k - 1]) if k > 0 else 0.0
    expected_random = k * random_precision
    epr = (tp_at_k / expected_random) if expected_random > 0 else 0.0

    return {
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "random_precision": random_precision,
        "average_precision": average_precision,
        "epr": epr,
    }
