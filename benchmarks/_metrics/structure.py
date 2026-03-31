"""RNA and protein structure prediction metrics.

Provides metrics for evaluating predicted base pair probability matrices
against known secondary structures in dot-bracket notation (DBN).

The primary metric is the F1 score over predicted base pairs, computed
from sensitivity (recall) and positive predictive value (precision).

DBN format:
    - ``(`` and ``)`` denote matched base pairs
    - ``.`` denotes unpaired positions
    - Nested parentheses represent nested helices
"""

from __future__ import annotations

import numpy as np


def parse_dbn_pairs(dbn: str) -> set[tuple[int, int]]:
    """Extract base pairs from a dot-bracket notation string.

    Pairs are returned as (i, j) tuples where i < j, matching
    the opening ``(`` at position i with the closing ``)`` at
    position j.

    Args:
        dbn: Dot-bracket notation string. Must contain only
            ``(``, ``)``, and ``.`` characters.

    Returns:
        Set of (i, j) base pair tuples with i < j.

    Raises:
        ValueError: If parentheses are unbalanced.
    """
    stack: list[int] = []
    pairs: set[tuple[int, int]] = set()
    for idx, char in enumerate(dbn):
        if char == "(":
            stack.append(idx)
        elif char == ")":
            if not stack:
                raise ValueError(f"Unbalanced ')' at position {idx} in DBN: {dbn!r}")
            opening = stack.pop()
            pairs.add((opening, idx))
        elif char != ".":
            raise ValueError(f"Invalid character {char!r} at position {idx} in DBN: {dbn!r}")
    if stack:
        raise ValueError(f"Unbalanced '(' at positions {stack} in DBN: {dbn!r}")
    return pairs


def base_pair_metrics(
    predicted_bp_probs: np.ndarray,
    true_structure_dbn: str,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute base pair prediction metrics.

    Thresholds the predicted base pair probability matrix to obtain
    predicted pairs, then computes precision (PPV), sensitivity
    (recall), and F1 score against the true structure.

    For the predicted matrix, a pair (i, j) is called if
    ``predicted_bp_probs[i, j] >= threshold`` and i < j (upper
    triangle only, to avoid double counting).

    Args:
        predicted_bp_probs: Base pair probability matrix of shape
            ``(length, length)``. Values should be in [0, 1].
        true_structure_dbn: Ground truth structure in dot-bracket
            notation.
        threshold: Probability threshold for calling a base pair.

    Returns:
        Dict with keys:
            - ``sensitivity``: TP / (TP + FN), recall of true pairs.
            - ``ppv``: TP / (TP + FP), positive predictive value.
            - ``f1``: Harmonic mean of sensitivity and PPV.
            - ``n_true_pairs``: Number of pairs in ground truth.
            - ``n_predicted_pairs``: Number of predicted pairs.
    """
    true_pairs = parse_dbn_pairs(true_structure_dbn)
    n_true = len(true_pairs)

    # Extract predicted pairs from upper triangle
    n = predicted_bp_probs.shape[0]
    predicted_pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if predicted_bp_probs[i, j] >= threshold:
                predicted_pairs.add((i, j))
    n_predicted = len(predicted_pairs)

    # True positives: pairs in both predicted and true
    tp = len(predicted_pairs & true_pairs)
    fn = n_true - tp
    fp = n_predicted - tp

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0.0

    return {
        "sensitivity": sensitivity,
        "ppv": ppv,
        "f1": f1,
        "n_true_pairs": n_true,
        "n_predicted_pairs": n_predicted,
    }
