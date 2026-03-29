"""Grader algorithms for benchmark evaluation.

Implements 5 pure-Python grading algorithms used to compare DiffBio operator
outputs against ground-truth answers from scBench and SpatialBench problems.

Each grader returns a ``GradeResult`` indicating pass/fail with a numeric
score and optional detail message.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class GradeResult:
    """Result of grading a single benchmark problem.

    Attributes:
        passed: Whether the prediction met the acceptance criteria.
        score: Numeric score in [0, 1] indicating quality of the prediction.
        detail: Human-readable explanation of the grading outcome.
    """

    passed: bool
    score: float
    detail: str


def grade_numeric_tolerance(
    predicted: float,
    truth: float,
    *,
    tolerance: float = 0.1,
    mode: str = "absolute",
) -> GradeResult:
    """Grade a numeric prediction against ground truth with tolerance.

    Args:
        predicted: The predicted numeric value.
        truth: The ground-truth numeric value.
        tolerance: Acceptable deviation threshold.
        mode: Tolerance mode — one of:
            - ``"absolute"``: ``|predicted - truth| <= tolerance``
            - ``"relative"``: ``|predicted - truth| / max(|truth|, 1e-8) <= tolerance``
            - ``"min"``: ``predicted >= truth - tolerance``
            - ``"max"``: ``predicted <= truth + tolerance``

    Returns:
        GradeResult with pass/fail and score.

    Raises:
        ValueError: If mode is not one of the supported modes.
    """
    valid_modes = {"absolute", "relative", "min", "max"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown tolerance mode {mode!r}, expected one of {valid_modes}")

    diff = abs(predicted - truth)

    if mode == "absolute":
        passed = diff <= tolerance
        score = max(0.0, 1.0 - diff / max(tolerance, 1e-12))
    elif mode == "relative":
        rel_diff = diff / max(abs(truth), 1e-8)
        passed = rel_diff <= tolerance
        score = max(0.0, 1.0 - rel_diff / max(tolerance, 1e-12))
    elif mode == "min":
        passed = predicted >= truth - tolerance
        overshoot = (truth - tolerance - predicted) / max(tolerance, 1e-12)
        score = 1.0 if passed else max(0.0, 1.0 - overshoot)
    else:  # mode == "max"
        passed = predicted <= truth + tolerance
        overshoot = (predicted - truth - tolerance) / max(tolerance, 1e-12)
        score = 1.0 if passed else max(0.0, 1.0 - overshoot)

    return GradeResult(
        passed=passed,
        score=min(1.0, max(0.0, score)),
        detail=f"pred={predicted}, truth={truth}, diff={diff:.6g}, mode={mode}, tol={tolerance}",
    )


def grade_multiple_choice(predicted: str, truth: str) -> GradeResult:
    """Grade a multiple-choice answer by case-insensitive string equality.

    Args:
        predicted: The predicted answer string.
        truth: The ground-truth answer string.

    Returns:
        GradeResult with exact match result.
    """
    match = predicted.strip().upper() == truth.strip().upper()
    return GradeResult(
        passed=match,
        score=1.0 if match else 0.0,
        detail=f"pred={predicted.strip()!r}, truth={truth.strip()!r}",
    )


def grade_marker_gene_precision_recall(
    predicted: list[str],
    truth: list[str],
    *,
    k: int | None = None,
) -> GradeResult:
    """Grade predicted marker genes by precision@K and recall@K.

    Computes precision and recall of the top-K predicted genes against
    the ground-truth gene list. The final score is the F1 harmonic mean.

    Args:
        predicted: Ordered list of predicted marker gene names.
        truth: Ground-truth list of marker gene names.
        k: Number of top predictions to evaluate. Defaults to ``len(truth)``.

    Returns:
        GradeResult with F1 score and precision/recall detail.
    """
    if not truth:
        passed = len(predicted) == 0
        return GradeResult(passed=passed, score=1.0 if passed else 0.0, detail="empty truth set")

    if k is None:
        k = len(truth)

    top_k = predicted[:k]
    truth_set = set(truth)
    hits = sum(1 for gene in top_k if gene in truth_set)

    precision = hits / len(top_k) if top_k else 0.0
    recall = hits / len(truth_set)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return GradeResult(
        passed=f1 >= 0.5,
        score=f1,
        detail=f"P@{k}={precision:.3f}, R@{k}={recall:.3f}, F1={f1:.3f}, hits={hits}/{len(top_k)}",
    )


def grade_distribution_comparison(
    predicted: dict[str, float],
    truth: dict[str, float],
    *,
    tolerance: float = 0.1,
) -> GradeResult:
    """Grade predicted distribution against truth by per-category absolute diff.

    For each category in the ground truth, checks that
    ``|predicted[cat] - truth[cat]| <= tolerance``. Missing categories in the
    prediction are treated as zero.

    Args:
        predicted: Predicted distribution mapping category names to values.
        truth: Ground-truth distribution mapping category names to values.
        tolerance: Maximum absolute deviation per category.

    Returns:
        GradeResult with proportion of categories within tolerance.
    """
    if not truth:
        return GradeResult(passed=True, score=1.0, detail="empty truth distribution")

    within_tol = 0
    details: list[str] = []
    all_keys = set(truth) | set(predicted)

    for cat in sorted(all_keys):
        pred_val = predicted.get(cat, 0.0)
        truth_val = truth.get(cat, 0.0)
        diff = abs(pred_val - truth_val)
        ok = diff <= tolerance
        if ok:
            within_tol += 1
        else:
            details.append(f"{cat}: |{pred_val:.3f}-{truth_val:.3f}|={diff:.3f}>{tolerance}")

    score = within_tol / len(all_keys)
    passed = score >= 0.8  # At least 80% of categories within tolerance

    detail_str = f"{within_tol}/{len(all_keys)} categories within tol={tolerance}"
    if details:
        detail_str += f"; violations: {', '.join(details[:3])}"

    return GradeResult(passed=passed, score=score, detail=detail_str)


def grade_label_set_jaccard(
    predicted: set[str],
    truth: set[str],
    *,
    threshold: float = 0.5,
) -> GradeResult:
    """Grade predicted label set by Jaccard similarity.

    Computes ``|A & B| / |A | B|`` and checks if it meets the threshold.

    Args:
        predicted: Predicted set of labels.
        truth: Ground-truth set of labels.
        threshold: Minimum Jaccard index to pass.

    Returns:
        GradeResult with Jaccard score.
    """
    if not truth and not predicted:
        return GradeResult(passed=True, score=1.0, detail="both sets empty")

    intersection = len(predicted & truth)
    union = len(predicted | truth)
    jaccard = intersection / union if union > 0 else 0.0

    return GradeResult(
        passed=jaccard >= threshold,
        score=jaccard,
        detail=f"Jaccard={jaccard:.3f}, |A&B|={intersection}, |A|B|={union}, threshold={threshold}",
    )
