"""Unit tests for benchmark grading algorithms.

Tests all 5 grader functions from ``diffbio.evaluation.graders``:
- grade_numeric_tolerance (absolute, relative, min, max modes)
- grade_multiple_choice
- grade_marker_gene_precision_recall
- grade_distribution_comparison
- grade_label_set_jaccard
"""

from __future__ import annotations

import pytest

from diffbio.evaluation.graders import (
    GradeResult,
    grade_distribution_comparison,
    grade_label_set_jaccard,
    grade_marker_gene_precision_recall,
    grade_multiple_choice,
    grade_numeric_tolerance,
)


# =========================================================================
# GradeResult
# =========================================================================


class TestGradeResult:
    """Tests for the GradeResult dataclass."""

    def test_fields(self) -> None:
        """GradeResult stores passed, score, and detail."""
        result = GradeResult(passed=True, score=0.95, detail="ok")
        assert result.passed is True
        assert result.score == 0.95
        assert result.detail == "ok"

    def test_immutable(self) -> None:
        """GradeResult is frozen (immutable)."""
        result = GradeResult(passed=True, score=1.0, detail="ok")
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]


# =========================================================================
# grade_numeric_tolerance
# =========================================================================


class TestGradeNumericTolerance:
    """Tests for numeric tolerance grading."""

    def test_absolute_exact_match(self) -> None:
        """Exact match should pass with score 1.0."""
        result = grade_numeric_tolerance(42.0, 42.0, tolerance=1.0, mode="absolute")
        assert result.passed is True
        assert result.score == 1.0

    def test_absolute_within_tolerance(self) -> None:
        """Value within absolute tolerance should pass."""
        result = grade_numeric_tolerance(43.0, 42.0, tolerance=2.0, mode="absolute")
        assert result.passed is True
        assert result.score == pytest.approx(0.5)

    def test_absolute_outside_tolerance(self) -> None:
        """Value outside absolute tolerance should fail."""
        result = grade_numeric_tolerance(50.0, 42.0, tolerance=2.0, mode="absolute")
        assert result.passed is False
        assert result.score == 0.0

    def test_absolute_boundary(self) -> None:
        """Value exactly at tolerance boundary should pass."""
        result = grade_numeric_tolerance(44.0, 42.0, tolerance=2.0, mode="absolute")
        assert result.passed is True

    def test_relative_mode(self) -> None:
        """Relative mode should use proportional deviation."""
        # 10% relative tolerance, 5% actual deviation -> pass
        result = grade_numeric_tolerance(105.0, 100.0, tolerance=0.1, mode="relative")
        assert result.passed is True
        assert result.score == pytest.approx(0.5)

    def test_relative_mode_fail(self) -> None:
        """Large relative deviation should fail."""
        # 10% tolerance, 50% actual deviation -> fail
        result = grade_numeric_tolerance(150.0, 100.0, tolerance=0.1, mode="relative")
        assert result.passed is False

    def test_min_mode_pass(self) -> None:
        """Value above min threshold should pass."""
        result = grade_numeric_tolerance(45.0, 42.0, tolerance=5.0, mode="min")
        assert result.passed is True
        assert result.score == 1.0

    def test_min_mode_fail(self) -> None:
        """Value below min threshold should fail."""
        result = grade_numeric_tolerance(30.0, 42.0, tolerance=5.0, mode="min")
        assert result.passed is False

    def test_max_mode_pass(self) -> None:
        """Value below max threshold should pass."""
        result = grade_numeric_tolerance(40.0, 42.0, tolerance=5.0, mode="max")
        assert result.passed is True
        assert result.score == 1.0

    def test_max_mode_fail(self) -> None:
        """Value above max threshold should fail."""
        result = grade_numeric_tolerance(55.0, 42.0, tolerance=5.0, mode="max")
        assert result.passed is False

    def test_invalid_mode_raises(self) -> None:
        """Unknown mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown tolerance mode"):
            grade_numeric_tolerance(1.0, 1.0, mode="invalid")

    def test_score_clamped_to_unit_interval(self) -> None:
        """Score should always be in [0, 1]."""
        result = grade_numeric_tolerance(1000.0, 0.0, tolerance=1.0, mode="absolute")
        assert 0.0 <= result.score <= 1.0

    def test_default_mode_is_absolute(self) -> None:
        """Default mode is absolute."""
        result = grade_numeric_tolerance(42.5, 42.0, tolerance=1.0)
        assert result.passed is True
        assert "mode=absolute" in result.detail


# =========================================================================
# grade_multiple_choice
# =========================================================================


class TestGradeMultipleChoice:
    """Tests for multiple-choice grading."""

    def test_exact_match(self) -> None:
        """Identical strings should match."""
        result = grade_multiple_choice("B", "B")
        assert result.passed is True
        assert result.score == 1.0

    def test_case_insensitive(self) -> None:
        """Matching should be case-insensitive."""
        result = grade_multiple_choice("b", "B")
        assert result.passed is True

    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace should be ignored."""
        result = grade_multiple_choice("  B  ", "B")
        assert result.passed is True

    def test_mismatch(self) -> None:
        """Different answers should fail."""
        result = grade_multiple_choice("A", "B")
        assert result.passed is False
        assert result.score == 0.0

    def test_multi_word_answer(self) -> None:
        """Multi-word answers should be compared after strip+upper."""
        result = grade_multiple_choice("T Cell", "t cell")
        assert result.passed is True

    def test_empty_strings(self) -> None:
        """Empty strings should match."""
        result = grade_multiple_choice("", "")
        assert result.passed is True


# =========================================================================
# grade_marker_gene_precision_recall
# =========================================================================


class TestGradeMarkerGenePrecisionRecall:
    """Tests for marker gene precision/recall grading."""

    def test_perfect_match(self) -> None:
        """Exact same genes in same order -> F1=1.0."""
        genes = ["TP53", "BRCA1", "EGFR"]
        result = grade_marker_gene_precision_recall(genes, genes)
        assert result.passed is True
        assert result.score == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        """Some correct genes -> partial F1 score."""
        predicted = ["TP53", "BRCA1", "FAKE1"]
        truth = ["TP53", "BRCA1", "EGFR"]
        result = grade_marker_gene_precision_recall(predicted, truth)
        # P=2/3, R=2/3, F1=2/3
        assert result.score == pytest.approx(2 / 3, abs=1e-3)

    def test_no_overlap(self) -> None:
        """No shared genes -> score=0."""
        result = grade_marker_gene_precision_recall(["X", "Y", "Z"], ["A", "B", "C"])
        assert result.passed is False
        assert result.score == 0.0

    def test_custom_k(self) -> None:
        """K parameter limits evaluated predictions."""
        predicted = ["TP53", "BRCA1", "FAKE1", "FAKE2", "FAKE3"]
        truth = ["TP53", "BRCA1", "EGFR"]
        # Only top 2 evaluated: P@2=2/2=1.0, R@2=2/3
        result = grade_marker_gene_precision_recall(predicted, truth, k=2)
        expected_p = 1.0
        expected_r = 2 / 3
        expected_f1 = 2 * expected_p * expected_r / (expected_p + expected_r)
        assert result.score == pytest.approx(expected_f1, abs=1e-3)

    def test_empty_truth(self) -> None:
        """Empty truth with empty prediction should pass."""
        result = grade_marker_gene_precision_recall([], [])
        assert result.passed is True

    def test_empty_truth_nonempty_pred(self) -> None:
        """Empty truth with non-empty prediction should fail."""
        result = grade_marker_gene_precision_recall(["TP53"], [])
        assert result.passed is False

    def test_empty_prediction(self) -> None:
        """Empty prediction with non-empty truth -> score=0."""
        result = grade_marker_gene_precision_recall([], ["TP53", "BRCA1"])
        assert result.score == 0.0
        assert result.passed is False

    def test_f1_threshold(self) -> None:
        """Pass threshold is F1 >= 0.5."""
        # 2 out of 4 -> P=2/4=0.5, R=2/4=0.5, F1=0.5 -> just passes
        result = grade_marker_gene_precision_recall(["A", "B", "X", "Y"], ["A", "B", "C", "D"])
        assert result.passed is True
        assert result.score == pytest.approx(0.5)


# =========================================================================
# grade_distribution_comparison
# =========================================================================


class TestGradeDistributionComparison:
    """Tests for distribution comparison grading."""

    def test_exact_match(self) -> None:
        """Identical distributions -> score=1.0."""
        dist = {"A": 0.5, "B": 0.3, "C": 0.2}
        result = grade_distribution_comparison(dist, dist, tolerance=0.1)
        assert result.passed is True
        assert result.score == 1.0

    def test_within_tolerance(self) -> None:
        """All categories within tolerance -> pass."""
        pred = {"A": 0.48, "B": 0.32, "C": 0.20}
        truth = {"A": 0.5, "B": 0.3, "C": 0.2}
        result = grade_distribution_comparison(pred, truth, tolerance=0.05)
        assert result.passed is True

    def test_one_category_outside(self) -> None:
        """One category outside tolerance, rest within -> partial score."""
        pred = {"A": 0.8, "B": 0.3, "C": 0.2}
        truth = {"A": 0.5, "B": 0.3, "C": 0.2}
        result = grade_distribution_comparison(pred, truth, tolerance=0.1)
        # 2/3 within tolerance = 0.667 -> below 80% threshold
        assert result.score == pytest.approx(2 / 3, abs=1e-3)
        assert result.passed is False

    def test_missing_category_in_prediction(self) -> None:
        """Missing category in prediction defaults to 0.0."""
        pred = {"A": 0.5, "B": 0.3}
        truth = {"A": 0.5, "B": 0.3, "C": 0.05}
        result = grade_distribution_comparison(pred, truth, tolerance=0.1)
        # C: |0.0 - 0.05| = 0.05 <= 0.1 -> within tolerance
        assert result.passed is True

    def test_extra_category_in_prediction(self) -> None:
        """Extra category in prediction treated against truth=0."""
        pred = {"A": 0.5, "B": 0.3, "C": 0.2, "D": 0.05}
        truth = {"A": 0.5, "B": 0.3, "C": 0.2}
        result = grade_distribution_comparison(pred, truth, tolerance=0.1)
        # D: |0.05 - 0| = 0.05 <= 0.1 -> ok, 4/4 within
        assert result.passed is True

    def test_empty_truth(self) -> None:
        """Empty truth distribution should pass."""
        result = grade_distribution_comparison({"A": 1.0}, {})
        assert result.passed is True

    def test_detail_includes_violations(self) -> None:
        """Detail string should mention violated categories."""
        pred = {"A": 1.0}
        truth = {"A": 0.0}
        result = grade_distribution_comparison(pred, truth, tolerance=0.1)
        assert "A:" in result.detail


# =========================================================================
# grade_label_set_jaccard
# =========================================================================


class TestGradeLabelSetJaccard:
    """Tests for label set Jaccard grading."""

    def test_identical_sets(self) -> None:
        """Identical sets -> Jaccard=1.0."""
        labels = {"T_cell", "B_cell", "Monocyte"}
        result = grade_label_set_jaccard(labels, labels)
        assert result.passed is True
        assert result.score == 1.0

    def test_partial_overlap(self) -> None:
        """Partial overlap -> Jaccard = |A&B|/|A|B|."""
        pred = {"T_cell", "B_cell", "NK"}
        truth = {"T_cell", "B_cell", "Monocyte"}
        result = grade_label_set_jaccard(pred, truth)
        # |A&B|=2, |A|B|=4, J=0.5
        assert result.score == pytest.approx(0.5)
        assert result.passed is True  # threshold default 0.5

    def test_no_overlap(self) -> None:
        """No overlap -> Jaccard=0."""
        result = grade_label_set_jaccard({"X", "Y"}, {"A", "B"})
        assert result.score == 0.0
        assert result.passed is False

    def test_both_empty(self) -> None:
        """Both empty sets -> pass."""
        result = grade_label_set_jaccard(set(), set())
        assert result.passed is True
        assert result.score == 1.0

    def test_custom_threshold(self) -> None:
        """Custom threshold adjusts pass criteria."""
        pred = {"A", "B"}
        truth = {"A", "B", "C", "D"}
        # J = 2/4 = 0.5
        result_strict = grade_label_set_jaccard(pred, truth, threshold=0.7)
        assert result_strict.passed is False

        result_lax = grade_label_set_jaccard(pred, truth, threshold=0.3)
        assert result_lax.passed is True

    def test_superset_prediction(self) -> None:
        """Prediction is superset of truth."""
        pred = {"A", "B", "C", "D"}
        truth = {"A", "B"}
        # J = 2/4 = 0.5
        result = grade_label_set_jaccard(pred, truth, threshold=0.5)
        assert result.passed is True

    def test_detail_contains_counts(self) -> None:
        """Detail should include intersection and union sizes."""
        result = grade_label_set_jaccard({"A", "B"}, {"B", "C"})
        assert "|A&B|=" in result.detail
        assert "|A|B|=" in result.detail
