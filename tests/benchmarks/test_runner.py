"""Tests for the evaluation runner (run_problem, run_benchmark, summarize)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from diffbio.evaluation.graders import GradeResult
from diffbio.evaluation.problem import BenchmarkProblem
from diffbio.evaluation.runner import (
    BenchmarkSummary,
    EvalResult,
    _grade_answer,
    run_benchmark,
    run_problem,
    summarize,
)


# ---------------------------------------------------------------------------
# _grade_answer
# ---------------------------------------------------------------------------


class TestGradeAnswer:
    """Tests for the internal grader dispatch."""

    def test_numeric_tolerance(self) -> None:
        """Dispatches to grade_numeric_tolerance."""
        problem = BenchmarkProblem(
            problem_id="t1",
            task_type="qc_filtering",
            grader_type="numeric_tolerance",
            expected_answer=42.0,
            grader_config={"tolerance": 5.0, "mode": "absolute"},
        )
        result = _grade_answer(43.0, problem)
        assert result.passed is True

    def test_multiple_choice(self) -> None:
        """Dispatches to grade_multiple_choice."""
        problem = BenchmarkProblem(
            problem_id="t2",
            task_type="cell_annotation",
            grader_type="multiple_choice",
            expected_answer="B",
        )
        result = _grade_answer("b", problem)
        assert result.passed is True

    def test_marker_gene(self) -> None:
        """Dispatches to grade_marker_gene_precision_recall."""
        problem = BenchmarkProblem(
            problem_id="t3",
            task_type="de",
            grader_type="marker_gene_precision_recall",
            expected_answer=["A", "B", "C"],
            grader_config={"k": 3},
        )
        result = _grade_answer(["A", "B", "X"], problem)
        assert 0.0 < result.score < 1.0

    def test_distribution_comparison(self) -> None:
        """Dispatches to grade_distribution_comparison."""
        problem = BenchmarkProblem(
            problem_id="t4",
            task_type="clustering",
            grader_type="distribution_comparison",
            expected_answer={"A": 0.5, "B": 0.5},
            grader_config={"tolerance": 0.1},
        )
        result = _grade_answer({"A": 0.5, "B": 0.5}, problem)
        assert result.passed is True

    def test_label_set_jaccard(self) -> None:
        """Dispatches to grade_label_set_jaccard."""
        problem = BenchmarkProblem(
            problem_id="t5",
            task_type="clustering",
            grader_type="label_set_jaccard",
            expected_answer=["A", "B"],
            grader_config={"threshold": 0.5},
        )
        result = _grade_answer({"A", "B"}, problem)
        assert result.passed is True

    def test_unknown_grader_raises(self) -> None:
        """Unknown grader_type raises ValueError."""
        problem = BenchmarkProblem(
            problem_id="bad",
            task_type="x",
            grader_type="nonexistent",
            expected_answer=0,
        )
        with pytest.raises(ValueError, match="Unknown grader_type"):
            _grade_answer(0, problem)


# ---------------------------------------------------------------------------
# run_problem
# ---------------------------------------------------------------------------


class TestRunProblem:
    """Tests for single-problem execution."""

    def test_successful_run(self) -> None:
        """Successful adapter run returns passing EvalResult."""
        problem = BenchmarkProblem(
            problem_id="ok1",
            task_type="clustering",
            grader_type="distribution_comparison",
            expected_answer={"0": 0.5, "1": 0.5},
            grader_config={"tolerance": 0.5},
        )
        mock_adapter = MagicMock()
        mock_adapter.solve_with_metrics.return_value = ({"0": 0.5, "1": 0.5}, {})

        result = run_problem(problem, {}, adapter=mock_adapter)
        assert isinstance(result, EvalResult)
        assert result.problem_id == "ok1"
        assert result.grade.passed is True
        assert result.error == ""

    def test_successful_run_without_metrics(self) -> None:
        """Successful run with collect_metrics=False uses solve()."""
        problem = BenchmarkProblem(
            problem_id="ok2",
            task_type="clustering",
            grader_type="distribution_comparison",
            expected_answer={"0": 0.5, "1": 0.5},
            grader_config={"tolerance": 0.5},
        )
        mock_adapter = MagicMock()
        mock_adapter.solve.return_value = {"0": 0.5, "1": 0.5}

        result = run_problem(problem, {}, adapter=mock_adapter, collect_metrics=False)
        assert result.grade.passed is True
        assert result.quality_metrics == {}

    def test_adapter_error_captured(self) -> None:
        """Adapter failure is captured as error in EvalResult."""
        problem = BenchmarkProblem(
            problem_id="fail1",
            task_type="clustering",
            grader_type="numeric_tolerance",
            expected_answer=0,
        )
        mock_adapter = MagicMock()
        mock_adapter.solve_with_metrics.side_effect = RuntimeError("boom")

        result = run_problem(problem, {}, adapter=mock_adapter)
        assert result.grade.passed is False
        assert "boom" in result.error

    def test_grading_error_captured(self) -> None:
        """Grading failure is captured as error in EvalResult."""
        problem = BenchmarkProblem(
            problem_id="gradefail1",
            task_type="x",
            grader_type="nonexistent_grader",
            expected_answer=0,
        )
        mock_adapter = MagicMock()
        mock_adapter.solve_with_metrics.return_value = (42, {})

        result = run_problem(problem, {}, adapter=mock_adapter)
        assert result.grade.passed is False
        assert result.error != ""

    def test_quality_metrics_propagated(self) -> None:
        """Quality metrics from solve_with_metrics appear in EvalResult."""
        problem = BenchmarkProblem(
            problem_id="qm1",
            task_type="clustering",
            grader_type="distribution_comparison",
            expected_answer={"0": 0.5},
            grader_config={"tolerance": 0.5},
        )
        mock_adapter = MagicMock()
        mock_adapter.solve_with_metrics.return_value = (
            {"0": 1.0},
            {"silhouette_score": 0.42},
        )

        result = run_problem(problem, {}, adapter=mock_adapter)
        assert result.quality_metrics == {"silhouette_score": 0.42}


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    """Tests for batch execution."""

    def test_runs_all_problems(self) -> None:
        """All problems are evaluated."""
        problems = [
            BenchmarkProblem(
                problem_id=f"p{i}",
                task_type="clustering",
                grader_type="distribution_comparison",
                expected_answer={"0": 0.5, "1": 0.5},
                grader_config={"tolerance": 0.5},
            )
            for i in range(3)
        ]
        mock_adapter = MagicMock()
        mock_adapter.solve_with_metrics.return_value = ({"0": 0.5, "1": 0.5}, {})

        loader = lambda prob: {}  # noqa: E731
        results = run_benchmark(problems, loader, adapter=mock_adapter)
        assert len(results) == 3

    def test_dict_data_loader(self) -> None:
        """Data loader can be a dict mapping problem_id to data."""
        problems = [
            BenchmarkProblem(
                problem_id="p1",
                task_type="clustering",
                grader_type="distribution_comparison",
                expected_answer={"0": 1.0},
                grader_config={"tolerance": 0.5},
            )
        ]
        mock_adapter = MagicMock()
        mock_adapter.solve_with_metrics.return_value = ({"0": 1.0}, {})

        data_map = {"p1": {"embeddings": "fake"}}
        results = run_benchmark(problems, data_map, adapter=mock_adapter)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    """Tests for result aggregation."""

    def test_empty_results(self) -> None:
        """Empty results produce zero summary."""
        summary = summarize([])
        assert summary.total == 0
        assert summary.pass_rate == 0.0

    def test_all_pass(self) -> None:
        """All passing results give 100% pass rate."""
        results = [
            EvalResult(
                problem_id=f"p{i}",
                task_type="clustering",
                grade=GradeResult(passed=True, score=1.0, detail="ok"),
            )
            for i in range(5)
        ]
        summary = summarize(results)
        assert summary.total == 5
        assert summary.passed == 5
        assert summary.pass_rate == 1.0
        assert summary.mean_score == 1.0

    def test_mixed_results(self) -> None:
        """Mixed results produce correct counts."""
        results = [
            EvalResult(
                problem_id="p1",
                task_type="clustering",
                grade=GradeResult(passed=True, score=1.0, detail="ok"),
            ),
            EvalResult(
                problem_id="p2",
                task_type="clustering",
                grade=GradeResult(passed=False, score=0.3, detail="fail"),
            ),
            EvalResult(
                problem_id="p3",
                task_type="de",
                grade=GradeResult(passed=False, score=0.0, detail="error"),
                error="boom",
            ),
        ]
        summary = summarize(results)
        assert summary.total == 3
        assert summary.passed == 1
        assert summary.errored == 1
        assert summary.failed == 1
        assert summary.pass_rate == pytest.approx(1 / 3)

    def test_by_task_type(self) -> None:
        """Per-task-type pass rates are computed."""
        results = [
            EvalResult(
                problem_id="p1",
                task_type="clustering",
                grade=GradeResult(passed=True, score=1.0, detail="ok"),
            ),
            EvalResult(
                problem_id="p2",
                task_type="clustering",
                grade=GradeResult(passed=False, score=0.0, detail="fail"),
            ),
            EvalResult(
                problem_id="p3",
                task_type="de",
                grade=GradeResult(passed=True, score=0.8, detail="ok"),
            ),
        ]
        summary = summarize(results)
        assert summary.by_task_type["clustering"] == pytest.approx(0.5)
        assert summary.by_task_type["de"] == pytest.approx(1.0)

    def test_summary_is_frozen(self) -> None:
        """BenchmarkSummary is immutable."""
        summary = BenchmarkSummary(
            total=1,
            passed=1,
            failed=0,
            errored=0,
            pass_rate=1.0,
            mean_score=1.0,
        )
        with pytest.raises(AttributeError):
            summary.total = 99  # type: ignore[misc]
