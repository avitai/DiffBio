"""Evaluation runner for executing benchmark problems and collecting results.

Provides ``run_problem`` for single-problem execution, ``run_benchmark``
for batch execution, and ``summarize`` for aggregating results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, cast

from diffbio.evaluation.adapters import TaskAdapter
from diffbio.evaluation.graders import (
    GradeResult,
    grade_distribution_comparison,
    grade_label_set_jaccard,
    grade_marker_gene_precision_recall,
    grade_multiple_choice,
    grade_numeric_tolerance,
)
from diffbio.evaluation.problem import BenchmarkProblem

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalResult:
    """Result of evaluating a single benchmark problem.

    Attributes:
        problem_id: Identifier of the evaluated problem.
        task_type: Task category of the problem.
        grade: The grading result from the appropriate grader.
        predicted_answer: The raw answer produced by the adapter.
        quality_metrics: Calibrax quality metrics on operator output
            (e.g., silhouette score for clustering, MMD for batch correction).
        elapsed_seconds: Wall-clock time for the solve step.
        error: Error message if the solve step failed, else empty string.
    """

    problem_id: str
    task_type: str
    grade: GradeResult
    predicted_answer: Any = None
    quality_metrics: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkSummary:
    """Aggregated summary of a benchmark run.

    Attributes:
        total: Total number of problems evaluated.
        passed: Number of problems that passed.
        failed: Number of problems that failed.
        errored: Number of problems that encountered errors.
        pass_rate: Fraction of problems that passed (0.0 to 1.0).
        mean_score: Mean score across all evaluated problems.
        by_task_type: Pass rate broken down by task_type.
    """

    total: int
    passed: int
    failed: int
    errored: int
    pass_rate: float
    mean_score: float
    by_task_type: dict[str, float] = field(default_factory=dict)


def _grade_answer(
    predicted: Any,
    problem: BenchmarkProblem,
) -> GradeResult:
    """Apply the appropriate grader to a predicted answer.

    Args:
        predicted: The answer produced by the TaskAdapter.
        problem: The benchmark problem (contains expected answer and grader config).

    Returns:
        GradeResult from the matching grader algorithm.

    Raises:
        ValueError: If grader_type is not recognised.
    """
    grader_type = problem.grader_type
    expected = problem.expected_answer
    config = problem.grader_config

    if grader_type == "numeric_tolerance":
        return grade_numeric_tolerance(
            float(predicted),
            float(expected),
            tolerance=config.get("tolerance", 0.1),
            mode=config.get("mode", "absolute"),
        )

    if grader_type == "multiple_choice":
        return grade_multiple_choice(str(predicted), str(expected))

    if grader_type == "marker_gene_precision_recall":
        return grade_marker_gene_precision_recall(
            list(predicted),
            list(expected),
            k=config.get("k"),
        )

    if grader_type == "distribution_comparison":
        return grade_distribution_comparison(
            dict(predicted),
            dict(expected),
            tolerance=config.get("tolerance", 0.1),
        )

    if grader_type == "label_set_jaccard":
        return grade_label_set_jaccard(
            set(predicted) if not isinstance(predicted, set) else predicted,
            set(expected) if not isinstance(expected, set) else expected,
            threshold=config.get("threshold", 0.5),
        )

    raise ValueError(
        f"Unknown grader_type {grader_type!r}. Supported: "
        "numeric_tolerance, multiple_choice, marker_gene_precision_recall, "
        "distribution_comparison, label_set_jaccard"
    )


def run_problem(
    problem: BenchmarkProblem,
    data_dict: dict[str, Any],
    *,
    adapter: TaskAdapter | None = None,
    collect_metrics: bool = True,
) -> EvalResult:
    """Run a single benchmark problem end-to-end.

    Args:
        problem: The benchmark problem to evaluate.
        data_dict: Operator-ready input data.
        adapter: Optional TaskAdapter instance. Created with defaults if None.
        collect_metrics: Whether to compute calibrax quality metrics.

    Returns:
        EvalResult with grading outcome and optional quality metrics.
    """
    if adapter is None:
        adapter = TaskAdapter()

    quality_metrics: dict[str, float] = {}
    try:
        t0 = time.monotonic()
        if collect_metrics:
            predicted, quality_metrics = adapter.solve_with_metrics(problem, data_dict)
        else:
            predicted = adapter.solve(problem, data_dict)
        elapsed = time.monotonic() - t0
    except Exception as exc:
        logger.warning("Problem %s failed: %s", problem.problem_id, exc)
        return EvalResult(
            problem_id=problem.problem_id,
            task_type=problem.task_type,
            grade=GradeResult(passed=False, score=0.0, detail=f"solve error: {exc}"),
            error=str(exc),
        )

    try:
        grade = _grade_answer(predicted, problem)
    except Exception as exc:
        logger.warning("Grading %s failed: %s", problem.problem_id, exc)
        return EvalResult(
            problem_id=problem.problem_id,
            task_type=problem.task_type,
            grade=GradeResult(passed=False, score=0.0, detail=f"grading error: {exc}"),
            predicted_answer=predicted,
            quality_metrics=quality_metrics,
            elapsed_seconds=elapsed,
            error=str(exc),
        )

    return EvalResult(
        problem_id=problem.problem_id,
        task_type=problem.task_type,
        grade=grade,
        predicted_answer=predicted,
        quality_metrics=quality_metrics,
        elapsed_seconds=elapsed,
    )


def run_benchmark(
    problems: list[BenchmarkProblem],
    data_loader: Any,
    *,
    adapter: TaskAdapter | None = None,
) -> list[EvalResult]:
    """Run a batch of benchmark problems.

    Args:
        problems: List of problems to evaluate.
        data_loader: Callable ``(problem) -> dict`` that provides operator-ready
            data for each problem. Can also be a dict mapping problem_id to data.
        adapter: Optional shared TaskAdapter.

    Returns:
        List of EvalResult, one per problem.
    """
    if adapter is None:
        adapter = TaskAdapter()

    results: list[EvalResult] = []
    for problem in problems:
        data_dict: dict[str, Any]
        if callable(data_loader):
            data_dict = data_loader(problem)
        else:
            data_dict = cast(dict[str, Any], data_loader[problem.problem_id])
        result = run_problem(problem, data_dict, adapter=adapter)
        results.append(result)
        logger.info(
            "Problem %s: %s (score=%.3f, %.2fs)",
            problem.problem_id,
            "PASS" if result.grade.passed else "FAIL",
            result.grade.score,
            result.elapsed_seconds,
        )

    return results


def summarize(results: list[EvalResult]) -> BenchmarkSummary:
    """Aggregate evaluation results into a summary.

    Args:
        results: List of EvalResult from a benchmark run.

    Returns:
        BenchmarkSummary with pass rates and score statistics.
    """
    if not results:
        return BenchmarkSummary(
            total=0,
            passed=0,
            failed=0,
            errored=0,
            pass_rate=0.0,
            mean_score=0.0,
        )

    passed = sum(1 for r in results if r.grade.passed)
    errored = sum(1 for r in results if r.error)
    failed = len(results) - passed - errored

    scores = [r.grade.score for r in results]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    # Per-task-type pass rate
    by_task: dict[str, list[bool]] = {}
    for r in results:
        by_task.setdefault(r.task_type, []).append(r.grade.passed)

    by_task_rate = {task: sum(passes) / len(passes) for task, passes in by_task.items()}

    return BenchmarkSummary(
        total=len(results),
        passed=passed,
        failed=failed,
        errored=errored,
        pass_rate=passed / len(results),
        mean_score=mean_score,
        by_task_type=by_task_rate,
    )
