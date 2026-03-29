"""End-to-end golden tests for the evaluation harness.

Exercises the full pipeline: BenchmarkProblem -> TaskAdapter -> Operator -> Grader.
Uses synthetic data and known answers to verify the entire evaluation path.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from diffbio.evaluation.adapters import TaskAdapter
from diffbio.evaluation.problem import BenchmarkProblem
from diffbio.evaluation.runner import EvalResult, run_benchmark, run_problem, summarize


@pytest.fixture()
def adapter() -> TaskAdapter:
    """Fixed-seed adapter for reproducible golden tests."""
    return TaskAdapter(seed=42)


def _make_embeddings(
    n_cells: int = 50, n_features: int = 10, seed: int = 42
) -> dict[str, jnp.ndarray]:
    """Build synthetic embeddings data dict."""
    rng = np.random.default_rng(seed)
    return {"embeddings": jnp.array(rng.standard_normal((n_cells, n_features)).astype(np.float32))}


def _make_counts(n_cells: int = 50, n_genes: int = 30, seed: int = 42) -> dict[str, jnp.ndarray]:
    """Build synthetic counts data dict with library size."""
    rng = np.random.default_rng(seed)
    counts = jnp.array(rng.poisson(5, (n_cells, n_genes)).astype(np.float32))
    return {"counts": counts, "library_size": jnp.sum(counts, axis=1, keepdims=True)}


class TestEndToEndGolden:
    """Golden tests exercising full problem -> adapter -> grader pipeline."""

    def test_clustering_distribution_e2e(self, adapter: TaskAdapter) -> None:
        """Clustering -> distribution comparison grading (lenient tolerance)."""
        problem = BenchmarkProblem(
            problem_id="golden_cluster",
            task_type="clustering",
            grader_type="distribution_comparison",
            expected_answer={"0": 0.33, "1": 0.33, "2": 0.34},
            grader_config={"tolerance": 0.5},
            task_config={"n_clusters": 3},
            source="synthetic",
        )
        data = _make_embeddings(60, 10)
        result = run_problem(problem, data, adapter=adapter)

        assert isinstance(result, EvalResult)
        assert result.problem_id == "golden_cluster"
        assert result.error == ""
        assert result.elapsed_seconds > 0
        # With lenient tolerance, untrained clustering should still produce
        # a valid distribution
        assert isinstance(result.predicted_answer, dict)
        assert abs(sum(result.predicted_answer.values()) - 1.0) < 0.01

    def test_qc_filtering_numeric_e2e(self, adapter: TaskAdapter) -> None:
        """QC filtering -> numeric tolerance grading."""
        problem = BenchmarkProblem(
            problem_id="golden_qc",
            task_type="qc_filtering",
            grader_type="numeric_tolerance",
            expected_answer=50.0,
            grader_config={"tolerance": 50.0, "mode": "absolute"},
            task_config={"initial_threshold": 20.0},
            source="synthetic",
        )
        data = _make_counts(50, 30)
        result = run_problem(problem, data, adapter=adapter)

        assert result.error == ""
        assert isinstance(result.predicted_answer, float)
        assert result.predicted_answer >= 0

    def test_cell_annotation_mc_e2e(self, adapter: TaskAdapter) -> None:
        """Cell annotation -> multiple choice grading."""
        problem = BenchmarkProblem(
            problem_id="golden_annot",
            task_type="cell_annotation",
            grader_type="multiple_choice",
            expected_answer="0",
            task_config={"n_clusters": 2},
            source="synthetic",
        )
        data = _make_embeddings(50, 10)
        result = run_problem(problem, data, adapter=adapter)

        assert result.error == ""
        assert isinstance(result.predicted_answer, str)

    def test_normalization_numeric_e2e(self, adapter: TaskAdapter) -> None:
        """Normalization -> numeric tolerance grading."""
        problem = BenchmarkProblem(
            problem_id="golden_norm",
            task_type="normalization",
            grader_type="numeric_tolerance",
            expected_answer=0.0,
            grader_config={"tolerance": 100.0, "mode": "absolute"},
            task_config={"latent_dim": 5},
            source="synthetic",
        )
        data = _make_counts(50, 30)
        result = run_problem(problem, data, adapter=adapter)

        assert result.error == ""
        assert isinstance(result.predicted_answer, float)

    def test_batch_execution_and_summary(self, adapter: TaskAdapter) -> None:
        """Run multiple problems and verify summary aggregation."""
        problems = [
            BenchmarkProblem(
                problem_id="batch_cluster",
                task_type="clustering",
                grader_type="distribution_comparison",
                expected_answer={"0": 0.5, "1": 0.5},
                grader_config={"tolerance": 0.5},
                task_config={"n_clusters": 2},
            ),
            BenchmarkProblem(
                problem_id="batch_annot",
                task_type="cell_annotation",
                grader_type="multiple_choice",
                expected_answer="0",
                task_config={"n_clusters": 2},
            ),
        ]
        data = _make_embeddings(50, 10)
        data_loader = {p.problem_id: data for p in problems}

        results = run_benchmark(problems, data_loader, adapter=adapter)
        assert len(results) == 2
        assert all(isinstance(r, EvalResult) for r in results)
        assert all(r.error == "" for r in results)

        summary = summarize(results)
        assert summary.total == 2
        assert summary.passed + summary.failed + summary.errored == 2
        assert 0.0 <= summary.pass_rate <= 1.0
        assert 0.0 <= summary.mean_score <= 1.0
