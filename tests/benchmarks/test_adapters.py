"""Tests for TaskAdapter dispatching to DiffBio operators.

Each test creates synthetic data in the operator-expected format and
verifies the adapter produces a grader-ready answer of the correct type.
Also validates calibrax quality metrics are computed correctly.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from diffbio.evaluation.adapters import TaskAdapter, compute_quality_metrics
from diffbio.evaluation.problem import BenchmarkProblem


@pytest.fixture()
def adapter() -> TaskAdapter:
    """Create a TaskAdapter with fixed seed."""
    return TaskAdapter(seed=42)


# ---------------------------------------------------------------------------
# Helpers to build operator-ready data dicts
# ---------------------------------------------------------------------------


def _make_embeddings(n_cells: int = 50, n_features: int = 10) -> dict[str, jnp.ndarray]:
    """Create a synthetic embeddings data dict."""
    raw = np.random.default_rng(42).standard_normal((n_cells, n_features))
    key = jnp.array(raw, dtype=jnp.float32)
    return {"embeddings": key}


def _make_counts(n_cells: int = 50, n_genes: int = 30) -> dict[str, jnp.ndarray]:
    """Create a synthetic counts data dict."""
    rng = np.random.default_rng(42)
    counts = jnp.array(rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32))
    library_size = jnp.sum(counts, axis=1, keepdims=True)
    return {"counts": counts, "library_size": library_size}


def _make_counts_with_design(
    n_cells: int = 50, n_genes: int = 30, n_conditions: int = 2
) -> dict[str, jnp.ndarray]:
    """Create counts with design matrix."""
    data = _make_counts(n_cells, n_genes)
    design = np.zeros((n_cells, n_conditions), dtype=np.float32)
    design[: n_cells // 2, 0] = 1.0
    design[n_cells // 2 :, 1] = 1.0
    data["design"] = jnp.array(design)
    return data


def _make_embeddings_with_batch(
    n_cells: int = 50, n_features: int = 10, n_batches: int = 2
) -> dict[str, jnp.ndarray]:
    """Create embeddings with batch labels."""
    data = _make_embeddings(n_cells, n_features)
    batch_labels = jnp.array([i % n_batches for i in range(n_cells)], dtype=jnp.int32)
    data["batch_labels"] = batch_labels
    return data


def _make_spatial_data(n_cells: int = 50, n_genes: int = 30) -> dict[str, jnp.ndarray]:
    """Create counts with spatial coordinates."""
    data = _make_counts(n_cells, n_genes)
    rng = np.random.default_rng(42)
    data["spatial_coords"] = jnp.array(rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32))
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTaskAdapterDispatch:
    """Test that TaskAdapter dispatches correctly and returns typed answers."""

    def test_unsupported_task_raises(self, adapter: TaskAdapter) -> None:
        """Unsupported task_type raises ValueError."""
        problem = BenchmarkProblem(
            problem_id="bad",
            task_type="nonexistent_task",
            grader_type="numeric_tolerance",
            expected_answer=0,
        )
        with pytest.raises(ValueError, match="Unsupported task_type"):
            adapter.solve(problem, {})

    def test_clustering_returns_dict(self, adapter: TaskAdapter) -> None:
        """Clustering adapter returns a dict distribution."""
        problem = BenchmarkProblem(
            problem_id="c1",
            task_type="clustering",
            grader_type="distribution_comparison",
            expected_answer={"0": 0.5, "1": 0.5},
            task_config={"n_clusters": 3},
        )
        data = _make_embeddings(50, 10)
        answer = adapter.solve(problem, data)
        assert isinstance(answer, dict)
        # All values should be non-negative floats summing to ~1
        assert all(isinstance(v, float) for v in answer.values())
        assert abs(sum(answer.values()) - 1.0) < 0.01

    def test_qc_filtering_returns_float(self, adapter: TaskAdapter) -> None:
        """QC filtering adapter returns a float cell count."""
        problem = BenchmarkProblem(
            problem_id="qc1",
            task_type="qc_filtering",
            grader_type="numeric_tolerance",
            expected_answer=40.0,
            task_config={"initial_threshold": 20.0},
        )
        data = _make_counts(50, 30)
        answer = adapter.solve(problem, data)
        assert isinstance(answer, float)
        assert answer >= 0

    def test_batch_correction_returns_float(self, adapter: TaskAdapter) -> None:
        """Batch correction adapter returns a float metric."""
        problem = BenchmarkProblem(
            problem_id="bc1",
            task_type="batch_correction",
            grader_type="numeric_tolerance",
            expected_answer=1.0,
            task_config={"n_clusters": 3},
        )
        data = _make_embeddings_with_batch(50, 10, 2)
        answer = adapter.solve(problem, data)
        assert isinstance(answer, float)

    def test_de_returns_gene_list(self, adapter: TaskAdapter) -> None:
        """Differential expression adapter returns a list of gene names."""
        problem = BenchmarkProblem(
            problem_id="de1",
            task_type="differential_expression",
            grader_type="marker_gene_precision_recall",
            expected_answer=["Gene_0", "Gene_1"],
            task_config={},
        )
        data = _make_counts_with_design(50, 30, 2)
        answer = adapter.solve(problem, data)
        assert isinstance(answer, list)
        assert all(isinstance(g, str) for g in answer)

    def test_normalization_returns_float(self, adapter: TaskAdapter) -> None:
        """Normalization adapter returns a float metric."""
        problem = BenchmarkProblem(
            problem_id="norm1",
            task_type="normalization",
            grader_type="numeric_tolerance",
            expected_answer=0.5,
            task_config={"latent_dim": 5},
        )
        data = _make_counts(50, 30)
        answer = adapter.solve(problem, data)
        assert isinstance(answer, float)

    def test_trajectory_returns_float(self, adapter: TaskAdapter) -> None:
        """Trajectory adapter returns a float pseudotime."""
        problem = BenchmarkProblem(
            problem_id="traj1",
            task_type="trajectory",
            grader_type="numeric_tolerance",
            expected_answer=1.0,
            task_config={"n_neighbors": 10, "n_diffusion_components": 5},
        )
        data = _make_embeddings(50, 10)
        answer = adapter.solve(problem, data)
        assert isinstance(answer, float)

    def test_spatial_analysis_returns_set(self, adapter: TaskAdapter) -> None:
        """Spatial analysis adapter returns a set of domain labels."""
        problem = BenchmarkProblem(
            problem_id="sp1",
            task_type="spatial_analysis",
            grader_type="label_set_jaccard",
            expected_answer=["0", "1", "2"],
            task_config={"n_domains": 3},
        )
        data = _make_spatial_data(50, 30)
        answer = adapter.solve(problem, data)
        assert isinstance(answer, set)

    def test_cell_annotation_returns_str(self, adapter: TaskAdapter) -> None:
        """Cell annotation adapter returns a string label."""
        problem = BenchmarkProblem(
            problem_id="ann1",
            task_type="cell_annotation",
            grader_type="multiple_choice",
            expected_answer="0",
            task_config={"n_clusters": 3},
        )
        data = _make_embeddings(50, 10)
        answer = adapter.solve(problem, data)
        assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# Calibrax quality metrics tests
# ---------------------------------------------------------------------------


class TestCalibraxQualityMetrics:
    """Test calibrax quality metrics computed alongside adapter results."""

    def test_clustering_metrics_computed(self, adapter: TaskAdapter) -> None:
        """Clustering produces silhouette and calinski-harabasz scores."""
        problem = BenchmarkProblem(
            problem_id="cm1",
            task_type="clustering",
            grader_type="distribution_comparison",
            expected_answer={"0": 0.5, "1": 0.5},
            task_config={"n_clusters": 3},
        )
        data = _make_embeddings(50, 10)
        answer, metrics = adapter.solve_with_metrics(problem, data)
        assert isinstance(answer, dict)
        assert "silhouette_score" in metrics
        assert "calinski_harabasz_score" in metrics
        assert -1.0 <= metrics["silhouette_score"] <= 1.0
        assert metrics["calinski_harabasz_score"] >= 0.0

    def test_batch_correction_metrics_computed(self, adapter: TaskAdapter) -> None:
        """Batch correction produces MMD and KL divergence."""
        problem = BenchmarkProblem(
            problem_id="bm1",
            task_type="batch_correction",
            grader_type="numeric_tolerance",
            expected_answer=1.0,
            task_config={"n_clusters": 3},
        )
        data = _make_embeddings_with_batch(50, 10, 2)
        answer, metrics = adapter.solve_with_metrics(problem, data)
        assert isinstance(answer, float)
        assert "mmd" in metrics
        assert "kl_divergence" in metrics
        assert metrics["mmd"] >= 0.0
        assert metrics["kl_divergence"] >= 0.0

    def test_unrelated_task_returns_empty_metrics(self, adapter: TaskAdapter) -> None:
        """Tasks without calibrax metrics return empty dict."""
        problem = BenchmarkProblem(
            problem_id="qm1",
            task_type="qc_filtering",
            grader_type="numeric_tolerance",
            expected_answer=40.0,
            task_config={"initial_threshold": 20.0},
        )
        data = _make_counts(50, 30)
        answer, metrics = adapter.solve_with_metrics(problem, data)
        assert isinstance(answer, float)
        assert metrics == {}

    def test_compute_quality_metrics_direct(self) -> None:
        """compute_quality_metrics works on raw operator output dicts."""
        rng = np.random.default_rng(42)
        embeddings = jnp.array(rng.standard_normal((50, 10)).astype(np.float32))
        labels = jnp.array([i % 3 for i in range(50)], dtype=jnp.int32)

        output = {"cluster_assignments": labels}
        data = {"embeddings": embeddings}

        metrics = compute_quality_metrics(output, "clustering", data)
        assert "silhouette_score" in metrics
        assert isinstance(metrics["silhouette_score"], float)
