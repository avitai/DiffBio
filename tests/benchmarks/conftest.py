"""Shared fixtures for benchmark evaluation tests.

Provides synthetic data generators and helper assertions for grading,
adapter, runner, and end-to-end tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from diffbio.evaluation.problem import BenchmarkProblem


# ---------------------------------------------------------------------------
# Synthetic AnnData fixture (requires anndata + pandas)
# ---------------------------------------------------------------------------


def _make_synthetic_adata(
    n_cells: int = 50,
    n_genes: int = 30,
    n_batches: int = 2,
    n_clusters: int = 3,
    *,
    include_spatial: bool = False,
    seed: int = 42,
) -> Any:
    """Create a minimal synthetic AnnData for testing.

    Returns:
        An anndata.AnnData object, or None if anndata is not installed.
    """
    try:
        import anndata as ad
        import pandas as pd
    except ImportError:
        pytest.skip("anndata/pandas not installed")

    rng = np.random.default_rng(seed)

    counts = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)

    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    batch_labels = np.array([f"batch_{i % n_batches}" for i in range(n_cells)])
    cluster_labels = np.array([f"cluster_{i % n_clusters}" for i in range(n_cells)])

    obs = pd.DataFrame(
        {"batch": batch_labels, "cell_type": cluster_labels},
        index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
    )
    var = pd.DataFrame(index=pd.Index(gene_names))

    adata = ad.AnnData(X=counts, obs=obs, var=var)

    # PCA embedding (random low-dim)
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, 10)).astype(np.float32)

    if include_spatial:
        adata.obsm["spatial"] = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)

    return adata


@pytest.fixture()
def synthetic_adata():
    """Fixture returning a minimal synthetic AnnData (50 cells, 30 genes)."""
    return _make_synthetic_adata()


@pytest.fixture()
def spatial_adata():
    """Fixture returning a synthetic AnnData with spatial coordinates."""
    return _make_synthetic_adata(include_spatial=True)


# ---------------------------------------------------------------------------
# Synthetic BenchmarkProblem fixtures
# ---------------------------------------------------------------------------


def make_numeric_problem(
    *,
    expected: float = 42.0,
    tolerance: float = 5.0,
    mode: str = "absolute",
    task_type: str = "qc_filtering",
    problem_id: str = "synth_numeric_001",
) -> BenchmarkProblem:
    """Create a synthetic numeric tolerance benchmark problem."""
    return BenchmarkProblem(
        problem_id=problem_id,
        task_type=task_type,
        grader_type="numeric_tolerance",
        expected_answer=expected,
        grader_config={"tolerance": tolerance, "mode": mode},
        description="Synthetic numeric tolerance problem",
        source="synthetic",
    )


def make_multiple_choice_problem(
    *,
    expected: str = "B",
    problem_id: str = "synth_mc_001",
) -> BenchmarkProblem:
    """Create a synthetic multiple-choice benchmark problem."""
    return BenchmarkProblem(
        problem_id=problem_id,
        task_type="cell_annotation",
        grader_type="multiple_choice",
        expected_answer=expected,
        description="Synthetic multiple-choice problem",
        source="synthetic",
    )


def make_marker_gene_problem(
    *,
    expected: list[str] | None = None,
    k: int = 5,
    problem_id: str = "synth_marker_001",
) -> BenchmarkProblem:
    """Create a synthetic marker gene benchmark problem."""
    if expected is None:
        expected = ["TP53", "BRCA1", "EGFR", "KRAS", "MYC"]
    return BenchmarkProblem(
        problem_id=problem_id,
        task_type="differential_expression",
        grader_type="marker_gene_precision_recall",
        expected_answer=expected,
        grader_config={"k": k},
        description="Synthetic marker gene problem",
        source="synthetic",
    )


def make_distribution_problem(
    *,
    expected: dict[str, float] | None = None,
    tolerance: float = 0.1,
    problem_id: str = "synth_dist_001",
) -> BenchmarkProblem:
    """Create a synthetic distribution comparison benchmark problem."""
    if expected is None:
        expected = {"T_cell": 0.4, "B_cell": 0.3, "Monocyte": 0.2, "NK": 0.1}
    return BenchmarkProblem(
        problem_id=problem_id,
        task_type="clustering",
        grader_type="distribution_comparison",
        expected_answer=expected,
        grader_config={"tolerance": tolerance},
        description="Synthetic distribution problem",
        source="synthetic",
    )


def make_jaccard_problem(
    *,
    expected: list[str] | None = None,
    threshold: float = 0.5,
    problem_id: str = "synth_jaccard_001",
) -> BenchmarkProblem:
    """Create a synthetic label set Jaccard benchmark problem."""
    if expected is None:
        expected = ["T_cell", "B_cell", "Monocyte", "NK"]
    return BenchmarkProblem(
        problem_id=problem_id,
        task_type="clustering",
        grader_type="label_set_jaccard",
        expected_answer=expected,
        grader_config={"threshold": threshold},
        description="Synthetic Jaccard problem",
        source="synthetic",
    )


@pytest.fixture()
def problems_json_path(tmp_path: Path) -> Path:
    """Write a small set of benchmark problems to a temp JSON file."""
    problems = [
        {
            "problem_id": "p1",
            "task_type": "qc_filtering",
            "grader_type": "numeric_tolerance",
            "expected_answer": 42.0,
            "grader_config": {"tolerance": 5.0, "mode": "absolute"},
        },
        {
            "problem_id": "p2",
            "task_type": "cell_annotation",
            "grader_type": "multiple_choice",
            "expected_answer": "B",
        },
    ]
    path = tmp_path / "problems.json"
    path.write_text(json.dumps(problems), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Shared benchmark result assertions
# ---------------------------------------------------------------------------


def assert_valid_benchmark_result(
    result: Any,
    *,
    expected_name: str | None = None,
    expected_domain: str = "diffbio_benchmarks",
    required_metric_keys: list[str] | None = None,
) -> None:
    """Assert that a benchmark result conforms to the standard contract.

    Checks that the result is a calibrax BenchmarkResult with the
    expected structure: correct type, domain, timing, gradient
    metrics, and all metric values are calibrax Metric objects.

    Args:
        result: The benchmark result to validate.
        expected_name: If set, verify the result name matches.
        expected_domain: Expected domain value.
        required_metric_keys: Extra metric keys that must be present.
    """
    from calibrax.core.models import Metric  # noqa: PLC0415
    from calibrax.core.result import BenchmarkResult  # noqa: PLC0415

    # Type check
    assert isinstance(result, BenchmarkResult), (
        f"Expected BenchmarkResult, got {type(result).__name__}"
    )

    # Name and domain
    if expected_name is not None:
        assert result.name == expected_name
    assert result.domain == expected_domain

    # Tags
    assert "operator" in result.tags
    assert "dataset" in result.tags
    assert result.tags.get("framework") == "diffbio"

    # Timing
    assert result.timing is not None
    assert result.timing.wall_clock_sec > 0

    # Gradient metrics (added by base class)
    assert "gradient_norm" in result.metrics
    assert "gradient_nonzero" in result.metrics

    # Throughput (added by base class)
    assert "items_per_sec" in result.metrics

    # All metrics are calibrax Metric objects
    for key, metric in result.metrics.items():
        assert isinstance(metric, Metric), (
            f"metrics[{key!r}] is {type(metric).__name__}, not Metric"
        )

    # Config has quick flag
    assert "quick" in result.config

    # Dataset info in metadata
    assert "dataset_info" in result.metadata

    # Extra required keys
    if required_metric_keys:
        for key in required_metric_keys:
            assert key in result.metrics, f"Missing required metric: {key}"
