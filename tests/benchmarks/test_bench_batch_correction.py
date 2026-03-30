"""Tests for benchmarks.singlecell.bench_batch_correction.

TDD: These tests define the expected behavior of the first real
benchmark before implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_batch_correction import (
    BatchCorrectionBenchmark,
)

_DATA_DIR = Path("/media/mahdi/ssd23/Data/scib")
_SKIP = not (_DATA_DIR / "Immune_ALL_human.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="Dataset not downloaded")
class TestBatchCorrectionBenchmark:
    """Tests for the batch correction benchmark."""

    @pytest.fixture()
    def result(self) -> BenchmarkResult:
        """Run benchmark in quick mode (subsampled)."""
        bench = BatchCorrectionBenchmark(quick=True)
        return bench.run()

    def test_returns_benchmark_result(
        self, result: BenchmarkResult
    ) -> None:
        assert isinstance(result, BenchmarkResult)

    def test_name_is_correct(self, result: BenchmarkResult) -> None:
        assert result.name == "singlecell/batch_correction"

    def test_domain(self, result: BenchmarkResult) -> None:
        assert result.domain == "diffbio_benchmarks"

    def test_has_operator_tag(self, result: BenchmarkResult) -> None:
        assert "operator" in result.tags
        assert "DifferentiableHarmony" in result.tags["operator"]

    def test_has_dataset_tag(self, result: BenchmarkResult) -> None:
        assert "dataset" in result.tags
        assert result.tags["dataset"] == "immune_human"

    def test_has_quality_metrics(self, result: BenchmarkResult) -> None:
        assert "aggregate_score" in result.metrics
        assert "silhouette_label" in result.metrics
        assert "nmi_kmeans" in result.metrics

    def test_metrics_are_calibrax_metric(
        self, result: BenchmarkResult
    ) -> None:
        for key, metric in result.metrics.items():
            assert isinstance(metric, Metric), (
                f"{key} is not a Metric"
            )

    def test_aggregate_score_in_range(
        self, result: BenchmarkResult
    ) -> None:
        score = result.metrics["aggregate_score"].value
        assert 0.0 <= score <= 1.0

    def test_has_gradient_metrics(
        self, result: BenchmarkResult
    ) -> None:
        assert "gradient_norm" in result.metrics
        assert "gradient_nonzero" in result.metrics

    def test_has_timing(self, result: BenchmarkResult) -> None:
        assert result.timing is not None
        assert result.timing.wall_clock_sec > 0

    def test_has_config(self, result: BenchmarkResult) -> None:
        assert "n_clusters" in result.config

    def test_has_dataset_metadata(
        self, result: BenchmarkResult
    ) -> None:
        assert "dataset_info" in result.metadata
        info = result.metadata["dataset_info"]
        assert "n_cells" in info
        assert "n_batches" in info
