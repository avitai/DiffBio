"""Tests for benchmarks.singlecell.bench_batch_correction."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_batch_correction import (
    BatchCorrectionBenchmark,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Data/scib")
_SKIP = not (_DATA_DIR / "Immune_ALL_human.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="Dataset not downloaded")
class TestBatchCorrectionBenchmark:
    """Tests for the batch correction benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        """Run benchmark once for the entire test class."""
        bench = BatchCorrectionBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="singlecell/batch_correction",
            required_metric_keys=[
                "aggregate_score",
                "silhouette_label",
                "nmi_kmeans",
            ],
        )

    def test_aggregate_score_in_range(self, result: BenchmarkResult) -> None:
        score = result.metrics["aggregate_score"].value
        assert 0.0 <= score <= 1.0

    def test_has_scib_bio_metrics(self, result: BenchmarkResult) -> None:
        for key in ["silhouette_label", "nmi_kmeans", "ari_kmeans"]:
            assert key in result.metrics

    def test_has_scib_batch_metrics(self, result: BenchmarkResult) -> None:
        for key in ["silhouette_batch", "ilisi"]:
            assert key in result.metrics

    def test_operator_tag(self, result: BenchmarkResult) -> None:
        assert "DifferentiableHarmony" in result.tags["operator"]

    def test_dataset_tag(self, result: BenchmarkResult) -> None:
        assert result.tags["dataset"] == "immune_human"
