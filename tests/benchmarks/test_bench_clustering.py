"""Tests for benchmarks.singlecell.bench_clustering."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_clustering import ClusteringBenchmark

_DATA_DIR = Path("/media/mahdi/ssd23/Data/scib")
_SKIP = not (_DATA_DIR / "Immune_ALL_human.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="Dataset not downloaded")
class TestClusteringBenchmark:
    """Tests for the clustering benchmark."""

    @pytest.fixture()
    def result(self) -> BenchmarkResult:
        bench = ClusteringBenchmark(quick=True)
        return bench.run()

    def test_returns_benchmark_result(
        self, result: BenchmarkResult
    ) -> None:
        assert isinstance(result, BenchmarkResult)

    def test_name(self, result: BenchmarkResult) -> None:
        assert result.name == "singlecell/clustering"

    def test_has_ari_and_nmi(self, result: BenchmarkResult) -> None:
        assert "ari_kmeans" in result.metrics
        assert "nmi_kmeans" in result.metrics

    def test_has_silhouette(self, result: BenchmarkResult) -> None:
        assert "silhouette_label" in result.metrics

    def test_ari_in_range(self, result: BenchmarkResult) -> None:
        ari = result.metrics["ari_kmeans"].value
        assert -1.0 <= ari <= 1.0

    def test_has_gradient_metrics(self, result: BenchmarkResult) -> None:
        assert "gradient_norm" in result.metrics

    def test_has_timing(self, result: BenchmarkResult) -> None:
        assert result.timing is not None

    def test_has_comparison_in_metadata(
        self, result: BenchmarkResult
    ) -> None:
        assert "baselines" in result.metadata
