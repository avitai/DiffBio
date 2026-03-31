"""Tests for benchmarks.singlecell.bench_clustering."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_clustering import ClusteringBenchmark
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Data/scib")
_SKIP = not (_DATA_DIR / "Immune_ALL_human.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="Dataset not downloaded")
class TestClusteringBenchmark:
    """Tests for the clustering benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        bench = ClusteringBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="singlecell/clustering",
            required_metric_keys=["ari_kmeans", "nmi_kmeans"],
        )

    def test_ari_in_range(self, result: BenchmarkResult) -> None:
        ari = result.metrics["ari_kmeans"].value
        assert -1.0 <= ari <= 1.0

    def test_has_silhouette(self, result: BenchmarkResult) -> None:
        assert "silhouette_label" in result.metrics

    def test_has_comparison_in_metadata(self, result: BenchmarkResult) -> None:
        assert "baselines" in result.metadata
