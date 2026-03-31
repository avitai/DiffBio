"""Tests for benchmarks.singlecell.bench_grn."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_grn import GRNBenchmark

_DATA_DIR = Path(
    "/media/mahdi/ssd23/Works/benGRN/data/GroundTruth/stone_and_sroy"
)
_SKIP = not (_DATA_DIR / "gold_standards" / "mESC").exists()


@pytest.mark.skipif(_SKIP, reason="benGRN data not available")
class TestGRNBenchmark:
    """Tests for the GRN inference benchmark."""

    @pytest.fixture()
    def result(self) -> BenchmarkResult:
        bench = GRNBenchmark(quick=True)
        return bench.run()

    def test_returns_benchmark_result(
        self, result: BenchmarkResult
    ) -> None:
        assert isinstance(result, BenchmarkResult)

    def test_name(self, result: BenchmarkResult) -> None:
        assert result.name == "singlecell/grn"

    def test_has_auprc(self, result: BenchmarkResult) -> None:
        assert "auprc" in result.metrics

    def test_has_precision(self, result: BenchmarkResult) -> None:
        assert "precision" in result.metrics

    def test_auprc_in_range(self, result: BenchmarkResult) -> None:
        auprc = result.metrics["auprc"].value
        assert 0.0 <= auprc <= 1.0

    def test_has_gradient_metrics(
        self, result: BenchmarkResult
    ) -> None:
        assert "gradient_norm" in result.metrics

    def test_has_timing(self, result: BenchmarkResult) -> None:
        assert result.timing is not None

    def test_has_baselines_in_metadata(
        self, result: BenchmarkResult
    ) -> None:
        assert "baselines" in result.metadata
