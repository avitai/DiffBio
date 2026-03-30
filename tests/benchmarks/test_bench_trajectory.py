"""Tests for benchmarks.singlecell.bench_trajectory."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_trajectory import TrajectoryBenchmark

_DATA_DIR = Path("/media/mahdi/ssd23/Data/scvelo")
_SKIP = not (_DATA_DIR / "endocrinogenesis_day15.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="Pancreas dataset not downloaded")
class TestTrajectoryBenchmark:
    """Tests for the trajectory benchmark."""

    @pytest.fixture()
    def result(self) -> BenchmarkResult:
        bench = TrajectoryBenchmark(quick=True)
        return bench.run()

    def test_returns_benchmark_result(
        self, result: BenchmarkResult
    ) -> None:
        assert isinstance(result, BenchmarkResult)

    def test_name(self, result: BenchmarkResult) -> None:
        assert result.name == "singlecell/trajectory"

    def test_has_pseudotime_metrics(
        self, result: BenchmarkResult
    ) -> None:
        assert "pseudotime_range" in result.metrics

    def test_has_velocity_metrics(
        self, result: BenchmarkResult
    ) -> None:
        assert "velocity_shape_correct" in result.metrics

    def test_has_gradient_metrics(
        self, result: BenchmarkResult
    ) -> None:
        assert "gradient_norm" in result.metrics

    def test_has_timing(self, result: BenchmarkResult) -> None:
        assert result.timing is not None

    def test_has_dataset_tag(self, result: BenchmarkResult) -> None:
        assert result.tags["dataset"] == "pancreas"
