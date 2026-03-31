"""Tests for benchmarks.singlecell.bench_trajectory."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_trajectory import TrajectoryBenchmark
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Data/scvelo")
_SKIP = not (_DATA_DIR / "endocrinogenesis_day15.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="Pancreas dataset not downloaded")
class TestTrajectoryBenchmark:
    """Tests for the trajectory benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        bench = TrajectoryBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="singlecell/trajectory",
            required_metric_keys=["pseudotime_range", "velocity_shape_correct"],
        )

    def test_has_dataset_tag(self, result: BenchmarkResult) -> None:
        assert result.tags["dataset"] == "pancreas"
