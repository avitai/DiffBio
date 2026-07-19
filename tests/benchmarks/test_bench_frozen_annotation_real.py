"""Real-data smoke tests for the frozen annotation baseline benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_frozen_annotation import (
    FrozenAnnotationBaselineBenchmark,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Data/scib")
_SKIP = not (_DATA_DIR / "Immune_ALL_human.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="Dataset not downloaded")
class TestFrozenAnnotationBaselineBenchmarkRealData:
    """Smoke tests for the quick real-data frozen annotation path."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        bench = FrozenAnnotationBaselineBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        assert_valid_benchmark_result(
            result,
            expected_name="singlecell/frozen_annotation",
            required_metric_keys=["macro_f1", "balanced_accuracy"],
        )

    def test_reports_immune_human_dataset(self, result: BenchmarkResult) -> None:
        assert result.tags["task"] == "cell_annotation"
        assert result.tags["dataset"] == "immune_human"
