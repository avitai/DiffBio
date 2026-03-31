"""Tests for benchmarks.singlecell.bench_grn."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_grn import GRNBenchmark
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Works/benGRN/data/GroundTruth/stone_and_sroy")
_SKIP = not (_DATA_DIR / "gold_standards" / "mESC").exists()


@pytest.mark.skipif(_SKIP, reason="benGRN data not available")
class TestGRNBenchmark:
    """Tests for the GRN inference benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        bench = GRNBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="singlecell/grn",
            required_metric_keys=["auprc", "precision"],
        )

    def test_auprc_in_range(self, result: BenchmarkResult) -> None:
        auprc = result.metrics["auprc"].value
        assert 0.0 <= auprc <= 1.0

    def test_has_baselines_in_metadata(self, result: BenchmarkResult) -> None:
        assert "baselines" in result.metadata
