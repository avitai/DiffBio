"""Tests for benchmarks.multiomics.bench_spatial_deconv."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.multiomics.bench_spatial_deconv import (
    SpatialDeconvBenchmark,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Data/spatial")
_SKIP = not (_DATA_DIR / "seqfish_cortex.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="seqFISH dataset not downloaded")
class TestSpatialDeconvBenchmark:
    """Tests for the spatial deconvolution benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        """Run benchmark once for the entire test class."""
        bench = SpatialDeconvBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="multiomics/spatial_deconvolution",
            required_metric_keys=[
                "pearson_correlation",
                "rmse",
                "proportion_sum_to_one",
            ],
        )

    def test_pearson_in_range(self, result: BenchmarkResult) -> None:
        """Pearson correlation should be between -1 and 1."""
        value = result.metrics["pearson_correlation"].value
        assert -1.0 <= value <= 1.0

    def test_rmse_non_negative(self, result: BenchmarkResult) -> None:
        """RMSE must be non-negative."""
        value = result.metrics["rmse"].value
        assert value >= 0.0

    def test_proportion_sum_close_to_one(self, result: BenchmarkResult) -> None:
        """Softmax proportions should sum to ~1.0 per spot."""
        value = result.metrics["proportion_sum_to_one"].value
        assert value > 0.99

    def test_operator_tag(self, result: BenchmarkResult) -> None:
        """Operator tag must identify SpatialDeconvolution."""
        assert "SpatialDeconvolution" in result.tags["operator"]

    def test_dataset_tag(self, result: BenchmarkResult) -> None:
        """Dataset tag must reference seqFISH cortex."""
        assert result.tags["dataset"] == "seqfish_cortex"

    def test_has_spot_count_in_metadata(self, result: BenchmarkResult) -> None:
        """Metadata should record the number of spots."""
        info = result.metadata["dataset_info"]
        assert info["n_spots"] > 0

    def test_has_reference_spatial_split(self, result: BenchmarkResult) -> None:
        """Metadata should record reference/spatial split."""
        info = result.metadata["dataset_info"]
        assert info["n_reference"] > 0
        assert info["n_spatial"] > 0
        assert info["n_reference"] > info["n_spatial"]
