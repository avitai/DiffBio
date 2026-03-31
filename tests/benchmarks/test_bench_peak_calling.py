"""Tests for benchmarks.epigenomics.bench_peak_calling.

Validates the peak calling benchmark, coverage signal generation,
and peak metric computation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.epigenomics.bench_peak_calling import (
    PeakCallingBenchmark,
    build_coverage_signal,
    compute_peak_metrics,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_PATH = Path("/media/mahdi/ssd23/Data/encode/CTCF_K562_narrowPeak.bed.gz")
_SKIP = not _DATA_PATH.exists()


# -------------------------------------------------------------------
# Unit tests: compute_peak_metrics
# -------------------------------------------------------------------


class TestComputePeakMetrics:
    """Tests for the peak metric computation helper."""

    def test_perfect_prediction(self) -> None:
        """Perfect prediction gives F1 = 1.0."""
        truth = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.float32)
        predicted = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.float32)
        result = compute_peak_metrics(predicted, truth)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["jaccard"] == 1.0

    def test_no_predictions(self) -> None:
        """No predictions gives F1 = 0.0."""
        truth = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.float32)
        predicted = np.zeros(7, dtype=np.float32)
        result = compute_peak_metrics(predicted, truth)
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_all_wrong_predictions(self) -> None:
        """Predictions at wrong positions give precision = 0.0."""
        truth = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
        predicted = np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.float32)
        result = compute_peak_metrics(predicted, truth)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap gives intermediate metrics."""
        truth = np.array([0, 1, 1, 1, 1, 0, 0], dtype=np.float32)
        predicted = np.array([0, 0, 1, 1, 1, 1, 0], dtype=np.float32)
        result = compute_peak_metrics(predicted, truth)
        # TP=3, FP=1, FN=1 -> P=3/4, R=3/4, F1=3/4
        np.testing.assert_allclose(result["precision"], 0.75)
        np.testing.assert_allclose(result["recall"], 0.75)
        np.testing.assert_allclose(result["f1"], 0.75)

    def test_jaccard_computation(self) -> None:
        """Jaccard index computed correctly."""
        truth = np.array([0, 1, 1, 1, 0, 0], dtype=np.float32)
        predicted = np.array([0, 0, 1, 1, 1, 0], dtype=np.float32)
        # TP=2, union=4 -> Jaccard=2/4=0.5
        result = compute_peak_metrics(predicted, truth)
        np.testing.assert_allclose(result["jaccard"], 0.5)

    def test_empty_truth_and_prediction(self) -> None:
        """No truth peaks and no predictions gives all zeros."""
        truth = np.zeros(10, dtype=np.float32)
        predicted = np.zeros(10, dtype=np.float32)
        result = compute_peak_metrics(predicted, truth)
        assert result["f1"] == 0.0
        assert result["jaccard"] == 0.0


# -------------------------------------------------------------------
# Unit tests: build_coverage_signal
# -------------------------------------------------------------------


class TestBuildCoverageSignal:
    """Tests for the coverage signal generation helper."""

    def test_output_shapes(self) -> None:
        """Coverage and truth mask have correct length."""
        starts = np.array([100, 300])
        ends = np.array([200, 400])
        signals = np.array([10.0, 20.0])
        coverage, truth = build_coverage_signal(
            starts,
            ends,
            signals,
            region_start=0,
            region_end=500,
        )
        assert coverage.shape == (500,)
        assert truth.shape == (500,)

    def test_truth_mask_marks_peaks(self) -> None:
        """Truth mask is 1 inside peak regions, 0 outside."""
        starts = np.array([100, 300])
        ends = np.array([200, 400])
        signals = np.array([10.0, 10.0])
        _, truth = build_coverage_signal(
            starts,
            ends,
            signals,
            region_start=0,
            region_end=500,
            noise_std=0.0,
        )
        # Inside first peak
        assert truth[150] == 1.0
        # Outside peaks
        assert truth[50] == 0.0
        assert truth[250] == 0.0

    def test_coverage_elevated_at_peaks(self) -> None:
        """Coverage signal is higher at peak centers than background."""
        starts = np.array([200])
        ends = np.array([300])
        signals = np.array([50.0])
        coverage, _ = build_coverage_signal(
            starts,
            ends,
            signals,
            region_start=0,
            region_end=500,
            noise_std=0.0,
        )
        peak_center = coverage[250]
        background = coverage[0]
        assert peak_center > background + 10.0

    def test_peaks_outside_region_skipped(self) -> None:
        """Peaks outside the region do not contribute signal."""
        starts = np.array([600])
        ends = np.array([700])
        signals = np.array([100.0])
        coverage, truth = build_coverage_signal(
            starts,
            ends,
            signals,
            region_start=0,
            region_end=500,
            noise_std=0.0,
        )
        assert np.sum(truth) == 0.0
        # Only noise-free background remains
        np.testing.assert_allclose(coverage, 0.0)

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces identical coverage arrays."""
        args = (
            np.array([100]),
            np.array([200]),
            np.array([10.0]),
        )
        c1, _ = build_coverage_signal(*args, region_start=0, region_end=300, seed=99)
        c2, _ = build_coverage_signal(*args, region_start=0, region_end=300, seed=99)
        np.testing.assert_array_equal(c1, c2)


# -------------------------------------------------------------------
# Unit tests: ENCODEPeakSource (requires data)
# -------------------------------------------------------------------


@pytest.mark.skipif(_SKIP, reason="ENCODE data not available")
class TestENCODEPeakSource:
    """Tests for loading real ENCODE narrowPeak data."""

    def test_load_chr22(self) -> None:
        """Loading chr22 returns a non-empty dataset."""
        from diffbio.sources.encode_peaks import (
            ENCODEPeakConfig,
            ENCODEPeakSource,
        )

        config = ENCODEPeakConfig(chromosome="chr22", max_peaks=100)
        source = ENCODEPeakSource(config)
        data = source.load()
        assert data["n_peaks"] == 100
        assert len(data["starts"]) == 100
        assert len(data["ends"]) == 100
        assert all(p.chromosome == "chr22" for p in data["peaks"])

    def test_starts_before_ends(self) -> None:
        """All peak starts are before their ends."""
        from diffbio.sources.encode_peaks import (
            ENCODEPeakConfig,
            ENCODEPeakSource,
        )

        config = ENCODEPeakConfig(chromosome="chr22", max_peaks=50)
        source = ENCODEPeakSource(config)
        data = source.load()
        assert np.all(data["starts"] < data["ends"])

    def test_signal_values_positive(self) -> None:
        """All signal values are positive."""
        from diffbio.sources.encode_peaks import (
            ENCODEPeakConfig,
            ENCODEPeakSource,
        )

        config = ENCODEPeakConfig(chromosome="chr22", max_peaks=50)
        source = ENCODEPeakSource(config)
        data = source.load()
        assert np.all(data["signal_values"] > 0)


# -------------------------------------------------------------------
# Integration test: full benchmark on real data
# -------------------------------------------------------------------


@pytest.mark.skipif(_SKIP, reason="ENCODE data not available")
class TestPeakCallingBenchmark:
    """Integration tests for the peak calling benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        """Run benchmark in quick mode."""
        bench = PeakCallingBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="epigenomics/peak_calling",
            required_metric_keys=[
                "precision",
                "recall",
                "f1",
                "jaccard",
            ],
        )

    def test_has_operator_tag(self, result: BenchmarkResult) -> None:
        """Result is tagged with the operator name."""
        assert "DifferentiablePeakCaller" in result.tags["operator"]

    def test_has_dataset_tag(self, result: BenchmarkResult) -> None:
        """Result is tagged with the dataset name."""
        assert result.tags["dataset"] == "ENCODE_CTCF_K562"

    def test_metrics_in_range(self, result: BenchmarkResult) -> None:
        """All quality metrics are between 0 and 1."""
        for key in ("precision", "recall", "f1", "jaccard"):
            value = result.metrics[key].value
            assert 0.0 <= value <= 1.0, f"{key}={value} out of [0, 1]"

    def test_has_config(self, result: BenchmarkResult) -> None:
        """Result config contains operator parameters."""
        assert "window_size" in result.config
        assert "temperature" in result.config

    def test_has_dataset_metadata(self, result: BenchmarkResult) -> None:
        """Result metadata contains dataset information."""
        info = result.metadata["dataset_info"]
        assert info["chromosome"] == "chr22"
        assert info["n_peaks_total"] > 0
        assert info["n_peaks_region"] > 0
