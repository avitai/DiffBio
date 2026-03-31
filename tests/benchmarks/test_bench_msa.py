"""Tests for benchmarks.alignment.bench_msa.

TDD: These tests define the expected behavior of the MSA benchmark
on BAliBASE reference alignments (balifam100).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult

from benchmarks.alignment.bench_msa import MSABenchmark

_DATA_DIR = Path("/media/mahdi/ssd23/Works/balifam")
_SKIP = not (_DATA_DIR / "balifam100" / "ref").exists()


@pytest.mark.skipif(_SKIP, reason="balifam dataset not available")
class TestMSABenchmark:
    """Tests for the MSA alignment benchmark."""

    @pytest.fixture()
    def result(self) -> BenchmarkResult:
        """Run benchmark in quick mode (3 families)."""
        bench = MSABenchmark(quick=True)
        return bench.run()

    def test_returns_benchmark_result(
        self, result: BenchmarkResult
    ) -> None:
        """Benchmark must return a calibrax BenchmarkResult."""
        assert isinstance(result, BenchmarkResult)

    def test_name_is_correct(
        self, result: BenchmarkResult
    ) -> None:
        """Result name must match the benchmark identifier."""
        assert result.name == "alignment/msa"

    def test_domain(self, result: BenchmarkResult) -> None:
        """Result must be in the diffbio_benchmarks domain."""
        assert result.domain == "diffbio_benchmarks"

    def test_has_operator_tag(
        self, result: BenchmarkResult
    ) -> None:
        """Tags must identify the operator under test."""
        assert "operator" in result.tags
        assert "SoftProgressiveMSA" in result.tags["operator"]

    def test_has_dataset_tag(
        self, result: BenchmarkResult
    ) -> None:
        """Tags must identify the dataset."""
        assert "dataset" in result.tags
        assert result.tags["dataset"] == "balifam100"

    def test_has_sp_score(self, result: BenchmarkResult) -> None:
        """Result must contain SP (Sum of Pairs) metric."""
        assert "sp_score" in result.metrics
        sp = result.metrics["sp_score"].value
        assert 0.0 <= sp <= 1.0

    def test_has_tc_score(self, result: BenchmarkResult) -> None:
        """Result must contain TC (Total Column) metric."""
        assert "tc_score" in result.metrics
        tc = result.metrics["tc_score"].value
        assert 0.0 <= tc <= 1.0

    def test_metrics_are_calibrax_metric(
        self, result: BenchmarkResult
    ) -> None:
        """All metrics must be calibrax Metric instances."""
        for key, metric in result.metrics.items():
            assert isinstance(metric, Metric), (
                f"{key} is not a Metric"
            )

    def test_has_gradient_metrics(
        self, result: BenchmarkResult
    ) -> None:
        """Result must include gradient flow diagnostics."""
        assert "gradient_norm" in result.metrics
        assert "gradient_nonzero" in result.metrics

    def test_has_timing(self, result: BenchmarkResult) -> None:
        """Result must include profiling timing data."""
        assert result.timing is not None
        assert result.timing.wall_clock_sec > 0

    def test_has_config(self, result: BenchmarkResult) -> None:
        """Result must record the operator configuration."""
        assert "max_seq_length" in result.config
        assert "hidden_dim" in result.config
        assert "alphabet_size" in result.config

    def test_has_dataset_metadata(
        self, result: BenchmarkResult
    ) -> None:
        """Result must contain dataset info in metadata."""
        assert "dataset_info" in result.metadata
        info = result.metadata["dataset_info"]
        assert "n_families" in info
        assert "n_evaluated" in info

    def test_has_baselines_in_metadata(
        self, result: BenchmarkResult
    ) -> None:
        """Result must store published baselines for comparison."""
        assert "baselines" in result.metadata
        baselines = result.metadata["baselines"]
        assert "MAFFT" in baselines
        assert "ClustalW" in baselines
        assert "MUSCLE" in baselines
        assert "T-Coffee" in baselines
