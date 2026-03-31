"""Tests for benchmarks.alignment.bench_msa.

Validates the MSA benchmark on BAliBASE reference alignments (balifam100).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.alignment.bench_msa import MSABenchmark
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Works/balifam")
_SKIP = not (_DATA_DIR / "balifam100" / "ref").exists()


@pytest.mark.skipif(_SKIP, reason="balifam dataset not available")
class TestMSABenchmark:
    """Tests for the MSA alignment benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        """Run benchmark in quick mode (3 families)."""
        bench = MSABenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="alignment/msa",
            required_metric_keys=["sp_score", "tc_score"],
        )

    def test_has_operator_tag(self, result: BenchmarkResult) -> None:
        """Tags must identify the operator under test."""
        assert "SoftProgressiveMSA" in result.tags["operator"]

    def test_has_dataset_tag(self, result: BenchmarkResult) -> None:
        """Tags must identify the dataset."""
        assert result.tags["dataset"] == "balifam100"

    def test_sp_score_in_range(self, result: BenchmarkResult) -> None:
        """SP (Sum of Pairs) score is between 0 and 1."""
        sp = result.metrics["sp_score"].value
        assert 0.0 <= sp <= 1.0

    def test_tc_score_in_range(self, result: BenchmarkResult) -> None:
        """TC (Total Column) score is between 0 and 1."""
        tc = result.metrics["tc_score"].value
        assert 0.0 <= tc <= 1.0

    def test_has_config(self, result: BenchmarkResult) -> None:
        """Result must record the operator configuration."""
        assert "max_seq_length" in result.config
        assert "hidden_dim" in result.config
        assert "alphabet_size" in result.config

    def test_has_dataset_metadata(self, result: BenchmarkResult) -> None:
        """Result must contain dataset info in metadata."""
        info = result.metadata["dataset_info"]
        assert "n_families" in info
        assert "n_evaluated" in info

    def test_has_baselines_in_metadata(self, result: BenchmarkResult) -> None:
        """Result must store published baselines for comparison."""
        baselines = result.metadata["baselines"]
        assert "MAFFT" in baselines
        assert "ClustalW" in baselines
        assert "MUSCLE" in baselines
        assert "T-Coffee" in baselines
