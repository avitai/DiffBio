"""Tests for benchmarks.alignment.bench_pairwise.

Validates the pairwise alignment benchmark using SmoothSmithWaterman
on BAliBASE reference alignments (balifam100).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.alignment.bench_pairwise import PairwiseBenchmark
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Works/balifam")
_SKIP = not (_DATA_DIR / "balifam100" / "ref").exists()


@pytest.mark.skipif(_SKIP, reason="balifam dataset not available")
class TestPairwiseBenchmark:
    """Tests for the pairwise alignment benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        """Run benchmark in quick mode (3 families)."""
        bench = PairwiseBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="alignment/pairwise",
            required_metric_keys=[
                "avg_alignment_score",
                "n_pairs_evaluated",
                "alignment_scores_finite",
            ],
        )

    def test_has_operator_tag(self, result: BenchmarkResult) -> None:
        """Tags must identify the operator under test."""
        assert "SmoothSmithWaterman" in result.tags["operator"]

    def test_has_dataset_tag(self, result: BenchmarkResult) -> None:
        """Tags must identify the dataset."""
        assert result.tags["dataset"] == "balifam100"

    def test_alignment_score_is_positive(self, result: BenchmarkResult) -> None:
        """Average alignment score must be positive."""
        score = result.metrics["avg_alignment_score"].value
        assert score > 0.0

    def test_pairs_evaluated_positive(self, result: BenchmarkResult) -> None:
        """At least one pair must have been evaluated."""
        n_pairs = result.metrics["n_pairs_evaluated"].value
        assert n_pairs >= 1.0

    def test_all_scores_finite(self, result: BenchmarkResult) -> None:
        """All alignment scores should be finite."""
        n_finite = result.metrics["alignment_scores_finite"].value
        n_pairs = result.metrics["n_pairs_evaluated"].value
        assert n_finite == n_pairs

    def test_has_config(self, result: BenchmarkResult) -> None:
        """Result must record the operator configuration."""
        assert "temperature" in result.config
        assert "gap_open" in result.config
        assert "gap_extend" in result.config
        assert "max_seq_length" in result.config

    def test_has_dataset_metadata(self, result: BenchmarkResult) -> None:
        """Result must contain dataset info in metadata."""
        info = result.metadata["dataset_info"]
        assert "n_families" in info
        assert "n_pairs" in info

    def test_has_baselines_in_metadata(self, result: BenchmarkResult) -> None:
        """Result must store published baselines for comparison."""
        baselines = result.metadata["baselines"]
        assert "BLAST" in baselines
        assert "SSEARCH" in baselines
        assert "FASTA" in baselines
