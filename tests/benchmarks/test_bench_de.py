"""Tests for benchmarks.statistical.bench_de."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.statistical.bench_de import DEBenchmark
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Data/scib")
_SKIP = not (_DATA_DIR / "Immune_ALL_human.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="Dataset not downloaded")
class TestDEBenchmark:
    """Tests for the differential expression benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        """Run benchmark once for the entire test class."""
        bench = DEBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(
        self,
        result: BenchmarkResult,
    ) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="statistical/de",
            required_metric_keys=[
                "concordance_with_ttest",
                "n_de_genes",
            ],
        )

    def test_concordance_in_range(
        self,
        result: BenchmarkResult,
    ) -> None:
        """Concordance (Jaccard) must be in [0, 1]."""
        score = result.metrics["concordance_with_ttest"].value
        assert 0.0 <= score <= 1.0

    def test_n_de_genes_positive(
        self,
        result: BenchmarkResult,
    ) -> None:
        """Must identify at least some DE genes."""
        n_de = result.metrics["n_de_genes"].value
        assert n_de > 0

    def test_has_log_likelihood(
        self,
        result: BenchmarkResult,
    ) -> None:
        """Total log-likelihood must be present and finite."""
        assert "total_log_likelihood" in result.metrics
        ll = result.metrics["total_log_likelihood"].value
        assert ll != 0.0

    def test_operator_tag(
        self,
        result: BenchmarkResult,
    ) -> None:
        """Operator tag must reference DifferentiableNBGLM."""
        assert "DifferentiableNBGLM" in result.tags["operator"]

    def test_dataset_tag(
        self,
        result: BenchmarkResult,
    ) -> None:
        """Dataset tag must be immune_human."""
        assert result.tags["dataset"] == "immune_human"

    def test_has_comparison_in_metadata(
        self,
        result: BenchmarkResult,
    ) -> None:
        """Baselines must be present in metadata."""
        assert "baselines" in result.metadata
        assert len(result.metadata["baselines"]) > 0
