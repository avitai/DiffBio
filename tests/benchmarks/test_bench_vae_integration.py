"""Tests for benchmarks.singlecell.bench_vae_integration.

TDD: These tests define the expected behavior of the VAE integration
benchmark before implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_vae_integration import (
    VAEIntegrationBenchmark,
)

_DATA_DIR = Path("/media/mahdi/ssd23/Data/scib")
_SKIP = not (_DATA_DIR / "Immune_ALL_human.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="Dataset not downloaded")
class TestVAEIntegrationBenchmark:
    """Tests for the VAE integration benchmark."""

    @pytest.fixture()
    def result(self) -> BenchmarkResult:
        """Run benchmark in quick mode (subsampled)."""
        bench = VAEIntegrationBenchmark(quick=True)
        return bench.run()

    def test_returns_benchmark_result(
        self, result: BenchmarkResult
    ) -> None:
        assert isinstance(result, BenchmarkResult)

    def test_name_is_correct(
        self, result: BenchmarkResult
    ) -> None:
        assert result.name == "singlecell/vae_integration"

    def test_has_aggregate_score(
        self, result: BenchmarkResult
    ) -> None:
        assert "aggregate_score" in result.metrics

    def test_has_elbo_metric(
        self, result: BenchmarkResult
    ) -> None:
        assert "elbo" in result.metrics

    def test_has_reconstruction_mse(
        self, result: BenchmarkResult
    ) -> None:
        assert "reconstruction_mse" in result.metrics

    def test_has_gradient_metrics(
        self, result: BenchmarkResult
    ) -> None:
        assert "gradient_norm" in result.metrics
        assert "gradient_nonzero" in result.metrics

    def test_has_timing(self, result: BenchmarkResult) -> None:
        assert result.timing is not None
        assert result.timing.wall_clock_sec > 0

    def test_metrics_are_calibrax_metric(
        self, result: BenchmarkResult
    ) -> None:
        for key, metric in result.metrics.items():
            assert isinstance(metric, Metric), (
                f"{key} is not a Metric"
            )
