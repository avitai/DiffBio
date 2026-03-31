"""Tests for benchmarks.singlecell.bench_vae_integration."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_vae_integration import (
    VAEIntegrationBenchmark,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Data/scib")
_SKIP = not (_DATA_DIR / "Immune_ALL_human.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="Dataset not downloaded")
class TestVAEIntegrationBenchmark:
    """Tests for the VAE integration benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        """Run benchmark in quick mode (subsampled)."""
        bench = VAEIntegrationBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="singlecell/vae_integration",
            required_metric_keys=[
                "aggregate_score",
                "elbo",
                "reconstruction_mse",
            ],
        )
