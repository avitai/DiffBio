"""Tests for benchmarks.drug_discovery.bench_molnet."""

from __future__ import annotations

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.drug_discovery.bench_molnet import MolNetBenchmark
from tests.benchmarks.conftest import assert_valid_benchmark_result


def _can_load_bbbp() -> bool:
    """Check whether the BBBP dataset can be loaded (network or cache)."""
    try:
        from diffbio.sources.molnet import MolNetSource, MolNetSourceConfig

        source = MolNetSource(
            MolNetSourceConfig(
                dataset_name="bbbp",
                split="train",
                download=True,
            )
        )
        return len(source) > 0
    except Exception:  # noqa: BLE001
        return False


_SKIP = not _can_load_bbbp()


@pytest.mark.skipif(_SKIP, reason="BBBP dataset not available")
class TestMolNetBenchmark:
    """Tests for the MolNet BBBP benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        """Run benchmark once for the entire test class."""
        bench = MolNetBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="drug_discovery/molnet",
            required_metric_keys=[
                "test_roc_auc",
                "train_roc_auc",
                "n_molecules",
            ],
        )

    def test_roc_auc_in_range(self, result: BenchmarkResult) -> None:
        """ROC-AUC must be in [0, 1]."""
        test_auc = result.metrics["test_roc_auc"].value
        assert 0.0 <= test_auc <= 1.0

    def test_train_roc_auc_in_range(self, result: BenchmarkResult) -> None:
        """Train ROC-AUC must be in [0, 1]."""
        train_auc = result.metrics["train_roc_auc"].value
        assert 0.0 <= train_auc <= 1.0

    def test_has_config(self, result: BenchmarkResult) -> None:
        """Config should contain operator hyperparameters."""
        assert "radius" in result.config
        assert "n_bits" in result.config
        assert "n_epochs" in result.config

    def test_operator_tag(self, result: BenchmarkResult) -> None:
        """Operator tag must reference the fingerprint operator."""
        assert "CircularFingerprint" in result.tags["operator"]

    def test_dataset_tag(self, result: BenchmarkResult) -> None:
        """Dataset tag must be bbbp."""
        assert result.tags["dataset"] == "bbbp"
