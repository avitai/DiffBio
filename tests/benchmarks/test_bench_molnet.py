"""Tests for benchmarks.drug_discovery.bench_molnet."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from calibrax.core.result import BenchmarkResult
from flax import nnx

from benchmarks.drug_discovery.bench_molnet import (
    MolNetBenchmark,
    featurize_molecules,
    rows_for_positions,
)
from diffbio.operators.drug_discovery import (
    DEFAULT_ATOM_FEATURES,
    CircularFingerprintConfig,
    CircularFingerprintOperator,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result


@dataclass(frozen=True)
class _Element:
    data: dict[str, Any]


def _source(records: list[tuple[str, float]]) -> list[_Element]:
    """A source of SMILES and labels, iterated the way the benchmark iterates MolNetSource."""
    return [_Element({"smiles": smiles, "y": label}) for smiles, label in records]


def _fingerprint_operator() -> CircularFingerprintOperator:
    config = CircularFingerprintConfig(
        radius=1, n_bits=16, differentiable=True, in_features=DEFAULT_ATOM_FEATURES
    )
    return CircularFingerprintOperator(config, rngs=nnx.Rngs(0))


_RECORDS = [("CCO", 1.0), ("not_a_smiles", 0.0), ("c1ccccc1", math.nan), ("CCN", 0.0), ("CCC", 1.0)]


def test_featurization_skips_unusable_molecules_and_records_their_positions() -> None:
    fingerprints, labels, positions = featurize_molecules(
        _source(_RECORDS), _fingerprint_operator()
    )

    np.testing.assert_array_equal(positions, [0, 3, 4])
    np.testing.assert_array_equal(np.asarray(labels), [1.0, 0.0, 1.0])
    assert fingerprints.shape == (3, 16)


def test_featurization_stops_at_the_limit_of_usable_molecules() -> None:
    fingerprints, labels, positions = featurize_molecules(
        _source(_RECORDS), _fingerprint_operator(), limit=2
    )

    np.testing.assert_array_equal(positions, [0, 3])
    np.testing.assert_array_equal(np.asarray(labels), [1.0, 0.0])
    assert fingerprints.shape == (2, 16)


def test_split_positions_select_the_rows_of_those_molecules() -> None:
    # Source positions 3 and 6 were not featurized, so rows run 0, 1, 2, 4, 5, 7.
    featurized = np.array([0, 1, 2, 4, 5, 7])

    rows = rows_for_positions(np.array([7, 3, 1, 0, 6]), featurized)

    np.testing.assert_array_equal(rows, [5, 1, 0])


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
