"""Tests for benchmarks.rna_structure.bench_rna_fold.

Validates the RNA folding benchmark and its helper functions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks._metrics.structure import (
    base_pair_metrics,
    parse_dbn_pairs,
)
from benchmarks.rna_structure.bench_rna_fold import (
    RNAFoldBenchmark,
    encode_rna_sequence,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Works/RNAFoldAssess/tutorial/processed_data")
_SKIP = not (_DATA_DIR / "example_data_structure.csv").exists()


# -------------------------------------------------------------------
# Unit tests: encode_rna_sequence
# -------------------------------------------------------------------


class TestEncodeRNASequence:
    """Tests for the RNA one-hot encoding helper."""

    def test_basic_encoding_shape(self) -> None:
        """Encoding a 4-nucleotide sequence gives (4, 4) array."""
        encoded = encode_rna_sequence("ACGU")
        assert encoded.shape == (4, 4)

    def test_one_hot_values(self) -> None:
        """Each position has exactly one 1.0 in the correct column."""
        encoded = encode_rna_sequence("ACGU")
        # A=0, C=1, G=2, U=3
        assert float(encoded[0, 0]) == 1.0  # A
        assert float(encoded[1, 1]) == 1.0  # C
        assert float(encoded[2, 2]) == 1.0  # G
        assert float(encoded[3, 3]) == 1.0  # U

    def test_t_treated_as_u(self) -> None:
        """T nucleotides are mapped to the U column."""
        encoded = encode_rna_sequence("T")
        assert float(encoded[0, 3]) == 1.0

    def test_lowercase_handled(self) -> None:
        """Lowercase sequences are uppercased before encoding."""
        encoded = encode_rna_sequence("acgu")
        assert encoded.shape == (4, 4)
        assert float(encoded[0, 0]) == 1.0

    def test_unknown_gets_uniform(self) -> None:
        """Unknown characters get uniform 0.25 distribution."""
        encoded = encode_rna_sequence("N")
        expected = 1.0 / 4.0
        for col in range(4):
            np.testing.assert_allclose(float(encoded[0, col]), expected, atol=1e-6)


# -------------------------------------------------------------------
# Unit tests: parse_dbn_pairs
# -------------------------------------------------------------------


class TestParseDbnPairs:
    """Tests for dot-bracket notation parsing."""

    def test_simple_hairpin(self) -> None:
        """Parse a simple hairpin loop."""
        pairs = parse_dbn_pairs("(((...)))")
        assert pairs == {(0, 8), (1, 7), (2, 6)}

    def test_no_pairs(self) -> None:
        """All dots means no pairs."""
        pairs = parse_dbn_pairs(".....")
        assert pairs == set()

    def test_nested(self) -> None:
        """Parse nested structures."""
        pairs = parse_dbn_pairs("((..))")
        assert pairs == {(0, 5), (1, 4)}

    def test_unbalanced_close_raises(self) -> None:
        """Unbalanced closing paren raises ValueError."""
        with pytest.raises(ValueError, match="Unbalanced"):
            parse_dbn_pairs("((.).))")

    def test_unbalanced_open_raises(self) -> None:
        """Unbalanced opening paren raises ValueError."""
        with pytest.raises(ValueError, match="Unbalanced"):
            parse_dbn_pairs("(((..))")

    def test_invalid_char_raises(self) -> None:
        """Non-DBN characters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid character"):
            parse_dbn_pairs("((.X.))")


# -------------------------------------------------------------------
# Unit tests: base_pair_metrics
# -------------------------------------------------------------------


class TestBasePairMetrics:
    """Tests for the base pair metric computation."""

    def test_perfect_prediction(self) -> None:
        """Perfect prediction gives F1 = 1.0."""
        # DBN: ((...)). -> pairs (0,6), (1,5)
        dbn = "((...))."
        n = len(dbn)
        bp_probs = np.zeros((n, n), dtype=np.float32)
        # Set true pair positions above threshold
        bp_probs[0, 6] = 0.9
        bp_probs[1, 5] = 0.9

        result = base_pair_metrics(bp_probs, dbn)
        assert result["sensitivity"] == 1.0
        assert result["ppv"] == 1.0
        assert result["f1"] == 1.0

    def test_no_predictions(self) -> None:
        """No predictions gives F1 = 0.0."""
        dbn = "((...))."
        n = len(dbn)
        bp_probs = np.zeros((n, n), dtype=np.float32)

        result = base_pair_metrics(bp_probs, dbn)
        assert result["sensitivity"] == 0.0
        assert result["f1"] == 0.0
        assert result["n_predicted_pairs"] == 0

    def test_all_wrong_predictions(self) -> None:
        """Wrong predictions give ppv = 0.0."""
        dbn = "......((...))."
        n = len(dbn)
        bp_probs = np.zeros((n, n), dtype=np.float32)
        # Predict a pair that doesn't exist in truth
        bp_probs[0, 5] = 0.9

        result = base_pair_metrics(bp_probs, dbn)
        assert result["ppv"] == 0.0

    def test_partial_prediction(self) -> None:
        """Partial prediction gives intermediate metrics."""
        # DBN: ((..(..))) -> pairs (0,9), (1,8), (4,7)
        dbn = "((..(..)))"
        n = len(dbn)
        bp_probs = np.zeros((n, n), dtype=np.float32)
        # Predict 2 of 3 correctly
        bp_probs[0, 9] = 0.9
        bp_probs[1, 8] = 0.9

        result = base_pair_metrics(bp_probs, dbn)
        np.testing.assert_allclose(result["sensitivity"], 2.0 / 3.0, atol=1e-6)
        assert result["ppv"] == 1.0

    def test_counts_correct(self) -> None:
        """n_true_pairs and n_predicted_pairs are correct."""
        dbn = "((...))"
        n = len(dbn)
        bp_probs = np.zeros((n, n), dtype=np.float32)
        bp_probs[0, 6] = 0.8
        bp_probs[2, 5] = 0.8  # False positive

        result = base_pair_metrics(bp_probs, dbn)
        assert result["n_true_pairs"] == 2
        assert result["n_predicted_pairs"] == 2

    def test_threshold_respected(self) -> None:
        """Pairs below threshold are not counted."""
        dbn = "((...))."
        n = len(dbn)
        bp_probs = np.zeros((n, n), dtype=np.float32)
        bp_probs[0, 5] = 0.3  # Below default threshold 0.5

        result = base_pair_metrics(bp_probs, dbn)
        assert result["n_predicted_pairs"] == 0

    def test_custom_threshold(self) -> None:
        """Custom threshold changes what is predicted."""
        dbn = "((...))."
        n = len(dbn)
        bp_probs = np.zeros((n, n), dtype=np.float32)
        bp_probs[0, 5] = 0.3

        result = base_pair_metrics(bp_probs, dbn, threshold=0.2)
        assert result["n_predicted_pairs"] == 1


# -------------------------------------------------------------------
# Integration test: full benchmark on real data
# -------------------------------------------------------------------


@pytest.mark.skipif(_SKIP, reason="ArchiveII dataset not found")
class TestRNAFoldBenchmark:
    """Integration tests for the RNA fold benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        """Run benchmark in quick mode."""
        bench = RNAFoldBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="rna_structure/rna_fold",
            required_metric_keys=["f1", "sensitivity", "ppv"],
        )

    def test_has_operator_tag(self, result: BenchmarkResult) -> None:
        """Result is tagged with the operator name."""
        assert "DifferentiableRNAFold" in result.tags["operator"]

    def test_has_dataset_tag(self, result: BenchmarkResult) -> None:
        """Result is tagged with the dataset name."""
        assert result.tags["dataset"] == "archiveII"

    def test_f1_in_range(self, result: BenchmarkResult) -> None:
        """F1 score is between 0 and 1."""
        score = result.metrics["f1"].value
        assert 0.0 <= score <= 1.0

    def test_has_config(self, result: BenchmarkResult) -> None:
        """Result config contains operator parameters."""
        assert "temperature" in result.config
        assert "min_hairpin_loop" in result.config

    def test_has_dataset_metadata(self, result: BenchmarkResult) -> None:
        """Result metadata contains dataset information."""
        info = result.metadata["dataset_info"]
        assert "n_sequences" in info
        assert info["n_sequences"] > 0
