"""Tests for benchmarks.protein.bench_secondary_structure.

Validates the protein secondary structure benchmark, its backbone
generators, and Q3 metric computation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from diffbio.operators.foundation_models import FOUNDATION_BENCHMARK_COMPARISON_AXES
from benchmarks.protein.bench_secondary_structure import (
    SecondaryStructureBenchmark,
    _generate_coil_backbone,
    _generate_helix_backbone,
    _generate_strand_backbone,
    compute_q3_metrics,
    generate_ideal_backbone,
)
from diffbio.operators.protein.secondary_structure import (
    SS_HELIX,
    SS_LOOP,
    SS_STRAND,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result


# -------------------------------------------------------------------
# Unit tests: backbone generators
# -------------------------------------------------------------------


class TestBackboneGenerators:
    """Tests for ideal backbone coordinate generators."""

    def test_helix_shape(self) -> None:
        """Helix generator returns (n, 4, 3) coordinates."""
        coords = _generate_helix_backbone(10)
        assert coords.shape == (10, 4, 3)

    def test_strand_shape(self) -> None:
        """Strand generator returns (n, 4, 3) coordinates."""
        coords = _generate_strand_backbone(8)
        assert coords.shape == (8, 4, 3)

    def test_coil_shape(self) -> None:
        """Coil generator returns (n, 4, 3) coordinates."""
        coords = _generate_coil_backbone(12)
        assert coords.shape == (12, 4, 3)

    def test_helix_no_nans(self) -> None:
        """Helix coordinates contain no NaN values."""
        coords = _generate_helix_backbone(20)
        assert not jnp.any(jnp.isnan(coords))

    def test_strand_no_nans(self) -> None:
        """Strand coordinates contain no NaN values."""
        coords = _generate_strand_backbone(15)
        assert not jnp.any(jnp.isnan(coords))

    def test_coil_no_nans(self) -> None:
        """Coil coordinates contain no NaN values."""
        coords = _generate_coil_backbone(15)
        assert not jnp.any(jnp.isnan(coords))

    def test_coil_reproducible(self) -> None:
        """Coil generator is deterministic with same seed."""
        a = _generate_coil_backbone(10, seed=42)
        b = _generate_coil_backbone(10, seed=42)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# -------------------------------------------------------------------
# Unit tests: generate_ideal_backbone
# -------------------------------------------------------------------


class TestGenerateIdealBackbone:
    """Tests for the combined backbone generator."""

    def test_coordinates_shape(self) -> None:
        """Combined backbone has shape (1, 50, 4, 3)."""
        coords, _ = generate_ideal_backbone()
        assert coords.shape == (1, 50, 4, 3)

    def test_labels_shape(self) -> None:
        """Labels have shape (50,)."""
        _, labels = generate_ideal_backbone()
        assert labels.shape == (50,)

    def test_label_counts(self) -> None:
        """Labels contain correct counts per class."""
        _, labels = generate_ideal_backbone()
        labels_np = np.asarray(labels)
        assert int(np.sum(labels_np == SS_HELIX)) == 20
        assert int(np.sum(labels_np == SS_STRAND)) == 15
        assert int(np.sum(labels_np == SS_LOOP)) == 15

    def test_no_nans(self) -> None:
        """Combined coordinates contain no NaN values."""
        coords, _ = generate_ideal_backbone()
        assert not jnp.any(jnp.isnan(coords))


# -------------------------------------------------------------------
# Unit tests: compute_q3_metrics
# -------------------------------------------------------------------


class TestComputeQ3Metrics:
    """Tests for Q3 accuracy metric computation."""

    def test_perfect_prediction(self) -> None:
        """Perfect prediction gives Q3 = 1.0 for all classes."""
        labels = jnp.array([0, 0, 1, 1, 2, 2])
        result = compute_q3_metrics(labels, labels)
        assert result["q3_overall"] == 1.0
        assert result["q3_helix"] == 1.0
        assert result["q3_strand"] == 1.0
        assert result["q3_coil"] == 1.0

    def test_all_wrong(self) -> None:
        """All-wrong prediction gives Q3 = 0.0."""
        true = jnp.array([0, 0, 1, 1, 2, 2])
        pred = jnp.array([1, 1, 2, 2, 0, 0])
        result = compute_q3_metrics(pred, true)
        assert result["q3_overall"] == 0.0
        assert result["q3_helix"] == 0.0
        assert result["q3_strand"] == 0.0
        assert result["q3_coil"] == 0.0

    def test_partial_prediction(self) -> None:
        """Partial prediction gives intermediate Q3."""
        true = jnp.array([0, 0, 1, 1, 2, 2])
        pred = jnp.array([0, 1, 1, 0, 2, 2])  # 4/6 correct
        result = compute_q3_metrics(pred, true)
        np.testing.assert_allclose(result["q3_overall"], 4.0 / 6.0, atol=1e-6)
        assert result["q3_coil"] == 0.5  # 1/2 correct
        assert result["q3_helix"] == 0.5  # 1/2 correct
        assert result["q3_strand"] == 1.0  # 2/2 correct

    def test_has_all_keys(self) -> None:
        """Result dict contains all expected metric keys."""
        labels = jnp.array([0, 1, 2])
        result = compute_q3_metrics(labels, labels)
        expected_keys = {
            "q3_overall",
            "q3_helix",
            "q3_strand",
            "q3_coil",
        }
        assert set(result.keys()) == expected_keys

    def test_values_in_range(self) -> None:
        """All Q3 values are between 0 and 1."""
        true = jnp.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        pred = jnp.array([0, 1, 1, 2, 2, 0, 0, 1, 1])
        result = compute_q3_metrics(pred, true)
        for value in result.values():
            assert 0.0 <= value <= 1.0


# -------------------------------------------------------------------
# Integration test: full benchmark
# -------------------------------------------------------------------


class TestSecondaryStructureBenchmark:
    """Integration tests for the SS benchmark."""

    @pytest.fixture(scope="class")
    def result(self):
        """Run benchmark in quick mode."""
        bench = SecondaryStructureBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="protein/secondary_structure",
            required_metric_keys=[
                "q3_overall",
                "q3_helix",
                "q3_strand",
                "q3_coil",
            ],
        )

    def test_has_operator_tag(self, result) -> None:
        """Result is tagged with the operator name."""
        assert result.tags["operator"] == ("DifferentiableSecondaryStructure")

    def test_has_dataset_tag(self, result) -> None:
        """Result is tagged with the dataset name."""
        assert result.tags["dataset"] == "ideal_structures"

    def test_q3_overall_in_range(self, result) -> None:
        """Q3 overall accuracy is between 0 and 1."""
        score = result.metrics["q3_overall"].value
        assert 0.0 <= score <= 1.0

    def test_q3_helix_in_range(self, result) -> None:
        """Q3 helix accuracy is between 0 and 1."""
        score = result.metrics["q3_helix"].value
        assert 0.0 <= score <= 1.0

    def test_q3_strand_in_range(self, result) -> None:
        """Q3 strand accuracy is between 0 and 1."""
        score = result.metrics["q3_strand"].value
        assert 0.0 <= score <= 1.0

    def test_q3_coil_in_range(self, result) -> None:
        """Q3 coil accuracy is between 0 and 1."""
        score = result.metrics["q3_coil"].value
        assert 0.0 <= score <= 1.0

    def test_has_config(self, result) -> None:
        """Result config contains operator parameters."""
        assert "temperature" in result.config
        assert "margin" in result.config
        assert "cutoff" in result.config

    def test_has_dataset_metadata(self, result) -> None:
        """Result metadata contains dataset information."""
        info = result.metadata["dataset_info"]
        assert info["n_residues"] == 50
        assert info["n_helix"] == 20
        assert info["n_strand"] == 15
        assert info["n_coil"] == 15

    def test_foundation_metadata_uses_protein_sequence_substrate(self, result) -> None:
        """Protein benchmark should expose the shared sequence foundation contract."""
        assert result.tags["model_family"] == "sequence_transformer"
        assert result.tags["adapter_mode"] == "frozen_encoder"
        assert result.tags["artifact_id"] == "diffbio.protein_sequence_encoder"
        assert result.tags["preprocessing_version"] == "protein_one_hot_v1"
        assert result.metadata["comparison_axes"] == list(FOUNDATION_BENCHMARK_COMPARISON_AXES)
        assert result.metadata["foundation_source_name"] == "diffbio_protein_frozen_encoder"
        assert result.metadata["embedding_source"] == "in_process_operator"
        assert result.metadata["protein_lm"] == {
            "adapter_scope": "protein_sequence_context",
            "stable_scope": "excluded",
            "scope_exclusions": [
                "external_protein_lm_checkpoint_import",
                "stable_imported_protein_lm_promotion",
            ],
        }


def test_protein_docs_keep_lm_scope_limited() -> None:
    """Protein docs should not overstate experimental foundation-model support."""
    from pathlib import Path

    doc = Path("docs/user-guide/operators/protein.md").read_text(encoding="utf-8")

    assert "Protein foundation-model scope" in doc
    assert "precomputed protein-LM artifacts" in doc
    assert "not stable imported protein-LM support" in doc
