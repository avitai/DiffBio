"""Tests for genomics foundation benchmark helper contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.genomics._foundation import (
    GENOMICS_FOUNDATION_DATASET_CONTRACT_KEYS,
    GENOMICS_FOUNDATION_DATASET_PROVENANCE,
    GENOMICS_FOUNDATION_DATASET_PROVENANCE_KEYS,
    GENOMICS_FOUNDATION_SUITE_SCENARIOS,
    compute_sequence_classification_metrics,
    resolve_genomics_dataset_provenance,
    stratified_sequence_classification_split,
    _resolve_task_report_dataset_provenance,
)


class TestGenomicsFoundationConstants:
    """Tests for the declared genomics benchmark contract."""

    def test_suite_scenarios_cover_planned_tasks(self) -> None:
        assert GENOMICS_FOUNDATION_SUITE_SCENARIOS == {
            "promoter": "genomics/promoter",
            "tfbs": "genomics/tfbs",
            "splice_site": "genomics/splice_site",
        }

    def test_dataset_contract_requires_explicit_sequence_ids(self) -> None:
        assert GENOMICS_FOUNDATION_DATASET_CONTRACT_KEYS == (
            "sequence_ids",
            "sequences",
            "one_hot_sequences",
            "labels",
        )

    def test_synthetic_dataset_provenance_marks_scaffold_not_promotable(self) -> None:
        assert GENOMICS_FOUNDATION_DATASET_PROVENANCE_KEYS == (
            "dataset_name",
            "source_type",
            "curation_status",
            "provenance_label",
            "biological_validation",
            "promotion_eligible",
        )
        assert GENOMICS_FOUNDATION_DATASET_PROVENANCE["synthetic_genomics"] == {
            "dataset_name": "synthetic_genomics",
            "source_type": "scaffold",
            "curation_status": "synthetic",
            "provenance_label": "deterministic_motif_scaffold",
            "biological_validation": "interface_validation_only",
            "promotion_eligible": False,
        }

    def test_unknown_dataset_requires_explicit_provenance(self) -> None:
        with pytest.raises(ValueError, match="dataset_provenance"):
            resolve_genomics_dataset_provenance("custom_curated_dataset")

    def test_explicit_curated_dataset_provenance_is_accepted(self) -> None:
        provenance = {
            "dataset_name": "custom_curated_dataset",
            "source_type": "curated",
            "curation_status": "curated",
            "provenance_label": "curated_sequence_panel_v1",
            "biological_validation": "heldout_biological_validation",
            "promotion_eligible": True,
        }

        assert (
            resolve_genomics_dataset_provenance("custom_curated_dataset", provenance) == provenance
        )

    def test_scaffold_dataset_provenance_cannot_be_promotion_eligible(self) -> None:
        provenance = {
            **GENOMICS_FOUNDATION_DATASET_PROVENANCE["synthetic_genomics"],
            "promotion_eligible": True,
        }

        with pytest.raises(ValueError, match="scaffold"):
            resolve_genomics_dataset_provenance("synthetic_genomics", provenance)

    def test_task_report_requires_each_model_dataset_provenance(self) -> None:
        task_report = {
            "models": {
                "diffbio_native": {
                    "metadata": {
                        "dataset_provenance": GENOMICS_FOUNDATION_DATASET_PROVENANCE[
                            "synthetic_genomics"
                        ]
                    }
                },
                "dnabert2_precomputed": {"metadata": {}},
            }
        }

        with pytest.raises(ValueError, match="missing dataset_provenance"):
            _resolve_task_report_dataset_provenance(task_report)

    def test_docs_keep_genomics_scaffold_outside_stable_promotion(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        user_guide = (repo_root / "docs/user-guide/operators/foundation-models.md").read_text(
            encoding="utf-8"
        )
        benchmark_guide = (repo_root / "docs/development/benchmarks.md").read_text(encoding="utf-8")

        assert "Phase 4 pre-promotion scaffold" in user_guide
        assert "not a stable genomics promotion claim" in user_guide
        assert "`dataset_provenance`" in user_guide
        assert "`promotion_eligible`: `false`" in user_guide
        assert "stable sequence integrations today" not in user_guide
        assert "Phase 4 scaffold: `DNABERT2PrecomputedAdapter`" in benchmark_guide
        assert "pending genomics realism and promotion evidence" in benchmark_guide
        assert "`source_type`: `scaffold`" in benchmark_guide


class TestStratifiedSequenceClassificationSplit:
    """Tests for deterministic genomics task splitting."""

    def test_split_is_deterministic_and_label_stratified(self) -> None:
        labels = np.repeat(np.arange(3, dtype=np.int32), 5)

        train_a, test_a = stratified_sequence_classification_split(
            labels,
            train_fraction=0.6,
            seed=11,
        )
        train_b, test_b = stratified_sequence_classification_split(
            labels,
            train_fraction=0.6,
            seed=11,
        )

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)

        for label in np.unique(labels):
            assert np.sum(labels[train_a] == label) == 3
            assert np.sum(labels[test_a] == label) == 2

    def test_split_rejects_singleton_label(self) -> None:
        labels = np.array([0, 0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="at least two sequences"):
            stratified_sequence_classification_split(labels)


class TestComputeSequenceClassificationMetrics:
    """Tests for genomics classification metric computation."""

    def test_metrics_match_expected_accuracy_and_macro_f1(self) -> None:
        true_labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        predicted_labels = np.array([0, 1, 1, 1, 2, 0], dtype=np.int32)

        metrics = compute_sequence_classification_metrics(true_labels, predicted_labels)

        assert metrics["accuracy"] == pytest.approx(4 / 6)
        assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)

    def test_metrics_require_matching_shapes(self) -> None:
        with pytest.raises(ValueError, match="identical shapes"):
            compute_sequence_classification_metrics(
                np.array([0, 1], dtype=np.int32),
                np.array([0], dtype=np.int32),
            )
