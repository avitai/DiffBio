"""Tests for single-cell foundation benchmark helper contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.singlecell._foundation import (
    SINGLECELL_FOUNDATION_DATASET_CONTRACT_KEYS,
    SINGLECELL_FOUNDATION_DEFERRED_TASKS,
    SINGLECELL_FOUNDATION_SUPPORT_MATRIX,
    SINGLECELL_FOUNDATION_SUITE_SCENARIOS,
    compute_annotation_metrics,
    stratified_cell_annotation_split,
)


class TestSingleCellFoundationConstants:
    """Tests for the declared single-cell benchmark contract."""

    def test_suite_scenarios_cover_planned_tasks(self) -> None:
        assert SINGLECELL_FOUNDATION_SUITE_SCENARIOS == {
            "cell_annotation": "singlecell/foundation_annotation",
            "batch_correction": "singlecell/batch_correction",
            "grn_transfer": "singlecell/grn",
        }

    def test_dataset_contract_requires_explicit_cell_ids(self) -> None:
        assert SINGLECELL_FOUNDATION_DATASET_CONTRACT_KEYS == (
            "counts",
            "batch_labels",
            "cell_type_labels",
            "cell_ids",
            "embeddings",
            "gene_names",
        )

    def test_support_matrix_tracks_only_verified_single_cell_modes(self) -> None:
        assert SINGLECELL_FOUNDATION_SUPPORT_MATRIX == {
            "GeneformerPrecomputedAdapter": {
                "adapter_key": "geneformer_precomputed",
                "stable_modes": ("precomputed",),
                "verified_tasks": ("cell_annotation", "batch_correction"),
                "planned_tasks": ("grn_transfer",),
                "stable_scope_exclusions": (
                    "direct_checkpoint_loading",
                    "tokenizer_interchange",
                    "generic_fine_tuning",
                ),
            },
            "ScGPTPrecomputedAdapter": {
                "adapter_key": "scgpt_precomputed",
                "stable_modes": ("precomputed",),
                "verified_tasks": ("cell_annotation", "batch_correction"),
                "planned_tasks": ("grn_transfer",),
                "stable_scope_exclusions": (
                    "direct_checkpoint_loading",
                    "tokenizer_interchange",
                    "generic_fine_tuning",
                ),
            },
        }

    def test_deferred_tasks_keep_grn_outside_phase_3_stable_scope(self) -> None:
        assert SINGLECELL_FOUNDATION_DEFERRED_TASKS == {
            "grn_transfer": {
                "scenario": "singlecell/grn",
                "status": "deferred",
                "stable_scope": "excluded",
                "reason": (
                    "Phase 3 stable imported-model claims are benchmarked only for "
                    "cell_annotation and batch_correction."
                ),
                "required_harness": "dedicated_foundation_grn_harness",
            }
        }

    def test_user_guide_exposes_the_same_single_cell_support_matrix(self) -> None:
        doc_path = (
            Path(__file__).resolve().parents[2] / "docs/user-guide/operators/foundation-models.md"
        )
        doc = doc_path.read_text(encoding="utf-8")

        assert "## Single-Cell Support Matrix" in doc
        assert "| `GeneformerPrecomputedAdapter` | `precomputed` |" in doc
        assert "| `ScGPTPrecomputedAdapter` | `precomputed` |" in doc
        assert "`cell_annotation`, `batch_correction`" in doc
        assert "`GRN transfer` remains planned" in doc
        assert "dedicated foundation-aware GRN harness" in doc
        assert "generic checkpoint loading" in doc


class TestStratifiedCellAnnotationSplit:
    """Tests for deterministic annotation splitting."""

    def test_split_is_deterministic_and_label_stratified(self) -> None:
        labels = np.repeat(np.arange(3, dtype=np.int32), 5)

        train_a, test_a = stratified_cell_annotation_split(labels, train_fraction=0.6, seed=7)
        train_b, test_b = stratified_cell_annotation_split(labels, train_fraction=0.6, seed=7)

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)

        for label in np.unique(labels):
            assert np.sum(labels[train_a] == label) == 3
            assert np.sum(labels[test_a] == label) == 2

    def test_split_rejects_singleton_label(self) -> None:
        labels = np.array([0, 0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="at least two cells"):
            stratified_cell_annotation_split(labels)


class TestComputeAnnotationMetrics:
    """Tests for cell-annotation metric computation."""

    def test_metrics_match_expected_accuracy_and_macro_f1(self) -> None:
        true_labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        predicted_labels = np.array([0, 1, 1, 1, 2, 0], dtype=np.int32)

        metrics = compute_annotation_metrics(true_labels, predicted_labels)

        assert metrics["accuracy"] == pytest.approx(4 / 6)
        assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)

    def test_metrics_require_matching_shapes(self) -> None:
        with pytest.raises(ValueError, match="identical shapes"):
            compute_annotation_metrics(
                np.array([0, 1], dtype=np.int32),
                np.array([0], dtype=np.int32),
            )
