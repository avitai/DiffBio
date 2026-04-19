"""Tests for the DTI data-contract benchmark scaffolds."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import benchmarks.drug_discovery.bench_dti as bench_dti
from benchmarks.drug_discovery.bench_dti import (
    BioSNAPDTIBenchmark,
    DavisDTIBenchmark,
    build_dti_pair_features,
    compute_affinity_regression_metrics,
    compute_binary_interaction_metrics,
    compute_ranking_metrics,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result


class TestDTIMetrics:
    """Tests for DTI metric helpers."""

    def test_affinity_regression_metrics_are_well_formed(self) -> None:
        metrics = compute_affinity_regression_metrics(
            targets=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            predictions=np.array([1.0, 2.5, 2.5], dtype=np.float32),
        )

        assert metrics["rmse"] >= 0.0
        assert -1.0 <= metrics["pearson"] <= 1.0
        assert -1.0 <= metrics["spearman"] <= 1.0

    def test_binary_interaction_metrics_are_well_formed(self) -> None:
        metrics = compute_binary_interaction_metrics(
            targets=np.array([1, 0, 1, 0], dtype=np.int32),
            scores=np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32),
        )

        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    def test_binary_calibration_metrics_are_well_formed(self) -> None:
        metrics = bench_dti.compute_calibration_metrics(
            targets=np.array([1, 0, 1, 0], dtype=np.int32),
            scores=np.array([4.0, -4.0, 2.0, -2.0], dtype=np.float32),
        )

        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert 0.0 <= metrics["expected_calibration_error"] <= 1.0
        assert metrics["brier_score"] < 0.05

    def test_ranking_metrics_are_well_formed(self) -> None:
        metrics = compute_ranking_metrics(
            targets=np.array([1, 0, 0, 1], dtype=np.int32),
            scores=np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32),
            group_ids=["P0", "P0", "P1", "P1"],
            ks=(1, 2),
        )

        assert 0.0 <= metrics["mrr"] <= 1.0
        assert 0.0 <= metrics["recall_at_1"] <= 1.0
        assert 0.0 <= metrics["recall_at_2"] <= 1.0

    def test_pair_features_reuse_source_contract_validation(self) -> None:
        bad = {
            "pair_ids": ["pair_0"],
            "protein_ids": ["P0"],
            "protein_sequences": ["MAAA"],
            "drug_ids": ["D0"],
            "targets": np.array([1.0], dtype=np.float32),
            "task_type": "affinity_regression",
            "dataset_provenance": {
                "dataset_name": "davis",
                "split": "train",
                "source_type": "synthetic_scaffold",
                "source_path": None,
                "seed": 42,
                "task_type": "affinity_regression",
                "n_pairs": 1,
                "promotion_eligible": False,
                "biological_validation": "contract_validation_only",
            },
        }

        with pytest.raises(ValueError, match="missing required keys"):
            build_dti_pair_features(bad)


class TestDTITrainingSubstrate:
    """Tests for the Opifex-owned DTI training boundary."""

    def test_optimizer_contract_uses_opifex_training_substrate(self) -> None:
        assert bench_dti.DTI_TRAINING_SUBSTRATE == {
            "optimizer_factory": "opifex.core.training.optimizers.create_optimizer",
            "optimizer_config": "opifex.core.training.optimizers.OptimizerConfig",
            "optimizer_type": "adam",
        }
        assert "optax.adam" not in inspect.getsource(bench_dti)


class TestDavisDTIBenchmark:
    """Tests for the Davis DTI scaffold benchmark."""

    def test_quick_benchmark_runs(self) -> None:
        result = DavisDTIBenchmark(quick=True).run()

        assert_valid_benchmark_result(
            result,
            expected_name="drug_discovery/dti_davis",
            required_metric_keys=["rmse", "pearson", "spearman"],
        )
        assert result.tags["dataset"] == "davis"
        assert result.tags["task"] == "affinity_regression"
        assert result.metadata["paired_contract"]["task_type"] == "affinity_regression"
        assert result.metadata["paired_contract"]["required_keys"] == [
            "pair_ids",
            "protein_ids",
            "protein_sequences",
            "drug_ids",
            "drug_smiles",
            "targets",
            "task_type",
            "dataset_provenance",
        ]
        assert result.metadata["dataset_provenance"]["dataset_name"] == "davis"
        assert result.metadata["dataset_provenance"]["split"] == "train"
        assert result.metadata["dataset_provenance"]["source_type"] == "synthetic_scaffold"
        assert result.metadata["dataset_provenance"]["promotion_eligible"] is False
        assert result.metadata["metric_contract"] == {
            "task_type": "affinity_regression",
            "primary_metric": "rmse",
            "metric_groups": {
                "regression": ["rmse", "pearson", "spearman"],
                "classification": [],
                "ranking": [],
            },
        }
        assert result.metadata["baseline_families"] == [
            "non_differentiable_fingerprint",
            "differentiable_drug_encoder",
        ]
        _assert_dti_comparison_report(
            result,
            dataset_name="davis",
            task_name="affinity_regression",
            primary_metric="rmse",
            metric_direction="lower_is_better",
        )
        _assert_differentiable_dti_pipeline_metadata(result)


class TestBioSNAPDTIBenchmark:
    """Tests for the BioSNAP DTI scaffold benchmark."""

    def test_quick_benchmark_runs(self) -> None:
        result = BioSNAPDTIBenchmark(quick=True).run()

        assert_valid_benchmark_result(
            result,
            expected_name="drug_discovery/dti_biosnap",
            required_metric_keys=["roc_auc", "pr_auc", "mrr", "recall_at_1", "recall_at_5"],
        )
        assert result.tags["dataset"] == "biosnap"
        assert result.tags["task"] == "binary_interaction"
        assert result.metadata["paired_contract"]["task_type"] == "binary_interaction"
        assert result.metadata["paired_contract"]["group_key"] == "protein_ids"
        assert result.metadata["dataset_provenance"]["dataset_name"] == "biosnap"
        assert result.metadata["dataset_provenance"]["split"] == "train"
        assert result.metadata["dataset_provenance"]["source_type"] == "synthetic_scaffold"
        assert result.metadata["dataset_provenance"]["promotion_eligible"] is False
        assert result.metadata["metric_contract"] == {
            "task_type": "binary_interaction",
            "primary_metric": "roc_auc",
            "metric_groups": {
                "regression": [],
                "classification": ["roc_auc", "pr_auc"],
                "calibration": ["brier_score", "expected_calibration_error"],
                "ranking": ["mrr", "recall_at_1", "recall_at_5"],
            },
        }
        assert 0.0 <= result.metrics["brier_score"].value <= 1.0
        assert 0.0 <= result.metrics["expected_calibration_error"].value <= 1.0
        _assert_dti_comparison_report(
            result,
            dataset_name="biosnap",
            task_name="binary_interaction",
            primary_metric="roc_auc",
            metric_direction="higher_is_better",
        )
        _assert_differentiable_dti_pipeline_metadata(result)


def test_dti_docs_keep_synthetic_comparisons_outside_stable_promotion() -> None:
    doc = Path("docs/user-guide/operators/drug-discovery.md").read_text(encoding="utf-8")

    assert "DTI comparison report" in doc
    assert "synthetic-scaffold comparison evidence" in doc
    assert "not stable biological promotion evidence" in doc


def _assert_dti_comparison_report(
    result,
    *,
    dataset_name: str,
    task_name: str,
    primary_metric: str,
    metric_direction: str,
) -> None:
    """Assert DTI benchmark output includes the shared comparison report."""
    report = result.metadata["dti_comparison_report"]

    assert report["report_version"] == "dti_comparison_v1"
    assert report["comparison_axes"] == ["dataset", "task", "encoder_path"]
    assert report["primary_metric"] == primary_metric
    assert report["metric_direction"] == metric_direction
    assert report["required_encoder_paths"] == [
        "differentiable_pipeline",
        "fixed_scaffold_baseline",
    ]
    assert report["stable_scope"] == "excluded"
    assert report["stable_claim"] == "synthetic_scaffold_comparison_only"
    assert set(report["models"]) == {
        "differentiable_pipeline",
        "fixed_scaffold_baseline",
    }
    assert report["models"]["differentiable_pipeline"]["comparison_key"] == {
        "dataset": dataset_name,
        "task": task_name,
        "encoder_path": "differentiable_pipeline",
    }
    assert report["models"]["fixed_scaffold_baseline"]["comparison_key"] == {
        "dataset": dataset_name,
        "task": task_name,
        "encoder_path": "fixed_scaffold_baseline",
    }
    assert primary_metric in report["models"]["differentiable_pipeline"]["metrics"]
    assert primary_metric in report["models"]["fixed_scaffold_baseline"]["metrics"]
    assert primary_metric in report["primary_delta_vs_fixed_scaffold"]


def _assert_differentiable_dti_pipeline_metadata(result) -> None:
    """Assert benchmark output is wired through the shared DTI pipeline."""
    assert result.tags["operator"] == "DifferentiableDTIPipeline"
    assert result.tags["model_family"] == "sequence_transformer"
    assert result.tags["adapter_mode"] == "native_trainable"
    assert result.metadata["foundation_model"]["preprocessing_version"] == "protein_one_hot_v1"
    assert result.metadata["comparison_axes"] == [
        "dataset",
        "task",
        "model_family",
        "adapter_mode",
        "artifact_id",
        "preprocessing_version",
    ]
    assert result.metadata["dti_pipeline"] == {
        "integration_layer": "shared_dti_pipeline_v1",
        "pipeline_name": "DifferentiableDTIPipeline",
        "protein_encoder": {
            "operator": "TransformerSequenceEncoder",
            "model_family": "sequence_transformer",
            "adapter_mode": "native_trainable",
            "preprocessing_version": "protein_one_hot_v1",
        },
        "drug_encoder": {
            "operator": "DifferentiableMolecularFingerprint",
            "differentiable": True,
        },
    }
    assert result.metadata["training"] == {
        "optimizer_factory": "opifex.core.training.optimizers.create_optimizer",
        "optimizer_config": "opifex.core.training.optimizers.OptimizerConfig",
        "optimizer_type": "adam",
        "n_steps": 40,
        "learning_rate": 0.01,
    }
    assert result.config["input_mode"] == "paired_encoded_graph_sequence"
    assert result.config["protein_encoder"] == "TransformerSequenceEncoder"
    assert result.config["drug_encoder"] == "DifferentiableMolecularFingerprint"
