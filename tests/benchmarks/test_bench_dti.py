"""Tests for the DTI data-contract benchmark scaffolds."""

from __future__ import annotations

import numpy as np

from benchmarks.drug_discovery.bench_dti import (
    BioSNAPDTIBenchmark,
    DavisDTIBenchmark,
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
        assert result.metadata["baseline_families"] == [
            "non_differentiable_fingerprint",
            "differentiable_drug_encoder",
        ]


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
