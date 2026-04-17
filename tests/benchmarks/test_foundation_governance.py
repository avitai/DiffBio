"""Tests for Calibrax-native storage and regression checks on foundation suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks._foundation_models import (
    build_foundation_suite_report,
    build_foundation_suite_run,
    check_foundation_suite_regressions,
    save_foundation_suite_run,
)
from calibrax.core.models import MetricDirection
from diffbio.operators.foundation_models import FOUNDATION_BENCHMARK_COMPARISON_AXES


def _make_task_reports(
    *,
    native_accuracy: float = 0.92,
    native_train_loss: float = 0.10,
    imported_accuracy: float = 0.90,
    imported_train_loss: float = 0.12,
) -> dict[str, dict[str, Any]]:
    comparison_axes = list(FOUNDATION_BENCHMARK_COMPARISON_AXES)
    return {
        "cell_annotation": {
            "benchmark": "singlecell/foundation_annotation",
            "dataset": "immune_human",
            "task": "cell_annotation",
            "comparison_axes": comparison_axes,
            "model_order": ["diffbio_native", "geneformer_precomputed"],
            "models": {
                "diffbio_native": {
                    "metrics": {
                        "accuracy": native_accuracy,
                        "train_loss": native_train_loss,
                    },
                    "tags": {
                        "dataset": "immune_human",
                        "task": "cell_annotation",
                    },
                    "metadata": {},
                    "foundation_model": None,
                    "comparison_key": {
                        "dataset": "immune_human",
                        "task": "cell_annotation",
                        "model_family": None,
                        "adapter_mode": None,
                        "artifact_id": None,
                        "preprocessing_version": None,
                    },
                },
                "geneformer_precomputed": {
                    "metrics": {
                        "accuracy": imported_accuracy,
                        "train_loss": imported_train_loss,
                    },
                    "tags": {
                        "dataset": "immune_human",
                        "task": "cell_annotation",
                        "model_family": "single_cell_transformer",
                        "adapter_mode": "precomputed",
                        "artifact_id": "geneformer.v1",
                        "preprocessing_version": "rank_value_v1",
                    },
                    "metadata": {
                        "embedding_source": "artifact_store",
                        "foundation_source_name": "geneformer_precomputed",
                    },
                    "foundation_model": {
                        "dataset": "immune_human",
                        "task": "cell_annotation",
                        "model_family": "single_cell_transformer",
                        "adapter_mode": "precomputed",
                        "artifact_id": "geneformer.v1",
                        "preprocessing_version": "rank_value_v1",
                        "pooling_strategy": "mean",
                    },
                    "comparison_key": {
                        "dataset": "immune_human",
                        "task": "cell_annotation",
                        "model_family": "single_cell_transformer",
                        "adapter_mode": "precomputed",
                        "artifact_id": "geneformer.v1",
                        "preprocessing_version": "rank_value_v1",
                    },
                },
            },
        }
    }


def _make_suite_report(
    *,
    native_accuracy: float = 0.92,
    native_train_loss: float = 0.10,
    imported_accuracy: float = 0.90,
    imported_train_loss: float = 0.12,
) -> dict[str, Any]:
    return build_foundation_suite_report(
        suite_name="singlecell/foundation_quick_suite",
        task_order=("cell_annotation",),
        task_reports=_make_task_reports(
            native_accuracy=native_accuracy,
            native_train_loss=native_train_loss,
            imported_accuracy=imported_accuracy,
            imported_train_loss=imported_train_loss,
        ),
        task_scenarios={"cell_annotation": {"mode": "quick"}},
    )


class TestFoundationSuiteCalibraxGovernance:
    """Tests for storing foundation-suite reports in Calibrax-owned layers."""

    def test_build_foundation_suite_run_preserves_metric_semantics(self) -> None:
        report = _make_suite_report()

        run = build_foundation_suite_run(
            report,
            commit="abc123",
            branch="main",
            environment={"device": "cpu"},
        )

        assert run.commit == "abc123"
        assert run.branch == "main"
        assert run.environment == {"device": "cpu"}
        assert run.metadata["suite"] == "singlecell/foundation_quick_suite"
        assert run.metadata["regression_expectations"]["calibrax"]["threshold"] == 0.05
        assert run.metric_defs["accuracy"].direction == MetricDirection.HIGHER
        assert run.metric_defs["train_loss"].direction == MetricDirection.LOWER
        assert len(run.points) == 2

        native_point = next(
            point for point in run.points if "artifact_id" not in point.tags
        )
        assert native_point.name == "singlecell/foundation_annotation"
        assert native_point.scenario == "immune_human"
        assert native_point.tags == {
            "dataset": "immune_human",
            "task": "cell_annotation",
        }
        assert native_point.metrics["accuracy"].value == pytest.approx(0.92)
        assert native_point.metrics["train_loss"].value == pytest.approx(0.10)

    def test_check_foundation_suite_regressions_uses_calibrax_guard(self, tmp_path: Path) -> None:
        store_path = tmp_path / "foundation_store"
        baseline_report = _make_suite_report()

        baseline_result = check_foundation_suite_regressions(
            baseline_report,
            store_path,
            set_baseline_if_missing=True,
        )

        assert baseline_result.passed is True
        assert baseline_result.threshold == pytest.approx(0.05)
        assert baseline_result.baseline_id == baseline_result.current_id

        current_report = _make_suite_report(
            native_accuracy=0.82,
            native_train_loss=0.10,
            imported_accuracy=0.90,
            imported_train_loss=0.12,
        )
        regression_result = check_foundation_suite_regressions(current_report, store_path)

        assert regression_result.passed is False
        assert regression_result.threshold == pytest.approx(0.05)
        assert len(regression_result.regressions) == 1
        regression = regression_result.regressions[0]
        assert regression.metric == "accuracy"
        assert regression.point_name == "singlecell/foundation_annotation"
        assert regression.baseline_value == pytest.approx(0.92)
        assert regression.current_value == pytest.approx(0.82)

    def test_save_foundation_suite_run_persists_calibrax_run(self, tmp_path: Path) -> None:
        report = _make_suite_report()

        saved_path, run = save_foundation_suite_run(report, tmp_path / "store")

        assert saved_path.name == f"{run.id}.json"
        assert saved_path.exists()
