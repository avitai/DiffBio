"""Tests for the contextual epigenomics benchmark and ablation suite."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from calibrax.core.models import MetricDirection

from benchmarks.epigenomics.bench_chromatin_state_prediction import (
    ChromatinStatePredictionBenchmark,
    run_chromatin_state_prediction_ablation_suite,
)
from benchmarks.epigenomics.bench_contextual_peak_calling import (
    ContextualPeakCallingBenchmark,
    run_contextual_peak_calling_ablation_suite,
)
from benchmarks.epigenomics._contextual import CONTEXTUAL_TRAINING_SUBSTRATE
from benchmarks.epigenomics.contextual_suite import (
    build_contextual_epigenomics_suite_report,
    check_contextual_epigenomics_suite_regressions,
    run_contextual_epigenomics_suite,
    save_contextual_epigenomics_suite_run,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result


class _MissingTFContextSource:
    """Broken source used to prove missing-context failures are enforced."""

    def load(self) -> dict[str, Any]:
        return {
            "sequence": [[[1.0, 0.0, 0.0, 0.0]]],
            "chromatin_contacts": [[[1.0]]],
            "targets": [[1]],
        }


@pytest.fixture(scope="module")
def contextual_suite_report() -> dict[str, Any]:
    """Build one contextual suite report for Calibrax governance assertions."""
    return build_contextual_epigenomics_suite_report(run_contextual_epigenomics_suite(quick=True))


class TestContextualPeakCallingBenchmark:
    """Tests for the contextual peak-calling benchmark."""

    def test_quick_benchmark_runs_with_full_context_operator(self) -> None:
        result = ContextualPeakCallingBenchmark(quick=True).run()

        assert_valid_benchmark_result(
            result,
            expected_name="epigenomics/contextual_peak_calling",
            required_metric_keys=["precision", "recall", "f1", "chromatin_consistency"],
        )
        assert result.tags["task"] == "contextual_peak_calling"
        assert result.tags["dataset"] == "synthetic_contextual_epigenomics"
        assert result.tags["operator"] == "ContextualEpigenomicsOperator"
        assert result.tags["contextual_variant"] == "tf_plus_chromatin"
        assert result.metadata["contextual_contract"]["required_keys"] == [
            "sequence",
            "tf_context",
            "chromatin_contacts",
            "targets",
        ]
        assert result.metadata["contextual_contract"]["target_semantics"] == "binary_peak_mask"
        assert result.metadata["contextual_contract"]["num_output_classes"] == 1
        assert result.metadata["ablation"]["use_tf_context"] is True
        assert result.metadata["ablation"]["use_chromatin_guidance"] is True
        assert result.metadata["training"]["optimizer_factory"] == (
            "opifex.core.training.optimizers.create_optimizer"
        )
        assert result.metadata["training"]["optimizer_type"] == "adam"

    def test_optimizer_contract_uses_opifex_training_substrate(self) -> None:
        assert CONTEXTUAL_TRAINING_SUBSTRATE == {
            "optimizer_factory": "opifex.core.training.optimizers.create_optimizer",
            "optimizer_config": "opifex.core.training.optimizers.OptimizerConfig",
            "optimizer_type": "adam",
        }

    def test_missing_context_is_rejected(self) -> None:
        benchmark = ContextualPeakCallingBenchmark(
            quick=True,
            source_factory=lambda subsample: _MissingTFContextSource(),
        )

        with pytest.raises(ValueError, match="missing required keys"):
            benchmark.run()


class TestChromatinStatePredictionBenchmark:
    """Tests for the chromatin-state benchmark."""

    def test_quick_benchmark_runs_with_full_context_operator(self) -> None:
        result = ChromatinStatePredictionBenchmark(quick=True).run()

        assert_valid_benchmark_result(
            result,
            expected_name="epigenomics/chromatin_state_prediction",
            required_metric_keys=["accuracy", "chromatin_consistency"],
        )
        assert result.tags["task"] == "chromatin_state_prediction"
        assert result.tags["contextual_variant"] == "tf_plus_chromatin"
        assert result.metadata["contextual_contract"]["target_semantics"] == "chromatin_state_id"
        assert result.metadata["contextual_contract"]["num_output_classes"] == 3
        assert result.metadata["ablation"]["use_chromatin_guidance"] is True


class TestContextualAblationSuites:
    """Tests for variant-by-variant contextual ablation comparisons."""

    def test_peak_calling_ablation_suite_shows_context_and_chromatin_gains(self) -> None:
        results = run_contextual_peak_calling_ablation_suite(quick=True)

        assert tuple(results) == (
            "sequence_only",
            "tf_context",
            "tf_plus_chromatin",
        )
        assert (
            results["tf_context"].metrics["f1"].value > results["sequence_only"].metrics["f1"].value
        )
        assert (
            results["tf_plus_chromatin"].metrics["chromatin_consistency"].value
            > results["tf_context"].metrics["chromatin_consistency"].value
        )

    def test_chromatin_state_ablation_suite_shows_context_and_chromatin_gains(self) -> None:
        results = run_chromatin_state_prediction_ablation_suite(quick=True)

        assert tuple(results) == (
            "sequence_only",
            "tf_context",
            "tf_plus_chromatin",
        )
        assert (
            results["tf_context"].metrics["accuracy"].value
            > results["sequence_only"].metrics["accuracy"].value
        )
        assert (
            results["tf_plus_chromatin"].metrics["chromatin_consistency"].value
            > results["tf_context"].metrics["chromatin_consistency"].value
        )


class TestContextualEpigenomicsSuite:
    """Tests for the deterministic contextual epigenomics suite harness."""

    def test_suite_report_is_reproducible(self) -> None:
        results_a = run_contextual_epigenomics_suite(quick=True)
        results_b = run_contextual_epigenomics_suite(quick=True)

        report_a = build_contextual_epigenomics_suite_report(results_a)
        report_b = build_contextual_epigenomics_suite_report(results_b)

        assert report_a == report_b
        assert report_a["contextual_contract"] == {
            "required_keys": ["sequence", "tf_context", "chromatin_contacts", "targets"],
            "target_semantics_by_task": {
                "contextual_peak_calling": "binary_peak_mask",
                "chromatin_state_prediction": "chromatin_state_id",
            },
            "num_output_classes_by_task": {
                "contextual_peak_calling": 1,
                "chromatin_state_prediction": 3,
            },
        }
        assert tuple(report_a["task_order"]) == (
            "contextual_peak_calling",
            "chromatin_state_prediction",
        )
        assert tuple(report_a["tasks"]["contextual_peak_calling"]["variant_order"]) == (
            "sequence_only",
            "tf_context",
            "tf_plus_chromatin",
        )
        assert (
            report_a["tasks"]["contextual_peak_calling"]["deltas_from_sequence_only"]["tf_context"][
                "f1"
            ]
            > 0.0
        )
        assert (
            report_a["tasks"]["contextual_peak_calling"]["deltas_from_sequence_only"][
                "tf_plus_chromatin"
            ]["chromatin_consistency"]
            > 0.0
        )
        assert (
            report_a["tasks"]["contextual_peak_calling"]["variants"]["tf_plus_chromatin"][
                "metadata"
            ]["ablation"]["use_chromatin_guidance"]
            is True
        )
        assert (
            report_a["tasks"]["chromatin_state_prediction"]["contract"]["target_semantics"]
            == "chromatin_state_id"
        )
        assert (
            report_a["tasks"]["chromatin_state_prediction"]["deltas_from_sequence_only"][
                "tf_context"
            ]["accuracy"]
            > 0.0
        )

    def test_suite_report_has_variant_comparison_axes(
        self,
        contextual_suite_report: dict[str, Any],
    ) -> None:
        report = contextual_suite_report

        assert report["comparison_axes"] == ["dataset", "task", "contextual_variant"]
        assert report["contextual_evidence_scope"] == {
            "dataset": "synthetic_contextual_epigenomics",
            "source_type": "synthetic",
            "promotion_eligible": False,
            "stable_scope": "excluded",
            "reason": (
                "Synthetic contextual epigenomics ablations are regression evidence, "
                "not stable biological promotion evidence."
            ),
        }
        assert report["regression_expectations"]["required_variants"] == {
            "contextual_peak_calling": [
                "sequence_only",
                "tf_context",
                "tf_plus_chromatin",
            ],
            "chromatin_state_prediction": [
                "sequence_only",
                "tf_context",
                "tf_plus_chromatin",
            ],
        }
        assert (
            report["regression_expectations"]["metric_defs"]["f1"]["direction"]
            == MetricDirection.HIGHER.value
        )
        assert (
            report["regression_expectations"]["metric_defs"]["chromatin_consistency"]["direction"]
            == MetricDirection.HIGHER.value
        )

        peak_chromatin = report["tasks"]["contextual_peak_calling"]["variants"]["tf_plus_chromatin"]
        assert peak_chromatin["comparison_key"] == {
            "dataset": "synthetic_contextual_epigenomics",
            "task": "contextual_peak_calling",
            "contextual_variant": "tf_plus_chromatin",
        }

    def test_suite_run_persists_calibrax_variant_points(
        self,
        contextual_suite_report: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        saved_path, run = save_contextual_epigenomics_suite_run(
            contextual_suite_report,
            tmp_path / "contextual_store",
            commit="abc123",
            branch="main",
        )

        assert saved_path.exists()
        assert run.commit == "abc123"
        assert run.branch == "main"
        assert run.metadata["suite"] == "epigenomics/contextual_quick_suite"
        assert run.metadata["contextual_evidence_scope"]["promotion_eligible"] is False
        assert run.metric_defs["accuracy"].direction == MetricDirection.HIGHER
        assert run.metric_defs["f1"].direction == MetricDirection.HIGHER
        assert len(run.points) == 6

        point = next(
            point
            for point in run.points
            if point.tags["task"] == "contextual_peak_calling"
            and point.tags["contextual_variant"] == "tf_plus_chromatin"
        )
        assert point.name == "epigenomics/contextual_peak_calling"
        assert point.scenario == "synthetic_contextual_epigenomics"
        assert "f1" in point.metrics
        assert "chromatin_consistency" in point.metrics

    def test_suite_regression_check_uses_calibrax_guard(
        self,
        contextual_suite_report: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        store_path = tmp_path / "contextual_store"
        baseline = check_contextual_epigenomics_suite_regressions(
            contextual_suite_report,
            store_path,
            set_baseline_if_missing=True,
        )

        assert baseline.passed is True
        assert baseline.baseline_id == baseline.current_id

        current_report = deepcopy(contextual_suite_report)
        current_report["tasks"]["contextual_peak_calling"]["variants"]["tf_plus_chromatin"][
            "metrics"
        ]["f1"] = 0.0

        regression = check_contextual_epigenomics_suite_regressions(
            current_report,
            store_path,
        )

        assert regression.passed is False
        assert any(item.metric == "f1" for item in regression.regressions)
