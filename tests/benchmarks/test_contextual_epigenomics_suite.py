"""Tests for the contextual epigenomics benchmark and ablation suite."""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.epigenomics.bench_chromatin_state_prediction import (
    ChromatinStatePredictionBenchmark,
    run_chromatin_state_prediction_ablation_suite,
)
from benchmarks.epigenomics.bench_contextual_peak_calling import (
    ContextualPeakCallingBenchmark,
    run_contextual_peak_calling_ablation_suite,
)
from benchmarks.epigenomics.contextual_suite import (
    build_contextual_epigenomics_suite_report,
    run_contextual_epigenomics_suite,
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
        assert result.metadata["ablation"]["use_tf_context"] is True
        assert result.metadata["ablation"]["use_chromatin_guidance"] is True

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
