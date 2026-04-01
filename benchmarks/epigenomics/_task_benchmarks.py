"""Task wrappers and ablation reports for contextual epigenomics benchmarks."""

from __future__ import annotations

from typing import Any, Protocol

from calibrax.core.result import BenchmarkResult

from benchmarks._base import DiffBioBenchmarkConfig
from benchmarks.epigenomics._contextual import (
    CONTEXTUAL_ABLATION_ORDER,
    ContextualEpigenomicsBenchmark,
    ContextualEpigenomicsTaskSpec,
)

_TASK_ORDER = (
    "contextual_peak_calling",
    "chromatin_state_prediction",
)
_TASK_METRIC_KEYS = {
    "contextual_peak_calling": ("precision", "recall", "f1", "chromatin_consistency"),
    "chromatin_state_prediction": ("accuracy", "chromatin_consistency"),
}
_DEFAULT_CONTEXTUAL_VARIANT = "tf_plus_chromatin"

_PEAK_CALLING_CONFIG = DiffBioBenchmarkConfig(
    name="epigenomics/contextual_peak_calling",
    domain="epigenomics",
    n_iterations_quick=5,
    n_iterations_full=20,
)
_PEAK_CALLING_SPEC = ContextualEpigenomicsTaskSpec(
    task_name="contextual_peak_calling",
    target_semantics="binary_peak_mask",
    num_output_classes=1,
)
_CHROMATIN_STATE_CONFIG = DiffBioBenchmarkConfig(
    name="epigenomics/chromatin_state_prediction",
    domain="epigenomics",
    n_iterations_quick=5,
    n_iterations_full=20,
)
_CHROMATIN_STATE_SPEC = ContextualEpigenomicsTaskSpec(
    task_name="chromatin_state_prediction",
    target_semantics="chromatin_state_id",
    num_output_classes=3,
)


class _ContextualSourceFactory(Protocol):
    """Protocol for task-specific benchmark source factories."""

    def __call__(self, subsample: int | None) -> Any:
        """Return a benchmark source with a ``load()`` method."""
        ...


class ContextualPeakCallingBenchmark(ContextualEpigenomicsBenchmark):
    """Context-aware peak-calling benchmark."""

    def __init__(
        self,
        *,
        quick: bool = False,
        variant_name: str = _DEFAULT_CONTEXTUAL_VARIANT,
        source_factory: _ContextualSourceFactory | None = None,
    ) -> None:
        super().__init__(
            _PEAK_CALLING_CONFIG,
            task_spec=_PEAK_CALLING_SPEC,
            quick=quick,
            variant_name=variant_name,
            source_factory=source_factory,
        )


class ChromatinStatePredictionBenchmark(ContextualEpigenomicsBenchmark):
    """Context-aware chromatin-state benchmark."""

    def __init__(
        self,
        *,
        quick: bool = False,
        variant_name: str = _DEFAULT_CONTEXTUAL_VARIANT,
        source_factory: _ContextualSourceFactory | None = None,
    ) -> None:
        super().__init__(
            _CHROMATIN_STATE_CONFIG,
            task_spec=_CHROMATIN_STATE_SPEC,
            quick=quick,
            variant_name=variant_name,
            source_factory=source_factory,
        )


def run_contextual_peak_calling_benchmark(
    *,
    quick: bool = False,
    variant_name: str = _DEFAULT_CONTEXTUAL_VARIANT,
    source_factory: _ContextualSourceFactory | None = None,
) -> BenchmarkResult:
    """Run the contextual peak-calling benchmark for one variant."""
    return ContextualPeakCallingBenchmark(
        quick=quick,
        variant_name=variant_name,
        source_factory=source_factory,
    ).run()


def run_chromatin_state_prediction_benchmark(
    *,
    quick: bool = False,
    variant_name: str = _DEFAULT_CONTEXTUAL_VARIANT,
    source_factory: _ContextualSourceFactory | None = None,
) -> BenchmarkResult:
    """Run the chromatin-state benchmark for one variant."""
    return ChromatinStatePredictionBenchmark(
        quick=quick,
        variant_name=variant_name,
        source_factory=source_factory,
    ).run()


def run_contextual_peak_calling_ablation_suite(
    *,
    quick: bool = False,
    source_factory: _ContextualSourceFactory | None = None,
) -> dict[str, BenchmarkResult]:
    """Run the contextual peak-calling benchmark across all ablations."""
    return {
        variant_name: run_contextual_peak_calling_benchmark(
            quick=quick,
            variant_name=variant_name,
            source_factory=source_factory,
        )
        for variant_name in CONTEXTUAL_ABLATION_ORDER
    }


def run_chromatin_state_prediction_ablation_suite(
    *,
    quick: bool = False,
    source_factory: _ContextualSourceFactory | None = None,
) -> dict[str, BenchmarkResult]:
    """Run the chromatin-state benchmark across all ablations."""
    return {
        variant_name: run_chromatin_state_prediction_benchmark(
            quick=quick,
            variant_name=variant_name,
            source_factory=source_factory,
        )
        for variant_name in CONTEXTUAL_ABLATION_ORDER
    }


def build_contextual_task_report(
    *,
    benchmark_name: str,
    results: dict[str, BenchmarkResult],
) -> dict[str, Any]:
    """Build a deterministic ablation report for one contextual benchmark."""
    variant_order = [variant for variant in CONTEXTUAL_ABLATION_ORDER if variant in results]
    reference_result = results["sequence_only"]
    reference_metrics = {
        key: float(reference_result.metrics[key].value)
        for key in _TASK_METRIC_KEYS[reference_result.tags["task"]]
        if key in reference_result.metrics
    }

    variants: dict[str, Any] = {}
    deltas_from_sequence_only: dict[str, dict[str, float]] = {}

    for variant_name in variant_order:
        result = results[variant_name]
        metric_keys = _TASK_METRIC_KEYS[result.tags["task"]]
        metrics = {
            key: float(result.metrics[key].value) for key in metric_keys if key in result.metrics
        }
        variants[variant_name] = {
            "metrics": metrics,
            "tags": {
                "dataset": result.tags["dataset"],
                "task": result.tags["task"],
                "operator": result.tags["operator"],
                "contextual_variant": result.tags["contextual_variant"],
            },
            "metadata": {
                "ablation": result.metadata["ablation"],
                "training": result.metadata["training"],
            },
        }

        if variant_name == "sequence_only":
            continue

        deltas_from_sequence_only[variant_name] = {
            metric_name: metrics[metric_name] - reference_metrics[metric_name]
            for metric_name in metrics
            if metric_name in reference_metrics
        }

    first_result = next(iter(results.values()))
    primary_metric = "f1" if first_result.tags["task"] == "contextual_peak_calling" else "accuracy"
    best_variant_by_metric = {
        primary_metric: max(
            variant_order,
            key=lambda variant_name: variants[variant_name]["metrics"][primary_metric],
        ),
        "chromatin_consistency": max(
            variant_order,
            key=lambda variant_name: variants[variant_name]["metrics"]["chromatin_consistency"],
        ),
    }

    return {
        "benchmark": benchmark_name,
        "contract": first_result.metadata["contextual_contract"],
        "variant_order": variant_order,
        "primary_metric": primary_metric,
        "best_variant_by_metric": best_variant_by_metric,
        "variants": variants,
        "deltas_from_sequence_only": deltas_from_sequence_only,
    }


def run_contextual_epigenomics_suite(
    *,
    quick: bool = False,
    source_factories: dict[str, _ContextualSourceFactory] | None = None,
) -> dict[str, dict[str, BenchmarkResult]]:
    """Run both contextual epigenomics tasks across all ablations."""
    source_factories = {} if source_factories is None else dict(source_factories)
    return {
        "contextual_peak_calling": run_contextual_peak_calling_ablation_suite(
            quick=quick,
            source_factory=source_factories.get("contextual_peak_calling"),
        ),
        "chromatin_state_prediction": run_chromatin_state_prediction_ablation_suite(
            quick=quick,
            source_factory=source_factories.get("chromatin_state_prediction"),
        ),
    }


def build_contextual_epigenomics_suite_report(
    task_results: dict[str, dict[str, BenchmarkResult]],
) -> dict[str, Any]:
    """Build a deterministic quick-suite report for contextual epigenomics."""
    tasks: dict[str, Any] = {}

    if "contextual_peak_calling" in task_results:
        tasks["contextual_peak_calling"] = build_contextual_task_report(
            benchmark_name=_PEAK_CALLING_CONFIG.name,
            results=task_results["contextual_peak_calling"],
        )
    if "chromatin_state_prediction" in task_results:
        tasks["chromatin_state_prediction"] = build_contextual_task_report(
            benchmark_name=_CHROMATIN_STATE_CONFIG.name,
            results=task_results["chromatin_state_prediction"],
        )

    task_order = [task_name for task_name in _TASK_ORDER if task_name in tasks]
    return {
        "suite": "epigenomics/contextual_quick_suite",
        "task_order": task_order,
        "tasks": {task_name: tasks[task_name] for task_name in task_order},
    }
