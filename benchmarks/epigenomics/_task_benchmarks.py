"""Task wrappers and ablation reports for contextual epigenomics benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from calibrax.ci.guard import GuardResult
from calibrax.core.models import (
    MetricDef,
    MetricDirection,
    MetricPriority,
    Point,
    Run,
)
from calibrax.core.result import BenchmarkResult

from benchmarks._base import DiffBioBenchmarkConfig
from benchmarks._calibrax import (
    build_calibrax_metric_defs,
    build_calibrax_point,
    check_calibrax_suite_regressions,
    resolve_calibrax_threshold,
    save_calibrax_suite_run,
)
from benchmarks.epigenomics._contextual import (
    CONTEXTUAL_ABLATION_COMPARISON_AXES,
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
_CONTEXTUAL_REGRESSION_THRESHOLD = 0.05
_CONTEXTUAL_EVIDENCE_SCOPE = {
    "dataset": "synthetic_contextual_epigenomics",
    "source_type": "synthetic",
    "promotion_eligible": False,
    "stable_scope": "excluded",
    "reason": (
        "Synthetic contextual epigenomics ablations are regression evidence, "
        "not stable biological promotion evidence."
    ),
}
_CONTEXTUAL_METRIC_DEF_FACTORIES = {
    "precision": lambda: MetricDef(
        name="precision",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.SECONDARY,
        description="Position-wise peak-calling precision",
    ),
    "recall": lambda: MetricDef(
        name="recall",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.SECONDARY,
        description="Position-wise peak-calling recall",
    ),
    "f1": lambda: MetricDef(
        name="f1",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.PRIMARY,
        description="Position-wise peak-calling F1 score",
    ),
    "accuracy": lambda: MetricDef(
        name="accuracy",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.PRIMARY,
        description="Position-wise chromatin-state accuracy",
    ),
    "chromatin_consistency": lambda: MetricDef(
        name="chromatin_consistency",
        unit="",
        direction=MetricDirection.HIGHER,
        group="structure",
        priority=MetricPriority.SECONDARY,
        description="Bounded structural consistency with chromatin contacts",
    ),
}

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
    dataset = str(reference_result.tags["dataset"])
    task = str(reference_result.tags["task"])
    comparison_axes = _resolve_contextual_comparison_axes(results)
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
            "comparison_key": _extract_contextual_comparison_key(result),
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
        "dataset": dataset,
        "task": task,
        "comparison_axes": comparison_axes,
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
    regression_expectations = _build_contextual_regression_expectations(tasks, task_order)
    return {
        "suite": "epigenomics/contextual_quick_suite",
        "comparison_axes": list(CONTEXTUAL_ABLATION_COMPARISON_AXES),
        "task_order": task_order,
        "contextual_evidence_scope": dict(_CONTEXTUAL_EVIDENCE_SCOPE),
        "contextual_contract": _build_contextual_suite_contract(tasks, task_order),
        "regression_expectations": regression_expectations,
        "tasks": {task_name: tasks[task_name] for task_name in task_order},
    }


def build_contextual_epigenomics_suite_run(
    report: dict[str, Any],
    *,
    commit: str | None = None,
    branch: str | None = None,
    environment: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Run:
    """Convert a contextual ablation suite report into a Calibrax run."""
    regression_expectations = report.get("regression_expectations", {})
    metric_defs = build_calibrax_metric_defs(
        regression_expectations.get("metric_defs", {}),
        context="Contextual suite report",
    )

    points: list[Point] = []
    task_reports = report.get("tasks", {})
    if not isinstance(task_reports, dict):
        raise ValueError("Contextual suite report tasks must be a dict")

    for task_name in report.get("task_order", ()):
        task_report = task_reports.get(task_name)
        if not isinstance(task_report, dict):
            continue
        benchmark_name = str(task_report.get("benchmark", task_name))
        dataset_name = str(task_report.get("dataset", "unknown"))
        variants = task_report.get("variants", {})
        if not isinstance(variants, dict):
            continue

        for variant_name in task_report.get("variant_order", ()):
            variant_report = variants.get(variant_name)
            if not isinstance(variant_report, dict):
                continue
            points.append(
                build_calibrax_point(
                    name=benchmark_name,
                    scenario=dataset_name,
                    comparison_key=variant_report.get("comparison_key", {}),
                    metrics=variant_report.get("metrics", {}),
                    context=(f"Contextual suite variant report {benchmark_name}/{variant_name}"),
                )
            )

    run_metadata = {
        "suite": report.get("suite"),
        "comparison_axes": report.get("comparison_axes"),
        "task_order": report.get("task_order"),
        "contextual_contract": report.get("contextual_contract"),
        "contextual_evidence_scope": report.get("contextual_evidence_scope"),
        "regression_expectations": regression_expectations,
    }
    if metadata is not None:
        run_metadata.update(metadata)

    return Run(
        points=tuple(points),
        commit=commit,
        branch=branch,
        environment={} if environment is None else dict(environment),
        metadata=run_metadata,
        metric_defs=metric_defs,
    )


def save_contextual_epigenomics_suite_run(
    report: dict[str, Any],
    store_path: Path,
    *,
    commit: str | None = None,
    branch: str | None = None,
    environment: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Run]:
    """Persist a contextual ablation suite report into a Calibrax store."""
    return save_calibrax_suite_run(
        report,
        store_path,
        build_run=build_contextual_epigenomics_suite_run,
        commit=commit,
        branch=branch,
        environment=environment,
        metadata=metadata,
    )


def check_contextual_epigenomics_suite_regressions(
    report: dict[str, Any],
    store_path: Path,
    *,
    commit: str | None = None,
    branch: str | None = None,
    environment: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    set_baseline_if_missing: bool = False,
) -> GuardResult:
    """Persist one contextual suite run and compare it against a Calibrax baseline."""
    regression_expectations = report.get("regression_expectations", {})
    if not isinstance(regression_expectations, dict):
        raise ValueError("Contextual suite report regression_expectations must be a dict")
    threshold = resolve_calibrax_threshold(
        regression_expectations,
        default_threshold=_CONTEXTUAL_REGRESSION_THRESHOLD,
        context="Contextual suite report",
    )

    return check_calibrax_suite_regressions(
        report,
        store_path,
        build_run=build_contextual_epigenomics_suite_run,
        threshold=threshold,
        commit=commit,
        branch=branch,
        environment=environment,
        metadata=metadata,
        set_baseline_if_missing=set_baseline_if_missing,
    )


def _build_contextual_suite_contract(
    tasks: dict[str, Any],
    task_order: list[str],
) -> dict[str, Any]:
    """Build one suite-level contextual source contract summary."""
    required_keys: list[str] | None = None
    target_semantics_by_task: dict[str, str] = {}
    num_output_classes_by_task: dict[str, int] = {}

    for task_name in task_order:
        contract = tasks[task_name]["contract"]
        task_required_keys = list(contract["required_keys"])
        if required_keys is None:
            required_keys = task_required_keys
        elif task_required_keys != required_keys:
            raise ValueError(
                "Contextual epigenomics tasks disagree on required source keys: "
                f"{task_required_keys} != {required_keys}"
            )

        target_semantics_by_task[task_name] = str(contract["target_semantics"])
        num_output_classes_by_task[task_name] = int(contract["num_output_classes"])

    return {
        "required_keys": [] if required_keys is None else required_keys,
        "target_semantics_by_task": target_semantics_by_task,
        "num_output_classes_by_task": num_output_classes_by_task,
    }


def _resolve_contextual_comparison_axes(
    results: dict[str, BenchmarkResult],
) -> list[str]:
    """Require one contextual ablation comparison-axis convention."""
    expected_axes: list[str] = [str(axis) for axis in CONTEXTUAL_ABLATION_COMPARISON_AXES]
    observed_axes = {
        variant_name: result.metadata.get("comparison_axes")
        for variant_name, result in results.items()
    }
    if any(axes != expected_axes for axes in observed_axes.values()):
        raise ValueError(
            f"Contextual epigenomics results disagree on comparison_axes: {observed_axes}"
        )
    return expected_axes


def _extract_contextual_comparison_key(result: BenchmarkResult) -> dict[str, Any]:
    """Return the stored comparison key for one contextual variant."""
    comparison_key = result.metadata.get("comparison_key")
    if not isinstance(comparison_key, dict):
        raise ValueError(
            "Contextual epigenomics result metadata comparison_key must be a dict: "
            f"{result.name}/{result.tags.get('contextual_variant')}"
        )
    return {axis: comparison_key.get(axis) for axis in CONTEXTUAL_ABLATION_COMPARISON_AXES}


def _build_contextual_regression_expectations(
    tasks: dict[str, Any],
    task_order: list[str],
) -> dict[str, Any]:
    """Build Calibrax regression expectations for contextual ablation variants."""
    metric_defs = _build_contextual_metric_defs(tasks)
    return {
        "comparison_axes": list(CONTEXTUAL_ABLATION_COMPARISON_AXES),
        "task_order": task_order,
        "required_variants": {
            task_name: list(CONTEXTUAL_ABLATION_ORDER) for task_name in task_order
        },
        "metric_defs": {
            metric_name: metric_def.to_dict() for metric_name, metric_def in metric_defs.items()
        },
        "calibrax": {
            "baseline_name": "main",
            "threshold": _CONTEXTUAL_REGRESSION_THRESHOLD,
        },
    }


def _build_contextual_metric_defs(tasks: dict[str, Any]) -> dict[str, MetricDef]:
    """Build explicit Calibrax metric semantics for contextual ablation reports."""
    metric_names: set[str] = set()
    for task_report in tasks.values():
        variants = task_report.get("variants", {})
        if not isinstance(variants, dict):
            continue
        for variant_report in variants.values():
            if not isinstance(variant_report, dict):
                continue
            metrics = variant_report.get("metrics", {})
            if isinstance(metrics, dict):
                metric_names.update(str(metric_name) for metric_name in metrics)

    unknown_metrics = sorted(
        metric_name
        for metric_name in metric_names
        if metric_name not in _CONTEXTUAL_METRIC_DEF_FACTORIES
    )
    if unknown_metrics:
        raise ValueError(
            "Contextual epigenomics suite report includes metrics without "
            f"Calibrax semantics: {unknown_metrics}"
        )

    return {
        metric_name: _CONTEXTUAL_METRIC_DEF_FACTORIES[metric_name]()
        for metric_name in sorted(metric_names)
    }
