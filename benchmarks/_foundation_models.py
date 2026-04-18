"""Shared helpers for foundation-model benchmark suites."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

from calibrax.ci.guard import CIGuard, GuardResult
from calibrax.core.models import (
    Metric,
    MetricDef,
    MetricDirection,
    MetricPriority,
    Point,
    Run,
)
from calibrax.core.result import BenchmarkResult
from calibrax.storage.store import Store

from benchmarks._base import build_benchmark_comparison_key
from diffbio.operators.foundation_models.contracts import FOUNDATION_BENCHMARK_COMPARISON_AXES

DEFAULT_FOUNDATION_REPORT_METADATA_KEYS = (
    "embedding_source",
    "foundation_source_name",
    "requires_batch_context",
    "batch_key",
    "context_version",
)
DEFAULT_FOUNDATION_REGRESSION_THRESHOLD = 0.05
_FOUNDATION_METRIC_DEF_FACTORIES: dict[str, Callable[[], MetricDef]] = {
    "accuracy": lambda: MetricDef(
        name="accuracy",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.PRIMARY,
        description="Classification accuracy",
    ),
    "macro_f1": lambda: MetricDef(
        name="macro_f1",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.PRIMARY,
        description="Macro-averaged F1 score",
    ),
    "train_loss": lambda: MetricDef(
        name="train_loss",
        unit="",
        direction=MetricDirection.LOWER,
        group="optimization",
        priority=MetricPriority.SECONDARY,
        description="Probe training loss",
    ),
    "aggregate_score": lambda: MetricDef(
        name="aggregate_score",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.PRIMARY,
        description="Aggregate scIB integration score",
    ),
    "silhouette_label": lambda: MetricDef(
        name="silhouette_label",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.SECONDARY,
        description="Cell-type silhouette score",
    ),
    "nmi_kmeans": lambda: MetricDef(
        name="nmi_kmeans",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.SECONDARY,
        description="K-means normalized mutual information",
    ),
    "ari_kmeans": lambda: MetricDef(
        name="ari_kmeans",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.SECONDARY,
        description="K-means adjusted Rand index",
    ),
    "clisi": lambda: MetricDef(
        name="clisi",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.SECONDARY,
        description="Cell-type local inverse Simpson's index",
    ),
    "isolated_labels": lambda: MetricDef(
        name="isolated_labels",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.SECONDARY,
        description="Isolated-label conservation score",
    ),
    "silhouette_batch": lambda: MetricDef(
        name="silhouette_batch",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.SECONDARY,
        description="Batch-mixing silhouette score",
    ),
    "ilisi": lambda: MetricDef(
        name="ilisi",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.SECONDARY,
        description="Integration local inverse Simpson's index",
    ),
    "graph_connectivity": lambda: MetricDef(
        name="graph_connectivity",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.SECONDARY,
        description="Graph connectivity score",
    ),
    "bio_score": lambda: MetricDef(
        name="bio_score",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.SECONDARY,
        description="Biological conservation score",
    ),
    "batch_score": lambda: MetricDef(
        name="batch_score",
        unit="",
        direction=MetricDirection.HIGHER,
        group="quality",
        priority=MetricPriority.SECONDARY,
        description="Batch-removal score",
    ),
}


class _BenchmarkRunner(Protocol):
    """Minimal benchmark runner protocol for suite helpers."""

    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return a BenchmarkResult."""
        ...


AdapterT = TypeVar("AdapterT")


def run_foundation_benchmark_suite(
    *,
    baseline_families: tuple[str, ...],
    benchmark_factory: Callable[[AdapterT | None], _BenchmarkRunner],
    adapters: dict[str, AdapterT] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run one benchmark across native and imported model families."""
    results = {"diffbio_native": benchmark_factory(None).run()}

    for baseline_name in baseline_families:
        if baseline_name == "diffbio_native":
            continue
        if adapters is None or baseline_name not in adapters:
            continue

        results[baseline_name] = benchmark_factory(adapters[baseline_name]).run()

    return results


def _get_shared_string_tag(
    results: dict[str, BenchmarkResult],
    *,
    key: str,
) -> str:
    """Require one shared string tag value across a report's model results."""
    values = {model_name: result.tags.get(key) for model_name, result in results.items()}
    unique_values = {value for value in values.values() if isinstance(value, str)}
    if len(unique_values) != 1:
        raise ValueError(f"Foundation benchmark results disagree on shared tag {key!r}: {values}")
    return unique_values.pop()


def _resolve_foundation_comparison_axes(
    results: dict[str, BenchmarkResult],
) -> list[str]:
    """Resolve one shared comparison-axis convention across model results."""
    rich_axes: list[tuple[str, ...]] = []

    for result in results.values():
        comparison_axes = result.metadata.get("comparison_axes")
        if not isinstance(comparison_axes, list | tuple):
            continue
        axes_tuple = tuple(str(axis) for axis in comparison_axes)
        if len(axes_tuple) > 2:
            rich_axes.append(axes_tuple)

    if not rich_axes:
        return ["dataset", "task"]

    canonical_axes = rich_axes[0]
    if any(axes != canonical_axes for axes in rich_axes[1:]):
        raise ValueError(
            "Foundation benchmark results disagree on comparison_axes: "
            f"{[list(axes) for axes in rich_axes]}"
        )

    return list(canonical_axes)


def _extract_foundation_provenance(result: BenchmarkResult) -> dict[str, Any] | None:
    """Return normalized foundation provenance when present on a result."""
    foundation_model = result.metadata.get("foundation_model")
    if not isinstance(foundation_model, dict):
        return None
    return dict(foundation_model)


def _build_foundation_metric_defs(
    task_reports: dict[str, dict[str, Any]],
) -> dict[str, MetricDef]:
    """Build explicit Calibrax metric semantics for stored suite reports."""
    metric_names: set[str] = set()
    for task_report in task_reports.values():
        models = task_report.get("models", {})
        if not isinstance(models, dict):
            continue
        for model_report in models.values():
            metrics = model_report.get("metrics", {})
            if isinstance(metrics, dict):
                metric_names.update(str(metric_name) for metric_name in metrics)

    unknown_metrics = sorted(
        metric_name
        for metric_name in metric_names
        if metric_name not in _FOUNDATION_METRIC_DEF_FACTORIES
    )
    if unknown_metrics:
        raise ValueError(
            "Foundation suite report includes metrics without Calibrax semantics: "
            f"{unknown_metrics}"
        )

    return {
        metric_name: _FOUNDATION_METRIC_DEF_FACTORIES[metric_name]()
        for metric_name in sorted(metric_names)
    }


def build_foundation_task_report(
    *,
    benchmark_name: str,
    results: dict[str, BenchmarkResult],
    metric_keys: tuple[str, ...],
    baseline_families: tuple[str, ...],
    metadata_keys: tuple[str, ...] = DEFAULT_FOUNDATION_REPORT_METADATA_KEYS,
) -> dict[str, Any]:
    """Build a deterministic comparison report for one benchmark task."""
    model_order = [name for name in baseline_families if name in results]
    dataset = _get_shared_string_tag(results, key="dataset")
    task = _get_shared_string_tag(results, key="task")
    comparison_axes = _resolve_foundation_comparison_axes(results)
    models: dict[str, Any] = {}

    for model_name in model_order:
        result = results[model_name]
        foundation_model = _extract_foundation_provenance(result)
        metrics = {
            key: float(result.metrics[key].value) for key in metric_keys if key in result.metrics
        }
        tags = {
            key: result.tags[key]
            for key in FOUNDATION_BENCHMARK_COMPARISON_AXES
            if key in result.tags
        }
        metadata = {key: result.metadata[key] for key in metadata_keys if key in result.metadata}
        models[model_name] = {
            "metrics": metrics,
            "tags": tags,
            "metadata": metadata,
            "foundation_model": foundation_model,
            "comparison_key": build_benchmark_comparison_key(
                comparison_axes=comparison_axes,
                tags=result.tags,
            ),
        }

    return {
        "benchmark": benchmark_name,
        "dataset": dataset,
        "task": task,
        "comparison_axes": comparison_axes,
        "model_order": model_order,
        "models": models,
    }


def build_foundation_suite_report(
    *,
    suite_name: str,
    task_order: tuple[str, ...],
    task_reports: dict[str, dict[str, Any]],
    task_scenarios: dict[str, Any],
    deferred_tasks: dict[str, Any] | None = None,
    relative_regression_threshold: float = DEFAULT_FOUNDATION_REGRESSION_THRESHOLD,
) -> dict[str, Any]:
    """Build one deterministic suite report plus regression expectations."""
    ordered_tasks = [task_name for task_name in task_order if task_name in task_reports]
    deferred_task_payload: dict[str, Any] = {}
    if deferred_tasks is not None:
        overlapping_tasks = sorted(set(ordered_tasks) & set(deferred_tasks))
        if overlapping_tasks:
            raise ValueError(
                f"Foundation suite deferred_tasks overlap executed tasks: {overlapping_tasks}"
            )
        deferred_task_payload = {
            task_name: deferred_tasks[task_name] for task_name in sorted(deferred_tasks)
        }
    comparison_axes = next(
        (
            report["comparison_axes"]
            for task_name in ordered_tasks
            if (report := task_reports[task_name]).get("comparison_axes")
        ),
        ["dataset", "task"],
    )
    metric_defs = _build_foundation_metric_defs(task_reports)

    regression_expectations = {
        "comparison_axes": list(comparison_axes),
        "task_order": ordered_tasks,
        "required_models": {
            task_name: list(task_reports[task_name].get("model_order", ()))
            for task_name in ordered_tasks
        },
        "metric_defs": {
            metric_name: metric_def.to_dict() for metric_name, metric_def in metric_defs.items()
        },
        "calibrax": {
            "baseline_name": "main",
            "threshold": float(relative_regression_threshold),
        },
    }

    return {
        "suite": suite_name,
        "comparison_axes": comparison_axes,
        "task_order": ordered_tasks,
        "suite_scenarios": {task_name: task_scenarios[task_name] for task_name in ordered_tasks},
        "deferred_tasks": deferred_task_payload,
        "regression_expectations": regression_expectations,
        "tasks": {task_name: task_reports[task_name] for task_name in ordered_tasks},
    }


def build_foundation_suite_run(
    report: dict[str, Any],
    *,
    commit: str | None = None,
    branch: str | None = None,
    environment: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Run:
    """Convert a deterministic suite report into a Calibrax Run."""
    regression_expectations = report.get("regression_expectations", {})
    metric_defs_payload = regression_expectations.get("metric_defs", {})
    if not isinstance(metric_defs_payload, dict):
        raise ValueError(
            "Foundation suite report regression_expectations.metric_defs must be a dict"
        )

    metric_defs = {
        metric_name: MetricDef.from_dict(metric_payload)
        for metric_name, metric_payload in metric_defs_payload.items()
        if isinstance(metric_payload, dict)
    }

    points: list[Point] = []
    for task_name in report.get("task_order", ()):
        task_report = report.get("tasks", {}).get(task_name)
        if not isinstance(task_report, dict):
            continue
        benchmark_name = str(task_report.get("benchmark", task_name))
        dataset_name = str(task_report.get("dataset", "unknown"))
        models = task_report.get("models", {})
        if not isinstance(models, dict):
            continue

        for model_name in task_report.get("model_order", ()):
            model_report = models.get(model_name)
            if not isinstance(model_report, dict):
                continue

            comparison_key = model_report.get("comparison_key", {})
            if not isinstance(comparison_key, dict):
                raise ValueError(
                    "Foundation suite model report comparison_key must be a dict: "
                    f"{benchmark_name}/{model_name}"
                )

            metrics = model_report.get("metrics", {})
            if not isinstance(metrics, dict):
                raise ValueError(
                    "Foundation suite model report metrics must be a dict: "
                    f"{benchmark_name}/{model_name}"
                )

            tags = {axis: str(value) for axis, value in comparison_key.items() if value is not None}
            points.append(
                Point(
                    name=benchmark_name,
                    scenario=dataset_name,
                    tags=tags,
                    metrics={
                        metric_name: Metric(value=float(metric_value))
                        for metric_name, metric_value in metrics.items()
                    },
                )
            )

    run_metadata = {
        "suite": report.get("suite"),
        "comparison_axes": report.get("comparison_axes"),
        "task_order": report.get("task_order"),
        "deferred_tasks": report.get("deferred_tasks", {}),
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


def save_foundation_suite_run(
    report: dict[str, Any],
    store_path: Path,
    *,
    commit: str | None = None,
    branch: str | None = None,
    environment: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Run]:
    """Persist a deterministic suite report into a Calibrax Store."""
    store = Store(store_path)
    run = build_foundation_suite_run(
        report,
        commit=commit,
        branch=branch,
        environment=environment,
        metadata=metadata,
    )
    return store.save(run), run


def check_foundation_suite_regressions(
    report: dict[str, Any],
    store_path: Path,
    *,
    commit: str | None = None,
    branch: str | None = None,
    environment: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    set_baseline_if_missing: bool = False,
) -> GuardResult:
    """Persist one suite run and compare it against the stored Calibrax baseline."""
    store = Store(store_path)
    _, run = save_foundation_suite_run(
        report,
        store_path,
        commit=commit,
        branch=branch,
        environment=environment,
        metadata=metadata,
    )

    regression_expectations = report.get("regression_expectations", {})
    calibrax_policy = regression_expectations.get("calibrax", {})
    if not isinstance(calibrax_policy, dict):
        raise ValueError("Foundation suite report regression_expectations.calibrax must be a dict")
    threshold = float(calibrax_policy.get("threshold", DEFAULT_FOUNDATION_REGRESSION_THRESHOLD))

    baseline = store.get_baseline()
    if baseline is None:
        if not set_baseline_if_missing:
            raise FileNotFoundError("No baseline set. Use store.set_baseline() first.")
        store.set_baseline(run.id)
        return GuardResult(
            passed=True,
            regressions=(),
            threshold=threshold,
            baseline_id=run.id,
            current_id=run.id,
        )

    guard = CIGuard(store, threshold=threshold)
    return guard.check(run.id)


def save_foundation_suite_report(
    report: dict[str, Any],
    output_path: Path,
) -> Path:
    """Persist one foundation-suite report as canonical JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
