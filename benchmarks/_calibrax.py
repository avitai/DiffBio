"""Calibrax-owned storage and regression helpers for benchmark suites."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from calibrax.ci.guard import CIGuard, GuardResult
from calibrax.core.models import Metric, MetricDef, Point, Run
from calibrax.core.result import BenchmarkResult
from calibrax.profiling.hardware import detect_hardware_specs
from calibrax.profiling.timing import TimingCollector
from calibrax.storage.store import Store


class CalibraxRunBuilder(Protocol):
    """Callable that converts a suite report into a Calibrax run."""

    def __call__(
        self,
        report: dict[str, Any],
        *,
        commit: str | None = None,
        branch: str | None = None,
        environment: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Run:
        """Build a Calibrax run from a suite report."""
        ...


def build_calibrax_metric_defs(
    metric_defs_payload: object,
    *,
    context: str,
) -> dict[str, MetricDef]:
    """Convert serialized metric definitions through Calibrax model types."""
    if not isinstance(metric_defs_payload, dict):
        raise ValueError(f"{context} regression_expectations.metric_defs must be a dict")

    return {
        str(metric_name): MetricDef.from_dict(metric_payload)
        for metric_name, metric_payload in metric_defs_payload.items()
        if isinstance(metric_payload, dict)
    }


def build_calibrax_point(
    *,
    name: str,
    scenario: str,
    comparison_key: object,
    metrics: object,
    context: str,
) -> Point:
    """Build one Calibrax point from a benchmark report row."""
    if not isinstance(comparison_key, dict):
        raise ValueError(f"{context} comparison_key must be a dict")
    if not isinstance(metrics, dict):
        raise ValueError(f"{context} metrics must be a dict")

    return Point(
        name=name,
        scenario=scenario,
        tags={axis: str(value) for axis, value in comparison_key.items() if value is not None},
        metrics={
            str(metric_name): Metric(value=float(metric_value))
            for metric_name, metric_value in metrics.items()
        },
    )


def resolve_calibrax_threshold(
    regression_expectations: dict[str, Any],
    *,
    default_threshold: float,
    context: str,
) -> float:
    """Resolve one Calibrax regression threshold from suite expectations."""
    calibrax_policy = regression_expectations.get("calibrax", {})
    if not isinstance(calibrax_policy, dict):
        raise ValueError(f"{context} regression_expectations.calibrax must be a dict")
    return float(calibrax_policy.get("threshold", default_threshold))


def save_calibrax_run(run: Run, store_path: Path) -> Path:
    """Persist one Calibrax run through the shared Store boundary."""
    store = Store(store_path)
    return store.save(run)


def measure_calibrax_throughput(
    *,
    iterate_fn: Callable[[], Any],
    n_items: int,
    n_iterations: int,
) -> Any:
    """Measure benchmark throughput through the Calibrax profiling boundary."""
    collector = TimingCollector(warmup_iterations=3)
    return collector.measure_iteration(
        iterator=iter(range(n_iterations)),
        num_batches=n_iterations,
        process_fn=lambda _: iterate_fn(),
        count_fn=lambda _: n_items,
    )


def build_calibrax_benchmark_run(results: Sequence[BenchmarkResult]) -> Run:
    """Build one Calibrax run from benchmark results for the shared runner."""
    points = tuple(
        Point(
            name=result.name,
            scenario=result.tags.get("dataset", "unknown"),
            tags=result.tags,
            metrics=result.metrics,
        )
        for result in results
    )
    return Run(points=points, environment=detect_hardware_specs())


def save_calibrax_suite_run(
    report: dict[str, Any],
    store_path: Path,
    *,
    build_run: CalibraxRunBuilder,
    commit: str | None = None,
    branch: str | None = None,
    environment: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Run]:
    """Build and persist one suite run through the shared Calibrax boundary."""
    run = build_run(
        report,
        commit=commit,
        branch=branch,
        environment=environment,
        metadata=metadata,
    )
    return save_calibrax_run(run, store_path), run


def check_calibrax_suite_regressions(
    report: dict[str, Any],
    store_path: Path,
    *,
    build_run: CalibraxRunBuilder,
    threshold: float,
    commit: str | None = None,
    branch: str | None = None,
    environment: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    set_baseline_if_missing: bool = False,
) -> GuardResult:
    """Persist one suite run and compare it against a Calibrax baseline."""
    store = Store(store_path)
    run = build_run(
        report,
        commit=commit,
        branch=branch,
        environment=environment,
        metadata=metadata,
    )
    store.save(run)

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
