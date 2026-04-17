"""Shared helpers for foundation-model benchmark suites."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from calibrax.core.result import BenchmarkResult

from benchmarks._base import build_benchmark_comparison_key
from diffbio.operators.foundation_models.contracts import FOUNDATION_BENCHMARK_COMPARISON_AXES

DEFAULT_FOUNDATION_REPORT_METADATA_KEYS = (
    "embedding_source",
    "foundation_source_name",
    "requires_batch_context",
    "batch_key",
    "context_version",
)


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
