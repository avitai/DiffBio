"""Shared helpers for foundation-model benchmark suites."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from calibrax.core.result import BenchmarkResult

_FOUNDATION_REPORT_TAG_KEYS = (
    "dataset",
    "task",
    "model_family",
    "adapter_mode",
    "artifact_id",
    "preprocessing_version",
)
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
    models: dict[str, Any] = {}

    for model_name in model_order:
        result = results[model_name]
        metrics = {
            key: float(result.metrics[key].value) for key in metric_keys if key in result.metrics
        }
        tags = {key: result.tags[key] for key in _FOUNDATION_REPORT_TAG_KEYS if key in result.tags}
        metadata = {key: result.metadata[key] for key in metadata_keys if key in result.metadata}
        models[model_name] = {
            "metrics": metrics,
            "tags": tags,
            "metadata": metadata,
        }

    dataset = next(iter(results.values())).tags["dataset"]
    task = next(iter(results.values())).tags["task"]

    return {
        "benchmark": benchmark_name,
        "dataset": dataset,
        "task": task,
        "model_order": model_order,
        "models": models,
    }
