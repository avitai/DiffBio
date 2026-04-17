"""Full quick-suite harness for imported single-cell foundation models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell._foundation import SINGLECELL_FOUNDATION_SUITE_SCENARIOS
from benchmarks.singlecell.bench_batch_correction import (
    build_foundation_batch_correction_report,
    run_foundation_batch_correction_suite,
)
from benchmarks.singlecell.bench_foundation_annotation import (
    build_foundation_annotation_report,
    run_foundation_annotation_suite,
)
from diffbio.operators.foundation_models import SingleCellPrecomputedAdapter

_TASK_ORDER = ("cell_annotation", "batch_correction")


def run_singlecell_foundation_suite(
    *,
    quick: bool = False,
    data_dir: str = "/media/mahdi/ssd23/Data/scib",
    source_factory: Callable[[int | None], Any] | None = None,
    adapters: dict[str, SingleCellPrecomputedAdapter] | None = None,
) -> dict[str, dict[str, BenchmarkResult]]:
    """Run the quick single-cell suite across native and imported embeddings."""
    return {
        "cell_annotation": run_foundation_annotation_suite(
            quick=quick,
            data_dir=data_dir,
            source_factory=source_factory,
            adapters=adapters,
        ),
        "batch_correction": run_foundation_batch_correction_suite(
            quick=quick,
            data_dir=data_dir,
            source_factory=source_factory,
            adapters=adapters,
        ),
    }


def build_singlecell_foundation_suite_report(
    task_results: dict[str, dict[str, BenchmarkResult]],
) -> dict[str, Any]:
    """Build a deterministic report over the quick single-cell suite."""
    tasks: dict[str, Any] = {}

    if "cell_annotation" in task_results:
        tasks["cell_annotation"] = build_foundation_annotation_report(
            task_results["cell_annotation"]
        )
    if "batch_correction" in task_results:
        tasks["batch_correction"] = build_foundation_batch_correction_report(
            task_results["batch_correction"]
        )

    task_order = [task_name for task_name in _TASK_ORDER if task_name in tasks]
    comparison_axes = next(
        (
            task_report["comparison_axes"]
            for task_name in task_order
            if (task_report := tasks[task_name]).get("comparison_axes")
        ),
        ["dataset", "task"],
    )

    return {
        "suite": "singlecell/foundation_quick_suite",
        "comparison_axes": comparison_axes,
        "task_order": task_order,
        "suite_scenarios": {
            task_name: SINGLECELL_FOUNDATION_SUITE_SCENARIOS[task_name] for task_name in task_order
        },
        "tasks": tasks,
    }
