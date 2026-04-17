"""Full quick-suite harness for imported single-cell foundation models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrax.core.result import BenchmarkResult

from benchmarks._foundation_models import build_foundation_suite_report
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
    task_reports: dict[str, Any] = {}

    if "cell_annotation" in task_results:
        task_reports["cell_annotation"] = build_foundation_annotation_report(
            task_results["cell_annotation"]
        )
    if "batch_correction" in task_results:
        task_reports["batch_correction"] = build_foundation_batch_correction_report(
            task_results["batch_correction"]
        )

    return build_foundation_suite_report(
        suite_name="singlecell/foundation_quick_suite",
        task_order=_TASK_ORDER,
        task_reports=task_reports,
        task_scenarios=SINGLECELL_FOUNDATION_SUITE_SCENARIOS,
    )
