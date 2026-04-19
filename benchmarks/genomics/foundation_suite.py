"""Quick suite harness for imported genomics foundation models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from calibrax.core.result import BenchmarkResult

from benchmarks._foundation_models import (
    build_foundation_promotion_report,
    build_foundation_suite_report,
    check_foundation_suite_regressions,
)
from benchmarks.genomics._foundation import GENOMICS_FOUNDATION_SUITE_SCENARIOS
from benchmarks.genomics.bench_promoter import (
    build_foundation_promoter_report,
    run_foundation_promoter_suite,
)
from benchmarks.genomics.bench_splice_site import (
    build_foundation_splice_site_report,
    run_foundation_splice_site_suite,
)
from benchmarks.genomics.bench_tfbs import (
    build_foundation_tfbs_report,
    run_foundation_tfbs_suite,
)
from diffbio.operators.foundation_models import SequenceFoundationAdapter

_TASK_ORDER = ("promoter", "tfbs", "splice_site")


def run_genomics_foundation_suite(
    *,
    quick: bool = False,
    source_factories: dict[str, Callable[[int | None], Any]] | None = None,
    adapters: dict[str, SequenceFoundationAdapter] | None = None,
) -> dict[str, dict[str, BenchmarkResult]]:
    """Run the quick genomics suite across native and imported embeddings."""
    source_factories = {} if source_factories is None else dict(source_factories)
    return {
        "promoter": run_foundation_promoter_suite(
            quick=quick,
            source_factory=source_factories.get("promoter"),
            adapters=adapters,
        ),
        "tfbs": run_foundation_tfbs_suite(
            quick=quick,
            source_factory=source_factories.get("tfbs"),
            adapters=adapters,
        ),
        "splice_site": run_foundation_splice_site_suite(
            quick=quick,
            source_factory=source_factories.get("splice_site"),
            adapters=adapters,
        ),
    }


def build_genomics_foundation_suite_report(
    task_results: dict[str, dict[str, BenchmarkResult]],
) -> dict[str, Any]:
    """Build a deterministic report over the genomics quick suite."""
    task_reports: dict[str, Any] = {}

    if "promoter" in task_results:
        task_reports["promoter"] = build_foundation_promoter_report(task_results["promoter"])
    if "tfbs" in task_results:
        task_reports["tfbs"] = build_foundation_tfbs_report(task_results["tfbs"])
    if "splice_site" in task_results:
        task_reports["splice_site"] = build_foundation_splice_site_report(
            task_results["splice_site"]
        )

    suite_report = build_foundation_suite_report(
        suite_name="genomics/foundation_quick_suite",
        task_order=_TASK_ORDER,
        task_reports=task_reports,
        task_scenarios=GENOMICS_FOUNDATION_SUITE_SCENARIOS,
    )
    suite_report["dataset_provenance"] = {
        task_name: task_reports[task_name]["dataset_provenance"]
        for task_name in _TASK_ORDER
        if task_name in task_reports
    }
    return suite_report


def build_genomics_foundation_promotion_report(
    suite_report: dict[str, Any],
    store_path: Path,
    *,
    commit: str | None = None,
    branch: str | None = None,
    environment: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    set_baseline_if_missing: bool = False,
) -> dict[str, Any]:
    """Attach the Calibrax guard result to the genomics promotion report."""
    if suite_report.get("suite") != "genomics/foundation_quick_suite":
        raise ValueError("Genomics promotion reports require the genomics foundation suite.")

    guard_result = check_foundation_suite_regressions(
        suite_report,
        store_path,
        commit=commit,
        branch=branch,
        environment=environment,
        metadata=metadata,
        set_baseline_if_missing=set_baseline_if_missing,
    )
    return build_foundation_promotion_report(suite_report, guard_result=guard_result)
