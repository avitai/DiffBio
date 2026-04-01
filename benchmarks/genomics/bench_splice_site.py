"""Splice-site classification benchmark for genomics foundation models."""

from benchmarks.genomics._task_benchmarks import (
    SpliceSiteBenchmark,
    build_foundation_splice_site_report,
    run_foundation_splice_site_suite,
)

__all__ = [
    "SpliceSiteBenchmark",
    "build_foundation_splice_site_report",
    "run_foundation_splice_site_suite",
]
