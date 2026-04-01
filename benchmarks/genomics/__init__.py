"""Genomics foundation-model benchmark scaffolds."""

from benchmarks.genomics.bench_promoter import (
    PromoterBenchmark,
    build_foundation_promoter_report,
    run_foundation_promoter_suite,
)
from benchmarks.genomics.bench_splice_site import (
    SpliceSiteBenchmark,
    build_foundation_splice_site_report,
    run_foundation_splice_site_suite,
)
from benchmarks.genomics.bench_tfbs import (
    TFBSBenchmark,
    build_foundation_tfbs_report,
    run_foundation_tfbs_suite,
)
from benchmarks.genomics.foundation_suite import (
    build_genomics_foundation_suite_report,
    run_genomics_foundation_suite,
)

__all__ = [
    "PromoterBenchmark",
    "SpliceSiteBenchmark",
    "TFBSBenchmark",
    "build_foundation_promoter_report",
    "build_foundation_splice_site_report",
    "build_foundation_tfbs_report",
    "build_genomics_foundation_suite_report",
    "run_foundation_promoter_suite",
    "run_foundation_splice_site_suite",
    "run_foundation_tfbs_suite",
    "run_genomics_foundation_suite",
]
