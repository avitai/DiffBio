"""Splice-site classification benchmark for genomics foundation models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrax.core.result import BenchmarkResult

from benchmarks._base import DiffBioBenchmarkConfig
from benchmarks.genomics._foundation import (
    build_genomics_foundation_task_report,
    run_genomics_foundation_benchmark_suite,
)
from benchmarks.genomics._sequence_classification import (
    SequenceClassificationBenchmark,
    SequenceTaskSpec,
)
from diffbio.operators.foundation_models import SequencePrecomputedAdapter

_TASK_SPEC = SequenceTaskSpec(
    benchmark_name="genomics/splice_site",
    task_name="splice_site",
)
_CONFIG = DiffBioBenchmarkConfig(
    name=_TASK_SPEC.benchmark_name,
    domain="genomics",
    quick_subsample=24,
)


class SpliceSiteBenchmark(SequenceClassificationBenchmark):
    """Evaluate splice-site classification on sequence embeddings."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        source_factory: Callable[[int | None], Any] | None = None,
        embedding_adapter: SequencePrecomputedAdapter | None = None,
    ) -> None:
        super().__init__(
            config,
            task_spec=_TASK_SPEC,
            quick=quick,
            source_factory=source_factory,
            embedding_adapter=embedding_adapter,
        )


def run_foundation_splice_site_suite(
    *,
    quick: bool = False,
    source_factory: Callable[[int | None], Any] | None = None,
    adapters: dict[str, SequencePrecomputedAdapter] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run the splice-site benchmark across native and imported model families."""
    return run_genomics_foundation_benchmark_suite(
        benchmark_factory=lambda embedding_adapter: SpliceSiteBenchmark(
            quick=quick,
            source_factory=source_factory,
            embedding_adapter=embedding_adapter,
        ),
        adapters=adapters,
    )


def build_foundation_splice_site_report(results: dict[str, BenchmarkResult]) -> dict[str, Any]:
    """Build a deterministic comparison report for splice-site runs."""
    return build_genomics_foundation_task_report(
        benchmark_name=_CONFIG.name,
        results=results,
        metric_keys=("accuracy", "macro_f1", "train_loss"),
    )
