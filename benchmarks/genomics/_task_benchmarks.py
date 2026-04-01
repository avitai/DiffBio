"""Shared task wrappers for genomics foundation-model benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

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
from diffbio.operators.foundation_models import SequenceFoundationAdapter

_METRIC_KEYS = ("accuracy", "macro_f1", "train_loss")


class _BenchmarkFactory(Protocol):
    """Callable benchmark factory with task-specific defaults."""

    def __call__(
        self,
        *,
        quick: bool = False,
        source_factory: Callable[[int | None], Any] | None = None,
        embedding_adapter: SequenceFoundationAdapter | None = None,
    ) -> SequenceClassificationBenchmark:
        """Build one task benchmark instance."""
        ...


_PROMOTER_TASK_SPEC = SequenceTaskSpec(
    benchmark_name="genomics/promoter",
    task_name="promoter",
)
_PROMOTER_CONFIG = DiffBioBenchmarkConfig(
    name=_PROMOTER_TASK_SPEC.benchmark_name,
    domain="genomics",
    quick_subsample=24,
)

_TFBS_TASK_SPEC = SequenceTaskSpec(
    benchmark_name="genomics/tfbs",
    task_name="tfbs",
)
_TFBS_CONFIG = DiffBioBenchmarkConfig(
    name=_TFBS_TASK_SPEC.benchmark_name,
    domain="genomics",
    quick_subsample=24,
)

_SPLICE_SITE_TASK_SPEC = SequenceTaskSpec(
    benchmark_name="genomics/splice_site",
    task_name="splice_site",
)
_SPLICE_SITE_CONFIG = DiffBioBenchmarkConfig(
    name=_SPLICE_SITE_TASK_SPEC.benchmark_name,
    domain="genomics",
    quick_subsample=24,
)


def _run_foundation_task_suite(
    *,
    benchmark_cls: _BenchmarkFactory,
    quick: bool = False,
    source_factory: Callable[[int | None], Any] | None = None,
    adapters: dict[str, SequenceFoundationAdapter] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run one genomics task across native, frozen, and imported adapters."""
    return run_genomics_foundation_benchmark_suite(
        benchmark_factory=lambda embedding_adapter: benchmark_cls(
            quick=quick,
            source_factory=source_factory,
            embedding_adapter=embedding_adapter,
        ),
        adapters=adapters,
    )


def _build_foundation_task_report(
    *,
    benchmark_name: str,
    results: dict[str, BenchmarkResult],
) -> dict[str, Any]:
    """Build the deterministic comparison report for one genomics task."""
    return build_genomics_foundation_task_report(
        benchmark_name=benchmark_name,
        results=results,
        metric_keys=_METRIC_KEYS,
    )


class PromoterBenchmark(SequenceClassificationBenchmark):
    """Evaluate promoter classification on sequence embeddings."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _PROMOTER_CONFIG,
        *,
        quick: bool = False,
        source_factory: Callable[[int | None], Any] | None = None,
        embedding_adapter: SequenceFoundationAdapter | None = None,
    ) -> None:
        super().__init__(
            config,
            task_spec=_PROMOTER_TASK_SPEC,
            quick=quick,
            source_factory=source_factory,
            embedding_adapter=embedding_adapter,
        )


def run_foundation_promoter_suite(
    *,
    quick: bool = False,
    source_factory: Callable[[int | None], Any] | None = None,
    adapters: dict[str, SequenceFoundationAdapter] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run the promoter benchmark across native and imported model families."""
    return _run_foundation_task_suite(
        benchmark_cls=PromoterBenchmark,
        quick=quick,
        source_factory=source_factory,
        adapters=adapters,
    )


def build_foundation_promoter_report(results: dict[str, BenchmarkResult]) -> dict[str, Any]:
    """Build a deterministic comparison report for promoter runs."""
    return _build_foundation_task_report(
        benchmark_name=_PROMOTER_CONFIG.name,
        results=results,
    )


class TFBSBenchmark(SequenceClassificationBenchmark):
    """Evaluate TFBS classification on sequence embeddings."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _TFBS_CONFIG,
        *,
        quick: bool = False,
        source_factory: Callable[[int | None], Any] | None = None,
        embedding_adapter: SequenceFoundationAdapter | None = None,
    ) -> None:
        super().__init__(
            config,
            task_spec=_TFBS_TASK_SPEC,
            quick=quick,
            source_factory=source_factory,
            embedding_adapter=embedding_adapter,
        )


def run_foundation_tfbs_suite(
    *,
    quick: bool = False,
    source_factory: Callable[[int | None], Any] | None = None,
    adapters: dict[str, SequenceFoundationAdapter] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run the TFBS benchmark across native and imported model families."""
    return _run_foundation_task_suite(
        benchmark_cls=TFBSBenchmark,
        quick=quick,
        source_factory=source_factory,
        adapters=adapters,
    )


def build_foundation_tfbs_report(results: dict[str, BenchmarkResult]) -> dict[str, Any]:
    """Build a deterministic comparison report for TFBS runs."""
    return _build_foundation_task_report(
        benchmark_name=_TFBS_CONFIG.name,
        results=results,
    )


class SpliceSiteBenchmark(SequenceClassificationBenchmark):
    """Evaluate splice-site classification on sequence embeddings."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _SPLICE_SITE_CONFIG,
        *,
        quick: bool = False,
        source_factory: Callable[[int | None], Any] | None = None,
        embedding_adapter: SequenceFoundationAdapter | None = None,
    ) -> None:
        super().__init__(
            config,
            task_spec=_SPLICE_SITE_TASK_SPEC,
            quick=quick,
            source_factory=source_factory,
            embedding_adapter=embedding_adapter,
        )


def run_foundation_splice_site_suite(
    *,
    quick: bool = False,
    source_factory: Callable[[int | None], Any] | None = None,
    adapters: dict[str, SequenceFoundationAdapter] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run the splice-site benchmark across native and imported model families."""
    return _run_foundation_task_suite(
        benchmark_cls=SpliceSiteBenchmark,
        quick=quick,
        source_factory=source_factory,
        adapters=adapters,
    )


def build_foundation_splice_site_report(results: dict[str, BenchmarkResult]) -> dict[str, Any]:
    """Build a deterministic comparison report for splice-site runs."""
    return _build_foundation_task_report(
        benchmark_name=_SPLICE_SITE_CONFIG.name,
        results=results,
    )
