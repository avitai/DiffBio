"""Shared helpers for genomics foundation-model benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from calibrax.core.result import BenchmarkResult

from benchmarks._baselines.genomics_foundation import (
    GENOMICS_FOUNDATION_BASELINE_FAMILIES,
)
from benchmarks._classification import (
    compute_multiclass_classification_metrics,
    stratified_label_split,
)
from benchmarks._foundation_models import (
    build_foundation_task_report,
    run_foundation_benchmark_suite,
)
from diffbio.operators.foundation_models import SequenceFoundationAdapter

GENOMICS_FOUNDATION_SUITE_SCENARIOS = {
    "promoter": "genomics/promoter",
    "tfbs": "genomics/tfbs",
    "splice_site": "genomics/splice_site",
}
GENOMICS_FOUNDATION_DATASET_CONTRACT_KEYS = (
    "sequence_ids",
    "sequences",
    "one_hot_sequences",
    "labels",
)
_GENOMICS_REPORT_METADATA_KEYS = (
    "embedding_source",
    "foundation_source_name",
)


def stratified_sequence_classification_split(
    labels: np.ndarray,
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic stratified train/test split for sequence labels."""
    return stratified_label_split(
        labels,
        train_fraction=train_fraction,
        seed=seed,
        minimum_count_name="sequences",
    )


def compute_sequence_classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy and macro-F1 for genomics classification tasks."""
    return compute_multiclass_classification_metrics(true_labels, predicted_labels)


def run_genomics_foundation_benchmark_suite(
    *,
    benchmark_factory: Callable[[SequenceFoundationAdapter | None], Any],
    adapters: dict[str, SequenceFoundationAdapter] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run one genomics benchmark across native and imported model families."""
    return run_foundation_benchmark_suite(
        baseline_families=GENOMICS_FOUNDATION_BASELINE_FAMILIES,
        benchmark_factory=benchmark_factory,
        adapters=adapters,
    )


def build_genomics_foundation_task_report(
    *,
    benchmark_name: str,
    results: dict[str, BenchmarkResult],
    metric_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Build a deterministic comparison report for one genomics task."""
    return build_foundation_task_report(
        benchmark_name=benchmark_name,
        results=results,
        metric_keys=metric_keys,
        baseline_families=GENOMICS_FOUNDATION_BASELINE_FAMILIES,
        metadata_keys=_GENOMICS_REPORT_METADATA_KEYS,
    )
