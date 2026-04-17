"""Shared helpers for single-cell foundation-model benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from calibrax.core.result import BenchmarkResult

from benchmarks._classification import (
    compute_multiclass_classification_metrics,
    stratified_label_split,
)
from benchmarks._foundation_models import (
    build_foundation_task_report,
    run_foundation_benchmark_suite,
)
from benchmarks._baselines.singlecell_foundation import (
    SINGLECELL_FOUNDATION_BASELINE_FAMILIES,
)
from diffbio.operators.foundation_models import SingleCellPrecomputedAdapter

SINGLECELL_FOUNDATION_SUITE_SCENARIOS = {
    "cell_annotation": "singlecell/foundation_annotation",
    "batch_correction": "singlecell/batch_correction",
    "grn_transfer": "singlecell/grn",
}
SINGLECELL_FOUNDATION_SUPPORT_MATRIX = {
    "GeneformerPrecomputedAdapter": {
        "adapter_key": "geneformer_precomputed",
        "stable_modes": ("precomputed",),
        "verified_tasks": ("cell_annotation", "batch_correction"),
        "planned_tasks": ("grn_transfer",),
        "stable_scope_exclusions": (
            "direct_checkpoint_loading",
            "tokenizer_interchange",
            "generic_fine_tuning",
        ),
    },
    "ScGPTPrecomputedAdapter": {
        "adapter_key": "scgpt_precomputed",
        "stable_modes": ("precomputed",),
        "verified_tasks": ("cell_annotation", "batch_correction"),
        "planned_tasks": ("grn_transfer",),
        "stable_scope_exclusions": (
            "direct_checkpoint_loading",
            "tokenizer_interchange",
            "generic_fine_tuning",
        ),
    },
}
SINGLECELL_FOUNDATION_DATASET_CONTRACT_KEYS = (
    "counts",
    "batch_labels",
    "cell_type_labels",
    "cell_ids",
    "embeddings",
    "gene_names",
)
_FOUNDATION_REPORT_METADATA_KEYS = (
    "embedding_source",
    "foundation_source_name",
    "requires_batch_context",
    "batch_key",
    "context_version",
)


def stratified_cell_annotation_split(
    labels: np.ndarray,
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic stratified train/test split for cell labels."""
    return stratified_label_split(
        labels,
        train_fraction=train_fraction,
        seed=seed,
        minimum_count_name="cells",
    )


def compute_annotation_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy and macro-F1 for cell annotation."""
    return compute_multiclass_classification_metrics(true_labels, predicted_labels)


def run_singlecell_foundation_benchmark_suite(
    *,
    benchmark_factory: Callable[[SingleCellPrecomputedAdapter | None], Any],
    adapters: dict[str, SingleCellPrecomputedAdapter] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run one single-cell benchmark across native and imported model families."""
    return run_foundation_benchmark_suite(
        baseline_families=SINGLECELL_FOUNDATION_BASELINE_FAMILIES,
        benchmark_factory=benchmark_factory,
        adapters=adapters,
    )


def build_singlecell_foundation_task_report(
    *,
    benchmark_name: str,
    results: dict[str, BenchmarkResult],
    metric_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Build a deterministic comparison report for one single-cell task."""
    return build_foundation_task_report(
        benchmark_name=benchmark_name,
        results=results,
        metric_keys=metric_keys,
        baseline_families=SINGLECELL_FOUNDATION_BASELINE_FAMILIES,
        metadata_keys=_FOUNDATION_REPORT_METADATA_KEYS,
    )
