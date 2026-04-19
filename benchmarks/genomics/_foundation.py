"""Shared helpers for genomics foundation-model benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
_GENOMICS_FOUNDATION_VERIFIED_TASKS = ("promoter", "tfbs", "splice_site")
_GENOMICS_FOUNDATION_PRECOMPUTED_EXCLUSIONS = (
    "generic_checkpoint_loading",
    "tokenizer_interchange",
    "stable_genomics_promotion",
)
GENOMICS_FOUNDATION_SUPPORT_MATRIX = {
    "FrozenSequenceEncoderAdapter": {
        "adapter_key": "diffbio_frozen_encoder",
        "adapter_modes": ("frozen_encoder",),
        "verified_tasks": _GENOMICS_FOUNDATION_VERIFIED_TASKS,
        "support_status": "phase_4_scaffold",
        "scope_exclusions": (
            "external_frozen_checkpoint_import",
            "stable_genomics_promotion",
        ),
    },
    "DNABERT2PrecomputedAdapter": {
        "adapter_key": "dnabert2_precomputed",
        "adapter_modes": ("precomputed",),
        "verified_tasks": _GENOMICS_FOUNDATION_VERIFIED_TASKS,
        "support_status": "phase_4_scaffold",
        "scope_exclusions": _GENOMICS_FOUNDATION_PRECOMPUTED_EXCLUSIONS,
    },
    "NucleotideTransformerPrecomputedAdapter": {
        "adapter_key": "nucleotide_transformer_precomputed",
        "adapter_modes": ("precomputed",),
        "verified_tasks": _GENOMICS_FOUNDATION_VERIFIED_TASKS,
        "support_status": "phase_4_scaffold",
        "scope_exclusions": _GENOMICS_FOUNDATION_PRECOMPUTED_EXCLUSIONS,
    },
}
GENOMICS_FOUNDATION_DATASET_CONTRACT_KEYS = (
    "sequence_ids",
    "sequences",
    "one_hot_sequences",
    "labels",
)
GENOMICS_FOUNDATION_DATASET_PROVENANCE_KEYS = (
    "dataset_name",
    "source_type",
    "curation_status",
    "provenance_label",
    "biological_validation",
    "promotion_eligible",
)
GENOMICS_FOUNDATION_DATASET_PROVENANCE: dict[str, dict[str, Any]] = {
    "synthetic_genomics": {
        "dataset_name": "synthetic_genomics",
        "source_type": "scaffold",
        "curation_status": "synthetic",
        "provenance_label": "deterministic_motif_scaffold",
        "biological_validation": "interface_validation_only",
        "promotion_eligible": False,
    }
}
_GENOMICS_FOUNDATION_DATASET_SOURCE_TYPES = frozenset({"scaffold", "curated"})
_GENOMICS_REPORT_METADATA_KEYS = (
    "embedding_source",
    "foundation_source_name",
    "dataset_provenance",
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


def _validate_genomics_dataset_provenance(
    dataset_name: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one genomics dataset provenance payload."""
    missing_keys = [
        key for key in GENOMICS_FOUNDATION_DATASET_PROVENANCE_KEYS if key not in provenance
    ]
    if missing_keys:
        raise ValueError(f"genomics dataset_provenance is missing required keys: {missing_keys}")

    normalized = {key: provenance[key] for key in GENOMICS_FOUNDATION_DATASET_PROVENANCE_KEYS}
    for key in GENOMICS_FOUNDATION_DATASET_PROVENANCE_KEYS:
        if key == "promotion_eligible":
            continue
        if not isinstance(normalized[key], str) or not normalized[key]:
            raise TypeError(f"genomics dataset_provenance.{key} must be a non-empty string.")
    if normalized["dataset_name"] != dataset_name:
        raise ValueError(
            "genomics dataset_provenance.dataset_name must match the benchmark dataset "
            f"({normalized['dataset_name']!r} vs {dataset_name!r})."
        )
    if normalized["source_type"] not in _GENOMICS_FOUNDATION_DATASET_SOURCE_TYPES:
        raise ValueError("genomics dataset_provenance.source_type must be 'scaffold' or 'curated'.")
    if not isinstance(normalized["promotion_eligible"], bool):
        raise TypeError("genomics dataset_provenance.promotion_eligible must be a bool.")
    if normalized["source_type"] == "scaffold" and normalized["promotion_eligible"]:
        raise ValueError("Genomics scaffold provenance cannot be promotion_eligible.")

    return normalized


def resolve_genomics_dataset_provenance(
    dataset_name: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve explicit scaffold-versus-curated provenance for a genomics dataset."""
    if provenance is not None:
        return _validate_genomics_dataset_provenance(dataset_name, provenance)

    known_provenance = GENOMICS_FOUNDATION_DATASET_PROVENANCE.get(dataset_name)
    if known_provenance is None:
        raise ValueError(
            "Custom genomics foundation datasets must provide dataset_provenance metadata."
        )
    return dict(known_provenance)


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
    report = build_foundation_task_report(
        benchmark_name=benchmark_name,
        results=results,
        metric_keys=metric_keys,
        baseline_families=GENOMICS_FOUNDATION_BASELINE_FAMILIES,
        metadata_keys=_GENOMICS_REPORT_METADATA_KEYS,
    )
    report["dataset_provenance"] = _resolve_task_report_dataset_provenance(report)
    return report


def _resolve_task_report_dataset_provenance(task_report: dict[str, Any]) -> dict[str, Any]:
    """Require one shared dataset provenance payload across a task report."""
    provenance_payloads: list[dict[str, Any]] = []
    models = task_report.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("Genomics task report models must be a dict.")

    for model_name, model_report in models.items():
        if not isinstance(model_report, dict):
            raise ValueError(f"Genomics model report must be a dict: {model_name}")
        metadata = model_report.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Genomics model report metadata must be a dict: {model_name}")
        provenance = metadata.get("dataset_provenance")
        if provenance is None:
            raise ValueError(
                f"Genomics model report is missing dataset_provenance metadata: {model_name}"
            )
        if not isinstance(provenance, dict):
            raise TypeError("Genomics model metadata dataset_provenance must be a dict.")
        provenance_payloads.append(provenance)

    if not provenance_payloads:
        raise ValueError("Genomics task reports require dataset_provenance metadata.")

    canonical = provenance_payloads[0]
    if any(provenance != canonical for provenance in provenance_payloads[1:]):
        raise ValueError("Genomics task report models disagree on dataset_provenance metadata.")

    return dict(canonical)
