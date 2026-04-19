#!/usr/bin/env python3
"""DTI contract benchmarks for Davis affinity and BioSNAP interaction tasks."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.dti import DTI_BASELINE_FAMILIES
from diffbio.operators.drug_discovery import (
    DTIPipelineConfig,
    DifferentiableDTIPipeline,
    build_dti_pipeline_inputs,
)
from diffbio.sources import (
    DTI_DATASET_CONTRACT_KEYS,
    BioSNAPDTISource,
    DTISourceConfig,
    DavisDTISource,
    validate_dti_dataset,
)

_DAVIS_CONFIG = DiffBioBenchmarkConfig(
    name="drug_discovery/dti_davis",
    domain="drug_discovery",
    quick_subsample=8,
    n_iterations_quick=5,
    n_iterations_full=20,
)
_BIOSNAP_CONFIG = DiffBioBenchmarkConfig(
    name="drug_discovery/dti_biosnap",
    domain="drug_discovery",
    quick_subsample=8,
    n_iterations_quick=5,
    n_iterations_full=20,
)
_TRAIN_STEPS_QUICK = 40
_TRAIN_STEPS_FULL = 120
_LEARNING_RATE = 1e-2
_DTI_METRIC_CONTRACTS = {
    "affinity_regression": {
        "task_type": "affinity_regression",
        "primary_metric": "rmse",
        "metric_groups": {
            "regression": ["rmse", "pearson", "spearman"],
            "classification": [],
            "ranking": [],
        },
    },
    "binary_interaction": {
        "task_type": "binary_interaction",
        "primary_metric": "roc_auc",
        "metric_groups": {
            "regression": [],
            "classification": ["roc_auc", "pr_auc"],
            "ranking": ["mrr", "recall_at_1", "recall_at_5"],
        },
    },
}


class DTIFeatureProbe(nnx.Module):
    """Small learnable probe over paired drug-protein contract features."""

    def __init__(self, input_dim: int, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.linear1 = nnx.Linear(input_dim, 16, rngs=rngs)
        self.linear2 = nnx.Linear(16, 1, rngs=rngs)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Apply the probe to one paired-input batch."""
        del random_params, stats
        pair_features = jnp.asarray(data["pair_features"], dtype=jnp.float32)
        hidden = nnx.gelu(self.linear1(pair_features))
        scores = self.linear2(hidden).squeeze(-1)
        return {"scores": scores}, state, metadata


def build_dti_pair_features(data: dict[str, Any]) -> jnp.ndarray:
    """Encode one paired-input batch as simple numeric scaffold features."""
    validate_dti_dataset(data)
    protein_lengths = np.asarray([len(sequence) for sequence in data["protein_sequences"]])
    drug_lengths = np.asarray([len(smiles) for smiles in data["drug_smiles"]])
    protein_ids = _encode_identifier_series(list(data["protein_ids"]))
    drug_ids = _encode_identifier_series(list(data["drug_ids"]))

    features = np.stack(
        [
            protein_lengths / np.maximum(protein_lengths.max(), 1),
            drug_lengths / np.maximum(drug_lengths.max(), 1),
            (protein_ids + drug_ids) / np.maximum(protein_ids.max() + drug_ids.max(), 1),
            (protein_lengths * drug_lengths)
            / np.maximum((protein_lengths * drug_lengths).max(), 1),
        ],
        axis=1,
    )
    return jnp.asarray(features, dtype=jnp.float32)


def compute_affinity_regression_metrics(
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Compute Davis-style regression metrics."""
    errors = predictions - targets
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    pearson = _safe_correlation(targets, predictions)
    spearman = _safe_correlation(_rank(predictions), _rank(targets))
    return {
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
    }


def compute_binary_interaction_metrics(
    *,
    targets: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Compute binary interaction metrics for BioSNAP."""
    return {
        "roc_auc": _roc_auc(targets, scores),
        "pr_auc": _pr_auc(targets, scores),
    }


def compute_ranking_metrics(
    *,
    targets: np.ndarray,
    scores: np.ndarray,
    group_ids: list[str],
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute ranking metrics grouped by protein id."""
    grouped_indices: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped_indices.setdefault(group_id, []).append(index)

    reciprocal_ranks: list[float] = []
    recall_counts = {k: 0 for k in ks}

    for indices in grouped_indices.values():
        group_targets = targets[indices]
        group_scores = scores[indices]
        order = np.argsort(-group_scores)
        ranked_targets = group_targets[order]

        positive_positions = np.flatnonzero(ranked_targets > 0)
        if positive_positions.size == 0:
            reciprocal_ranks.append(0.0)
            continue

        first_hit = int(positive_positions[0]) + 1
        reciprocal_ranks.append(1.0 / first_hit)
        for k in ks:
            if np.any(ranked_targets[:k] > 0):
                recall_counts[k] += 1

    n_groups = max(len(grouped_indices), 1)
    metrics = {"mrr": float(np.mean(reciprocal_ranks))}
    for k in ks:
        metrics[f"recall_at_{k}"] = recall_counts[k] / n_groups
    return metrics


class DavisDTIBenchmark(DiffBioBenchmark):
    """Quick scaffold benchmark for Davis affinity regression."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _DAVIS_CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "",
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        source = DavisDTISource(
            DTISourceConfig(
                dataset_name="davis",
                split="train",
                data_dir=None if not self.data_dir else Path(self.data_dir),
            )
        )
        return _run_dti_probe_benchmark(
            data=source.load(),
            dataset_name="davis",
            task_name="affinity_regression",
            quick=self.quick,
            loss_mode="regression",
        )


class BioSNAPDTIBenchmark(DiffBioBenchmark):
    """Quick scaffold benchmark for BioSNAP binary interaction prediction."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _BIOSNAP_CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "",
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        source = BioSNAPDTISource(
            DTISourceConfig(
                dataset_name="biosnap",
                split="train",
                data_dir=None if not self.data_dir else Path(self.data_dir),
            )
        )
        return _run_dti_probe_benchmark(
            data=source.load(),
            dataset_name="biosnap",
            task_name="binary_interaction",
            quick=self.quick,
            loss_mode="classification",
        )


def _run_dti_probe_benchmark(
    *,
    data: dict[str, Any],
    dataset_name: str,
    task_name: str,
    quick: bool,
    loss_mode: str,
) -> dict[str, Any]:
    """Train a small paired-input probe and package one DTI benchmark result."""
    pipeline_config = _build_dti_pipeline_config(data)
    input_data = build_dti_pipeline_inputs(data, config=pipeline_config)
    targets = jnp.asarray(input_data["targets"], dtype=jnp.float32)
    pipeline = DifferentiableDTIPipeline(pipeline_config, rngs=nnx.Rngs(42))
    optimizer = nnx.Optimizer(pipeline, optax.adam(_LEARNING_RATE), wrt=nnx.Param)
    n_train_steps = _TRAIN_STEPS_QUICK if quick else _TRAIN_STEPS_FULL

    def loss_fn(model: DifferentiableDTIPipeline, batch_data: dict[str, Any]) -> jnp.ndarray:
        scores = model.apply(batch_data, {}, None)[0]["scores"]
        if loss_mode == "regression":
            return jnp.mean(jnp.square(scores - targets))
        return jnp.mean(optax.sigmoid_binary_cross_entropy(scores, targets))

    for _ in range(n_train_steps):
        loss, grads = nnx.value_and_grad(lambda model: loss_fn(model, input_data))(pipeline)
        optimizer.update(pipeline, grads)

    result = pipeline.apply(input_data, {}, None)[0]
    predictions = np.asarray(result["scores"], dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)

    if loss_mode == "regression":
        metrics = compute_affinity_regression_metrics(
            targets=target_array,
            predictions=predictions,
        )
        paired_contract = _build_paired_contract(task_name, group_key=None)
    else:
        metrics = compute_binary_interaction_metrics(
            targets=target_array.astype(np.int32),
            scores=predictions,
        )
        metrics.update(
            compute_ranking_metrics(
                targets=target_array.astype(np.int32),
                scores=predictions,
                group_ids=list(data["protein_ids"]),
                ks=(1, 5),
            )
        )
        paired_contract = _build_paired_contract(task_name, group_key="protein_ids")

    return {
        "metrics": metrics,
        "operator": pipeline,
        "input_data": input_data,
        "loss_fn": loss_fn,
        "n_items": int(targets.shape[0]),
        "iterate_fn": lambda: pipeline.apply(input_data, {}, None),
        "dataset_info": {
            "n_pairs": int(targets.shape[0]),
            "n_unique_proteins": len(set(data["protein_ids"])),
            "n_unique_drugs": len(set(data["drug_ids"])),
        },
        "operator_config": {
            "input_mode": "paired_encoded_graph_sequence",
            "protein_encoder": "TransformerSequenceEncoder",
            "protein_hidden_dim": pipeline_config.protein_hidden_dim,
            "drug_encoder": "DifferentiableMolecularFingerprint",
            "drug_fingerprint_dim": pipeline_config.drug_fingerprint_dim,
            "pair_hidden_dim": pipeline_config.pair_hidden_dim,
            "n_train_steps": n_train_steps,
            "learning_rate": _LEARNING_RATE,
        },
        "operator_name": "DifferentiableDTIPipeline",
        "dataset_name": dataset_name,
        "task_name": task_name,
        "result_data": result,
        "benchmark_metadata": {
            "paired_contract": paired_contract,
            "dataset_provenance": dict(data["dataset_provenance"]),
            "metric_contract": _build_metric_contract(task_name),
            "baseline_families": list(DTI_BASELINE_FAMILIES),
            "dti_pipeline": result["dti_pipeline"],
        },
    }


def _build_dti_pipeline_config(data: dict[str, Any]) -> DTIPipelineConfig:
    """Build a small shared DTI pipeline config for benchmark-sized runs."""
    max_sequence_length = max(len(sequence) for sequence in data["protein_sequences"])
    return DTIPipelineConfig(
        protein_hidden_dim=8,
        protein_num_layers=1,
        protein_num_heads=2,
        protein_intermediate_dim=16,
        max_protein_length=max(8, max_sequence_length),
        drug_fingerprint_dim=8,
        drug_hidden_dim=8,
        drug_num_layers=1,
        pair_hidden_dim=12,
    )


def _build_paired_contract(
    task_name: str,
    *,
    group_key: str | None,
) -> dict[str, Any]:
    """Build shared paired-input contract metadata for a DTI benchmark."""
    return {
        "task_type": task_name,
        "group_key": group_key,
        "required_keys": list(DTI_DATASET_CONTRACT_KEYS),
    }


def _build_metric_contract(task_name: str) -> dict[str, Any]:
    """Return the metric grouping contract for a DTI task."""
    return dict(_DTI_METRIC_CONTRACTS[task_name])


def _rank(values: np.ndarray) -> np.ndarray:
    """Compute simple integer ranks for correlation metrics."""
    return np.argsort(np.argsort(values))


def _encode_identifier_series(identifiers: list[str]) -> np.ndarray:
    """Map arbitrary string identifiers to compact deterministic integers."""
    mapping: dict[str, int] = {}
    encoded: list[int] = []
    for identifier in identifiers:
        mapping.setdefault(identifier, len(mapping))
        encoded.append(mapping[identifier])
    return np.asarray(encoded, dtype=np.float32)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a correlation coefficient with a zero-variance fallback."""
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _roc_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Compute a simple trapezoidal ROC-AUC."""
    n_pos = int(targets.sum())
    n_neg = len(targets) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(-scores)
    sorted_targets = targets[order]
    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]
    for target in sorted_targets:
        if target == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)

    area = 0.0
    for index in range(1, len(tpr)):
        area += (fpr[index] - fpr[index - 1]) * (tpr[index] + tpr[index - 1]) / 2.0
    return float(area)


def _pr_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Compute a simple trapezoidal PR-AUC."""
    positives = int(targets.sum())
    if positives == 0:
        return 0.0

    order = np.argsort(-scores)
    sorted_targets = targets[order]
    tp = 0
    fp = 0
    recall = [0.0]
    precision = [1.0]
    for target in sorted_targets:
        if target == 1:
            tp += 1
        else:
            fp += 1
        recall.append(tp / positives)
        precision.append(tp / max(tp + fp, 1))

    area = 0.0
    for index in range(1, len(recall)):
        area += (
            (recall[index] - recall[index - 1]) * (precision[index] + precision[index - 1]) / 2.0
        )
    return float(area)
