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
from diffbio.sources import BioSNAPDTISource, DTISourceConfig, DavisDTISource

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
    pair_features = build_dti_pair_features(data)
    targets = jnp.asarray(data["targets"], dtype=jnp.float32)
    probe = DTIFeatureProbe(int(pair_features.shape[1]), rngs=nnx.Rngs(42))
    optimizer = nnx.Optimizer(probe, optax.adam(_LEARNING_RATE), wrt=nnx.Param)
    n_train_steps = _TRAIN_STEPS_QUICK if quick else _TRAIN_STEPS_FULL

    def loss_fn(model: DTIFeatureProbe, batch_data: dict[str, Any]) -> jnp.ndarray:
        scores = model.apply(batch_data, {}, None)[0]["scores"]
        if loss_mode == "regression":
            return jnp.mean(jnp.square(scores - targets))
        return jnp.mean(optax.sigmoid_binary_cross_entropy(scores, targets))

    input_data = {"pair_features": pair_features}
    for _ in range(n_train_steps):
        loss, grads = nnx.value_and_grad(lambda model: loss_fn(model, input_data))(probe)
        optimizer.update(probe, grads)

    result = probe.apply(input_data, {}, None)[0]
    predictions = np.asarray(result["scores"], dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)

    if loss_mode == "regression":
        metrics = compute_affinity_regression_metrics(
            targets=target_array,
            predictions=predictions,
        )
        paired_contract = {"task_type": task_name, "group_key": None}
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
        paired_contract = {"task_type": task_name, "group_key": "protein_ids"}

    return {
        "metrics": metrics,
        "operator": probe,
        "input_data": input_data,
        "loss_fn": loss_fn,
        "n_items": int(pair_features.shape[0]),
        "iterate_fn": lambda: probe.apply(input_data, {}, None),
        "dataset_info": {
            "n_pairs": int(pair_features.shape[0]),
            "n_unique_proteins": len(set(data["protein_ids"])),
            "n_unique_drugs": len(set(data["drug_ids"])),
        },
        "operator_config": {
            "input_dim": int(pair_features.shape[1]),
            "n_train_steps": n_train_steps,
            "learning_rate": _LEARNING_RATE,
        },
        "operator_name": "DTIFeatureProbe",
        "dataset_name": dataset_name,
        "task_name": task_name,
        "benchmark_metadata": {
            "paired_contract": paired_contract,
            "baseline_families": list(DTI_BASELINE_FAMILIES),
        },
    }


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
