#!/usr/bin/env python3
"""DTI contract benchmarks for Davis affinity and BioSNAP interaction tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from calibrax.core.models import Metric, Point
from flax import nnx

from benchmarks._base import (
    DiffBioBenchmark,
    DiffBioBenchmarkConfig,
    build_benchmark_comparison_key,
)
from benchmarks._baselines.dti import DTI_BASELINE_FAMILIES
from benchmarks._optimizers import (
    BENCHMARK_OPTIMIZER_SUBSTRATE,
    create_benchmark_optimizer,
)
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
_OPTIMIZER_TYPE = "adam"
_DIFFERENTIABLE_ENCODER_PATH = "differentiable_pipeline"
_FIXED_SCAFFOLD_ENCODER_PATH = "fixed_scaffold_baseline"
DTI_TRAINING_SUBSTRATE = {
    **BENCHMARK_OPTIMIZER_SUBSTRATE,
    "optimizer_type": _OPTIMIZER_TYPE,
}
DTI_COMPARISON_AXES = ("dataset", "task", "encoder_path")
DTI_COMPARISON_ENCODER_PATHS = (
    _DIFFERENTIABLE_ENCODER_PATH,
    _FIXED_SCAFFOLD_ENCODER_PATH,
)
_DTI_METRIC_DIRECTIONS = {
    "rmse": "lower_is_better",
    "pearson": "higher_is_better",
    "spearman": "higher_is_better",
    "roc_auc": "higher_is_better",
    "pr_auc": "higher_is_better",
    "mrr": "higher_is_better",
    "recall_at_1": "higher_is_better",
    "recall_at_5": "higher_is_better",
    "brier_score": "lower_is_better",
    "expected_calibration_error": "lower_is_better",
}
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
            "calibration": ["brier_score", "expected_calibration_error"],
            "ranking": ["mrr", "recall_at_1", "recall_at_5"],
        },
    },
}


@dataclass(frozen=True)
class _DTITrainingArtifacts:
    """Artifacts from one Opifex-aligned DTI training run."""

    pipeline: DifferentiableDTIPipeline
    pipeline_config: DTIPipelineConfig
    input_data: dict[str, Any]
    targets: jnp.ndarray
    loss_fn: Callable[[DifferentiableDTIPipeline, dict[str, Any]], jnp.ndarray]
    n_train_steps: int


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


def compute_calibration_metrics(
    *,
    targets: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 5,
) -> dict[str, float]:
    """Compute probability calibration metrics from binary labels and logits."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    target_array = np.asarray(targets, dtype=np.float32).ravel()
    probabilities = _sigmoid_np(np.asarray(scores, dtype=np.float32).ravel())
    brier_score = float(np.mean(np.square(probabilities - target_array)))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    expected_calibration_error = 0.0

    for bin_index in range(n_bins):
        lower = bins[bin_index]
        upper = bins[bin_index + 1]
        if bin_index == n_bins - 1:
            in_bin = (probabilities >= lower) & (probabilities <= upper)
        else:
            in_bin = (probabilities >= lower) & (probabilities < upper)
        if not np.any(in_bin):
            continue

        bin_weight = float(np.mean(in_bin))
        confidence = float(np.mean(probabilities[in_bin]))
        accuracy = float(np.mean(target_array[in_bin]))
        expected_calibration_error += bin_weight * abs(confidence - accuracy)

    return {
        "brier_score": brier_score,
        "expected_calibration_error": float(expected_calibration_error),
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
        return _run_dti_pipeline_benchmark(
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
        return _run_dti_pipeline_benchmark(
            data=source.load(),
            dataset_name="biosnap",
            task_name="binary_interaction",
            quick=self.quick,
            loss_mode="classification",
        )


def _run_dti_pipeline_benchmark(
    *,
    data: dict[str, Any],
    dataset_name: str,
    task_name: str,
    quick: bool,
    loss_mode: str,
) -> dict[str, Any]:
    """Train the shared DTI pipeline and package one benchmark result."""
    training = _train_dti_pipeline(data=data, quick=quick, loss_mode=loss_mode)
    result = training.pipeline.apply(training.input_data, {}, None)[0]
    predictions = np.asarray(result["scores"], dtype=np.float32)
    target_array = np.asarray(training.targets, dtype=np.float32)
    baseline_metrics = _run_fixed_scaffold_baseline(data=data, loss_mode=loss_mode)

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
            compute_calibration_metrics(
                targets=target_array.astype(np.int32),
                scores=predictions,
            )
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
    comparison_report = _build_dti_comparison_report(
        dataset_name=dataset_name,
        task_name=task_name,
        primary_metric=str(_build_metric_contract(task_name)["primary_metric"]),
        diffbio_metrics=metrics,
        baseline_metrics=baseline_metrics,
        dataset_provenance=dict(data["dataset_provenance"]),
    )

    return {
        "metrics": metrics,
        "operator": training.pipeline,
        "input_data": training.input_data,
        "loss_fn": training.loss_fn,
        "n_items": int(training.targets.shape[0]),
        "iterate_fn": lambda: training.pipeline.apply(training.input_data, {}, None),
        "baselines": {
            _FIXED_SCAFFOLD_ENCODER_PATH: _build_baseline_point(
                dataset_name=dataset_name,
                task_name=task_name,
                metrics=baseline_metrics,
            )
        },
        "dataset_info": {
            "n_pairs": int(training.targets.shape[0]),
            "n_unique_proteins": len(set(data["protein_ids"])),
            "n_unique_drugs": len(set(data["drug_ids"])),
        },
        "operator_config": {
            "input_mode": "paired_encoded_graph_sequence",
            "protein_encoder": "TransformerSequenceEncoder",
            "protein_hidden_dim": training.pipeline_config.protein_hidden_dim,
            "drug_encoder": "DifferentiableMolecularFingerprint",
            "drug_fingerprint_dim": training.pipeline_config.drug_fingerprint_dim,
            "pair_hidden_dim": training.pipeline_config.pair_hidden_dim,
            "n_train_steps": training.n_train_steps,
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
            "dti_comparison_report": comparison_report,
            "training": {
                **DTI_TRAINING_SUBSTRATE,
                "n_steps": training.n_train_steps,
                "learning_rate": _LEARNING_RATE,
            },
        },
    }


def _train_dti_pipeline(
    *,
    data: dict[str, Any],
    quick: bool,
    loss_mode: str,
) -> _DTITrainingArtifacts:
    """Train one DTI pipeline using the shared Opifex optimizer substrate."""
    pipeline_config = _build_dti_pipeline_config(data)
    input_data = build_dti_pipeline_inputs(data, config=pipeline_config)
    targets = jnp.asarray(input_data["targets"], dtype=jnp.float32)
    pipeline = DifferentiableDTIPipeline(pipeline_config, rngs=nnx.Rngs(42))
    optimizer = nnx.Optimizer(
        pipeline,
        create_benchmark_optimizer(
            optimizer_type=_OPTIMIZER_TYPE,
            learning_rate=_LEARNING_RATE,
        ),
        wrt=nnx.Param,
    )
    n_train_steps = _TRAIN_STEPS_QUICK if quick else _TRAIN_STEPS_FULL

    def loss_fn(model: DifferentiableDTIPipeline, batch_data: dict[str, Any]) -> jnp.ndarray:
        scores = model.apply(batch_data, {}, None)[0]["scores"]
        if loss_mode == "regression":
            return jnp.mean(jnp.square(scores - targets))
        return jnp.mean(_sigmoid_binary_cross_entropy(scores, targets))

    for _ in range(n_train_steps):
        _, grads = nnx.value_and_grad(lambda model: loss_fn(model, input_data))(pipeline)
        optimizer.update(pipeline, grads)

    return _DTITrainingArtifacts(
        pipeline=pipeline,
        pipeline_config=pipeline_config,
        input_data=input_data,
        targets=targets,
        loss_fn=loss_fn,
        n_train_steps=n_train_steps,
    )


def _run_fixed_scaffold_baseline(
    *,
    data: dict[str, Any],
    loss_mode: str,
) -> dict[str, float]:
    """Evaluate a deterministic non-differentiable scaffold-feature baseline."""
    features = np.asarray(build_dti_pair_features(data), dtype=np.float64)
    target_array = np.asarray(data["targets"], dtype=np.float32)
    baseline_predictions = _fit_ridge_baseline(features, target_array)

    if loss_mode == "regression":
        return compute_affinity_regression_metrics(
            targets=target_array,
            predictions=baseline_predictions.astype(np.float32),
        )

    baseline_scores = _probabilities_to_logits(baseline_predictions)
    metrics = compute_binary_interaction_metrics(
        targets=target_array.astype(np.int32),
        scores=baseline_scores,
    )
    metrics.update(
        compute_calibration_metrics(
            targets=target_array.astype(np.int32),
            scores=baseline_scores,
        )
    )
    metrics.update(
        compute_ranking_metrics(
            targets=target_array.astype(np.int32),
            scores=baseline_scores,
            group_ids=list(data["protein_ids"]),
            ks=(1, 5),
        )
    )
    return metrics


def _fit_ridge_baseline(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    regularization: float = 1e-3,
) -> np.ndarray:
    """Fit a closed-form ridge scorer for fixed non-differentiable features."""
    design = np.concatenate(
        [np.ones((features.shape[0], 1), dtype=np.float64), features],
        axis=1,
    )
    penalty = regularization * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return np.asarray(design @ weights, dtype=np.float32)


def _build_baseline_point(
    *,
    dataset_name: str,
    task_name: str,
    metrics: dict[str, float],
) -> Point:
    """Build a Calibrax point for the fixed scaffold comparison row."""
    return Point(
        name=_FIXED_SCAFFOLD_ENCODER_PATH,
        scenario=f"{dataset_name}/{task_name}",
        tags={
            "framework": "diffbio",
            "encoder_path": _FIXED_SCAFFOLD_ENCODER_PATH,
            "source": "closed_form_fixed_scaffold_features",
        },
        metrics={name: Metric(value=float(value)) for name, value in metrics.items()},
    )


def _build_dti_comparison_report(
    *,
    dataset_name: str,
    task_name: str,
    primary_metric: str,
    diffbio_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    dataset_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build the shared DTI comparison report for benchmark metadata."""
    direction = _DTI_METRIC_DIRECTIONS[primary_metric]
    primary_delta = _compute_primary_delta(
        metric_name=primary_metric,
        diffbio_value=diffbio_metrics[primary_metric],
        baseline_value=baseline_metrics[primary_metric],
    )
    promotion_eligible = bool(dataset_provenance.get("promotion_eligible", False))
    stable_claim = (
        "promotion_eligible_external_comparison"
        if promotion_eligible
        else "synthetic_scaffold_comparison_only"
    )

    return {
        "report_version": "dti_comparison_v1",
        "comparison_axes": list(DTI_COMPARISON_AXES),
        "required_encoder_paths": list(DTI_COMPARISON_ENCODER_PATHS),
        "primary_metric": primary_metric,
        "metric_direction": direction,
        "primary_delta_vs_fixed_scaffold": {primary_metric: primary_delta},
        "diffbio_outperforms_fixed_scaffold_on_primary": primary_delta > 0.0,
        "differentiated_value_benchmarked": True,
        "stable_scope": "candidate" if promotion_eligible else "excluded",
        "stable_claim": stable_claim,
        "models": {
            _DIFFERENTIABLE_ENCODER_PATH: {
                "encoder_path": _DIFFERENTIABLE_ENCODER_PATH,
                "model_family": "sequence_transformer",
                "adapter_mode": "native_trainable",
                "comparison_key": build_benchmark_comparison_key(
                    comparison_axes=DTI_COMPARISON_AXES,
                    tags={
                        "dataset": dataset_name,
                        "task": task_name,
                        "encoder_path": _DIFFERENTIABLE_ENCODER_PATH,
                    },
                ),
                "metrics": _serialize_metric_values(diffbio_metrics),
            },
            _FIXED_SCAFFOLD_ENCODER_PATH: {
                "encoder_path": _FIXED_SCAFFOLD_ENCODER_PATH,
                "model_family": "non_differentiable_scaffold_features",
                "adapter_mode": "fixed_features",
                "comparison_key": build_benchmark_comparison_key(
                    comparison_axes=DTI_COMPARISON_AXES,
                    tags={
                        "dataset": dataset_name,
                        "task": task_name,
                        "encoder_path": _FIXED_SCAFFOLD_ENCODER_PATH,
                    },
                ),
                "metrics": _serialize_metric_values(baseline_metrics),
            },
        },
    }


def _serialize_metric_values(metrics: dict[str, float]) -> dict[str, float]:
    """Return JSON-serializable metric values."""
    return {metric_name: float(metric_value) for metric_name, metric_value in metrics.items()}


def _compute_primary_delta(
    *,
    metric_name: str,
    diffbio_value: float,
    baseline_value: float,
) -> float:
    """Compute positive-is-better primary-metric delta against the fixed baseline."""
    direction = _DTI_METRIC_DIRECTIONS[metric_name]
    if direction == "lower_is_better":
        return float(baseline_value - diffbio_value)
    return float(diffbio_value - baseline_value)


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


def _sigmoid_binary_cross_entropy(scores: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute stable binary cross entropy from logits without local optimizer APIs."""
    return jnp.maximum(scores, 0.0) - scores * targets + jnp.log1p(jnp.exp(-jnp.abs(scores)))


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


def _sigmoid_np(scores: np.ndarray) -> np.ndarray:
    """Apply a numerically safe sigmoid to a NumPy score vector."""
    clipped = np.clip(scores, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _probabilities_to_logits(values: np.ndarray) -> np.ndarray:
    """Convert bounded probability-like values to logits."""
    clipped = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float32)


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
