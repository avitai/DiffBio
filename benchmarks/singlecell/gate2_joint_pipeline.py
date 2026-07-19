#!/usr/bin/env python3
"""Gate 2: does joint optimization beat frozen preprocessing on cell annotation?

Runs the two-arm head-to-head that tests the joint-optimization moat on a large
atlas: a frozen genuine-PCA arm (eigenvectors fixed; the number of components is the
swept dimensionality knob) versus a learnable-projection arm (a PCA-initialized
projection trained jointly with the probe). Both share the frozen preprocessing
statistics fitted once on the training split, are trained by deterministic mini-batch
SGD, and are scored on a held-out split across seeds -- reporting macro-F1 and balanced
accuracy with a rare-cell-type breakdown, the evidence the moat is real at scale.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from benchmarks._classification import stratified_label_split
from benchmarks.singlecell._gate2_arms import (
    ArmResult,
    run_frozen_pca_arm,
    run_learnable_projection_arm,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig

logger = logging.getLogger(__name__)

_TRAIN_FRACTION = 0.8
_DEFAULT_RARE_QUANTILE = 0.25
_DEFAULT_DATA_PATH = os.environ.get(
    "DIFFBIO_TS_CACHE", "/mnt/ssd2/Data/tabula_sapiens/ts_cached.npz"
)
_RESULTS_PATH = Path("benchmarks/results/singlecell/gate2_joint_pipeline.json")


@dataclass(frozen=True, slots=True)
class Gate2Comparison:
    """Aggregated Gate-2 metrics across seeds for both arms.

    Means and standard deviations are taken over the seeds; ``macro_f1_gain`` is the
    per-seed joint-minus-frozen macro-F1, aggregated the same way.
    """

    frozen_macro_f1_mean: float
    frozen_macro_f1_std: float
    joint_macro_f1_mean: float
    joint_macro_f1_std: float
    macro_f1_gain_mean: float
    macro_f1_gain_std: float
    frozen_balanced_accuracy_mean: float
    joint_balanced_accuracy_mean: float
    frozen_rare_macro_f1_mean: float
    joint_rare_macro_f1_mean: float
    n_seeds: int
    per_seed: tuple[dict[str, float], ...]

    def to_dict(self) -> dict:
        """Return a JSON-serializable mapping of the comparison."""
        payload = asdict(self)
        payload["per_seed"] = [dict(entry) for entry in self.per_seed]
        return payload


def rare_classes_from_counts(
    train_labels: np.ndarray, n_classes: int, quantile: float
) -> np.ndarray:
    """Return the indices of the rare classes (train frequency at or below a quantile).

    Args:
        train_labels: ``(n_train,)`` integer training labels.
        n_classes: Number of classes.
        quantile: Frequency quantile; classes at or below it are rare.

    Returns:
        A sorted ``int64`` array of rare class indices.
    """
    class_counts = np.array([int(np.sum(train_labels == c)) for c in range(n_classes)])
    threshold = np.quantile(class_counts, quantile)
    return np.flatnonzero(class_counts <= threshold).astype(np.int64)


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return the mean and (population) standard deviation of ``values``."""
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def _run_seed(
    counts: np.ndarray,
    labels: np.ndarray,
    seed: int,
    *,
    n_classes: int,
    n_top_genes: int,
    n_components: int,
    hidden_dim: int | None,
    rare_quantile: float,
    config: MiniBatchConfig,
    hvg_method: str,
) -> tuple[ArmResult, ArmResult]:
    """Run both arms on one stratified split and return (frozen, joint) results."""
    train_indices, test_indices = stratified_label_split(
        labels, train_fraction=_TRAIN_FRACTION, seed=seed, minimum_count_name="cells"
    )
    train_counts, test_counts = counts[train_indices], counts[test_indices]
    train_labels, test_labels = labels[train_indices], labels[test_indices]
    rare = rare_classes_from_counts(train_labels, n_classes, rare_quantile)

    shared = dict(
        n_classes=n_classes,
        n_top_genes=n_top_genes,
        n_components=n_components,
        hidden_dim=hidden_dim,
        rare_classes=rare,
        config=config,
        hvg_method=hvg_method,
        probe_seed=seed,
    )
    frozen = run_frozen_pca_arm(train_counts, train_labels, test_counts, test_labels, **shared)
    joint = run_learnable_projection_arm(
        train_counts, train_labels, test_counts, test_labels, **shared
    )
    return frozen, joint


def gate2_comparison(
    counts: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_top_genes: int = 2000,
    n_components: int = 10,
    hidden_dim: int | None = 128,
    batch_size: int | None = 4096,
    n_epochs: int = 100,
    learning_rate: float = 1.0e-2,
    weight_decay: float = 5.0e-2,
    rare_quantile: float = _DEFAULT_RARE_QUANTILE,
    hvg_method: str = "dispersion",
) -> Gate2Comparison:
    """Compare the frozen-PCA and learnable-projection arms across seeds.

    Args:
        counts: ``(n_cells, n_genes)`` raw count matrix.
        labels: ``(n_cells,)`` integer cell-type labels.
        n_classes: Number of cell-type classes.
        seeds: Seeds for the split, initialization, and shuffle.
        n_top_genes: Highly-variable genes retained by the frozen transform.
        n_components: Dimensionality of both arms' reduction.
        hidden_dim: Probe hidden width (``None`` for a linear head).
        batch_size: Mini-batch size (``None`` for full-batch).
        n_epochs: Training epochs per arm.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        rare_quantile: Frequency quantile defining rare classes.
        hvg_method: Gene selection for both arms -- ``"dispersion"`` or ``"supervised"``.

    Returns:
        The aggregated :class:`Gate2Comparison`.
    """
    frozen_results: list[ArmResult] = []
    joint_results: list[ArmResult] = []
    per_seed: list[dict[str, float]] = []

    for seed in seeds:
        config = MiniBatchConfig(
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        )
        frozen, joint = _run_seed(
            counts,
            labels,
            seed,
            n_classes=n_classes,
            n_top_genes=n_top_genes,
            n_components=n_components,
            hidden_dim=hidden_dim,
            rare_quantile=rare_quantile,
            config=config,
            hvg_method=hvg_method,
        )
        frozen_results.append(frozen)
        joint_results.append(joint)
        per_seed.append(
            {
                "seed": float(seed),
                "frozen_macro_f1": frozen.macro_f1,
                "joint_macro_f1": joint.macro_f1,
                "macro_f1_gain": joint.macro_f1 - frozen.macro_f1,
                "frozen_balanced_accuracy": frozen.balanced_accuracy,
                "joint_balanced_accuracy": joint.balanced_accuracy,
                "frozen_rare_macro_f1": frozen.rare_macro_f1,
                "joint_rare_macro_f1": joint.rare_macro_f1,
            }
        )

    frozen_macro_mean, frozen_macro_std = _mean_std([r.macro_f1 for r in frozen_results])
    joint_macro_mean, joint_macro_std = _mean_std([r.macro_f1 for r in joint_results])
    gain_mean, gain_std = _mean_std(
        [j.macro_f1 - f.macro_f1 for f, j in zip(frozen_results, joint_results, strict=True)]
    )
    return Gate2Comparison(
        frozen_macro_f1_mean=frozen_macro_mean,
        frozen_macro_f1_std=frozen_macro_std,
        joint_macro_f1_mean=joint_macro_mean,
        joint_macro_f1_std=joint_macro_std,
        macro_f1_gain_mean=gain_mean,
        macro_f1_gain_std=gain_std,
        frozen_balanced_accuracy_mean=_mean_std([r.balanced_accuracy for r in frozen_results])[0],
        joint_balanced_accuracy_mean=_mean_std([r.balanced_accuracy for r in joint_results])[0],
        frozen_rare_macro_f1_mean=float(np.nanmean([r.rare_macro_f1 for r in frozen_results])),
        joint_rare_macro_f1_mean=float(np.nanmean([r.rare_macro_f1 for r in joint_results])),
        n_seeds=len(seeds),
        per_seed=tuple(per_seed),
    )


def sweep_frozen_dimensions(
    counts: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int,
    k_values: tuple[int, ...],
    seeds: tuple[int, ...] = (0, 1, 2),
    n_top_genes: int = 2000,
    hidden_dim: int | None = 128,
    batch_size: int | None = 4096,
    n_epochs: int = 100,
    learning_rate: float = 1.0e-2,
    weight_decay: float = 5.0e-2,
    rare_quantile: float = _DEFAULT_RARE_QUANTILE,
) -> dict[int, tuple[float, float]]:
    """Sweep the frozen-PCA arm's number of components (the dimensionality knob).

    Args:
        counts: ``(n_cells, n_genes)`` raw count matrix.
        labels: ``(n_cells,)`` integer cell-type labels.
        n_classes: Number of cell-type classes.
        k_values: The component counts to evaluate.
        seeds: Seeds to average over.
        n_top_genes: Highly-variable genes retained by the frozen transform.
        hidden_dim: Probe hidden width.
        batch_size: Mini-batch size.
        n_epochs: Training epochs.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        rare_quantile: Frequency quantile defining rare classes.

    Returns:
        A mapping ``k -> (mean_macro_f1, std_macro_f1)`` over seeds.
    """
    curve: dict[int, tuple[float, float]] = {}
    for k in k_values:
        macro_f1s: list[float] = []
        for seed in seeds:
            config = MiniBatchConfig(
                batch_size=batch_size,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=seed,
            )
            train_indices, test_indices = stratified_label_split(
                labels, train_fraction=_TRAIN_FRACTION, seed=seed, minimum_count_name="cells"
            )
            rare = rare_classes_from_counts(labels[train_indices], n_classes, rare_quantile)
            result = run_frozen_pca_arm(
                counts[train_indices],
                labels[train_indices],
                counts[test_indices],
                labels[test_indices],
                n_classes=n_classes,
                n_top_genes=n_top_genes,
                n_components=k,
                hidden_dim=hidden_dim,
                rare_classes=rare,
                config=config,
                probe_seed=seed,
            )
            macro_f1s.append(result.macro_f1)
        curve[k] = _mean_std(macro_f1s)
    return curve


def main() -> None:
    """Run Gate 2 on the cached Tabula Sapiens atlas and record the comparison."""
    logging.basicConfig(level=logging.INFO)
    with np.load(_DEFAULT_DATA_PATH) as data:
        counts = np.asarray(data["counts"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int32)
        n_classes = int(data["n_types"])

    comparison = gate2_comparison(counts, labels, n_classes=n_classes)
    logger.info(
        "Gate 2 | frozen macro_f1=%.4f joint macro_f1=%.4f gain=%.4f (+/-%.4f, %d seeds)",
        comparison.frozen_macro_f1_mean,
        comparison.joint_macro_f1_mean,
        comparison.macro_f1_gain_mean,
        comparison.macro_f1_gain_std,
        comparison.n_seeds,
    )
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(comparison.to_dict(), indent=2))


if __name__ == "__main__":
    main()
