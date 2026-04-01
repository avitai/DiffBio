"""Shared helpers for single-cell foundation-model benchmarks."""

from __future__ import annotations

import numpy as np

SINGLECELL_FOUNDATION_SUITE_SCENARIOS = {
    "cell_annotation": "singlecell/foundation_annotation",
    "batch_correction": "singlecell/batch_correction",
    "grn_transfer": "singlecell/grn",
}
SINGLECELL_FOUNDATION_DATASET_CONTRACT_KEYS = (
    "counts",
    "batch_labels",
    "cell_type_labels",
    "cell_ids",
    "embeddings",
    "gene_names",
)


def stratified_cell_annotation_split(
    labels: np.ndarray,
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic stratified train/test split for cell labels."""
    if labels.ndim != 1:
        raise ValueError("Cell annotation labels must be a rank-1 array.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        if label_indices.size < 2:
            raise ValueError(
                "Each cell type must have at least two cells for stratified annotation "
                f"splitting; label {label} has {label_indices.size}."
            )

        shuffled = np.array(label_indices, copy=True)
        rng.shuffle(shuffled)
        n_train = int(np.floor(shuffled.size * train_fraction))
        n_train = min(shuffled.size - 1, max(1, n_train))

        train_indices.extend(int(index) for index in shuffled[:n_train])
        test_indices.extend(int(index) for index in shuffled[n_train:])

    train = np.asarray(sorted(train_indices), dtype=np.int32)
    test = np.asarray(sorted(test_indices), dtype=np.int32)
    return train, test


def compute_annotation_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy and macro-F1 for cell annotation."""
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("True and predicted label arrays must have identical shapes.")

    accuracy = float(np.mean(true_labels == predicted_labels))

    f1_scores: list[float] = []
    for label in np.unique(true_labels):
        true_positive = np.sum((true_labels == label) & (predicted_labels == label))
        false_positive = np.sum((true_labels != label) & (predicted_labels == label))
        false_negative = np.sum((true_labels == label) & (predicted_labels != label))

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(float(2 * precision * recall / (precision + recall)))

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
    }
