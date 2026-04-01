"""Shared classification helpers for benchmark suites."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from diffbio.operators.foundation_models import LinearEmbeddingProbe


def stratified_label_split(
    labels: np.ndarray,
    *,
    train_fraction: float,
    seed: int,
    minimum_count_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic stratified train/test split."""
    if labels.ndim != 1:
        raise ValueError("Classification labels must be a rank-1 array.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        if label_indices.size < 2:
            raise ValueError(
                f"Each class must contain at least two {minimum_count_name} for "
                f"stratified splitting; label {label} has {label_indices.size}."
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


def compute_multiclass_classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy and macro-F1 for multiclass classification."""
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


def create_embedding_probe_train_step() -> Any:
    """Create a JIT-compiled training step for the shared embedding probe."""

    @nnx.jit
    def train_step(
        probe: LinearEmbeddingProbe,
        opt: nnx.Optimizer,
        embeddings: jax.Array,
        labels: jax.Array,
    ) -> jax.Array:
        def loss_fn(model_inner: LinearEmbeddingProbe) -> jax.Array:
            result, _, _ = model_inner.apply({"embeddings": embeddings}, {}, None)
            log_probs = jax.nn.log_softmax(result["logits"], axis=-1)
            return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])

        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(probe)
        opt.update(probe, grads)
        return loss

    return train_step
