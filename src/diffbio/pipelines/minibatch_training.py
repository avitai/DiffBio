"""Deterministic mini-batch training for the single-cell annotation arms.

The joint-preprocessing moat is trained by mini-batch SGD, not full-batch descent:
full-batch holds the entire activation graph (``O(n_cells)`` tensors) and removes the
gradient noise that regularizes, whereas mini-batching bounds peak memory to one batch
and scales to atlas-sized data. The global preprocessing statistics (mean/variance/PCA
basis) are fitted once and frozen (a fit-then-transform ``FrozenTransform``), so nothing
here re-estimates a dataset-global statistic per batch -- the objective stays stationary
while only the projection/probe parameters are optimized.

The trainer is deliberately projection-agnostic: it optimizes any model against a
caller-supplied ``forward_fn`` mapping ``(model, features) -> logits``. The frozen-PCA
arm passes the probe over precomputed PCA features; the learnable-projection arm passes a
projection-plus-probe model over the scaled features. Shuffling is derived from an
explicit JAX key folded per epoch, so a fixed seed reproduces training exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from artifex.generative_models.core.configuration.optimizer_config import OptimizerConfig
from artifex.generative_models.training.optimizers.factory import create_optimizer
from flax import nnx

from diffbio.utils.training import cross_entropy_loss

# ``features`` may be a single array or an arbitrary pytree of per-sample arrays (each
# with a leading sample axis), so ``forward_fn`` receives whatever pytree was passed in.
ForwardFn = Callable[[nnx.Module, Any], jnp.ndarray]


def _as_batched_leaf(array: Any) -> jnp.ndarray:
    """Materialize one feature leaf, casting floats to float32 but preserving int dtypes."""
    materialized = jnp.asarray(array)
    if jnp.issubdtype(materialized.dtype, jnp.floating):
        return materialized.astype(jnp.float32)
    return materialized


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniBatchConfig:
    """Configuration for :func:`train_minibatch`.

    Attributes:
        batch_size: Cells per SGD step; ``None`` trains full-batch (one step per
            epoch). A ragged final batch is dropped (``drop_last``).
        n_epochs: Number of passes over the training set.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW decoupled weight decay.
        grad_clip_norm: Global-norm gradient clip.
        seed: Seed for the per-epoch shuffle key.
    """

    batch_size: int | None = 4096
    n_epochs: int = 50
    learning_rate: float = 1.0e-2
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration, failing fast on non-positive values.

        Raises:
            ValueError: If any size/rate field is out of range.
        """
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive or None, got {self.batch_size}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.grad_clip_norm <= 0.0:
            raise ValueError(f"grad_clip_norm must be positive, got {self.grad_clip_norm}")


@dataclass(frozen=True, slots=True)
class MiniBatchResult:
    """Outcome of a :func:`train_minibatch` run.

    Attributes:
        loss_history: Mean cross-entropy loss per epoch.
    """

    loss_history: tuple[float, ...]


def _epoch_batches(n_samples: int, batch_size: int | None, key: jax.Array) -> list[np.ndarray]:
    """Return the shuffled, drop-last mini-batch index arrays for one epoch."""
    if batch_size is None or batch_size >= n_samples:
        return [np.arange(n_samples)]
    permutation = np.asarray(jax.random.permutation(key, n_samples))
    n_batches = n_samples // batch_size
    trimmed = permutation[: n_batches * batch_size]
    return list(trimmed.reshape(n_batches, batch_size))


def train_minibatch(
    model: nnx.Module,
    forward_fn: ForwardFn,
    features: Any,
    labels: Any,
    *,
    n_classes: int,
    config: MiniBatchConfig | None = None,
    aux_loss_fn: Callable[[nnx.Module], jnp.ndarray] | None = None,
) -> MiniBatchResult:
    """Train ``model`` by deterministic mini-batch SGD against the label loss.

    Optimizes ``model``'s :class:`flax.nnx.Param` leaves (fixed
    :class:`flax.nnx.Variable` anchors are left untouched) in place using AdamW from
    the shared artifex factory. Each epoch shuffles the training set with a key folded
    from ``config.seed`` and iterates drop-last mini-batches.

    Args:
        model: The module to optimize (mutated in place).
        forward_fn: Maps ``(model, features_batch) -> logits`` ``(batch, n_classes)``.
        features: Training features as either a single ``(n_samples, ...)`` array or a
            pytree of such arrays (each sharing the leading sample axis); the same pytree
            structure, sliced to the batch, is handed to ``forward_fn``. Float leaves are
            cast to float32; integer leaves (e.g. index tensors) keep their dtype.
        labels: ``(n_samples,)`` integer class labels.
        n_classes: Number of classes for the cross-entropy loss.
        config: Training configuration; defaults are used when ``None``.
        aux_loss_fn: Optional model-only regularizer added to each step's loss, e.g. an
            L0 sparsity penalty on a gate. Receives the module and returns a scalar.

    Returns:
        A :class:`MiniBatchResult` with the mean loss per epoch.
    """
    config = config or MiniBatchConfig()
    feature_tree = jax.tree.map(_as_batched_leaf, features)
    label_array = jnp.asarray(labels, dtype=jnp.int32)
    n_samples = jax.tree.leaves(feature_tree)[0].shape[0]

    optimizer = nnx.Optimizer(
        model,
        create_optimizer(
            OptimizerConfig(
                name="diffbio_minibatch",
                optimizer_type="adamw",
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                gradient_clip_norm=config.grad_clip_norm,
            )
        ),
        wrt=nnx.Param,
    )

    def loss_fn(module: nnx.Module, batch_features: Any, batch_labels: jnp.ndarray) -> jnp.ndarray:
        loss = cross_entropy_loss(
            forward_fn(module, batch_features), batch_labels, num_classes=n_classes
        )
        if aux_loss_fn is not None:
            loss = loss + aux_loss_fn(module)
        return loss

    @nnx.jit
    def train_step(
        module: nnx.Module,
        opt: nnx.Optimizer,
        batch_features: Any,
        batch_labels: jnp.ndarray,
    ) -> jnp.ndarray:
        loss, grads = nnx.value_and_grad(loss_fn)(module, batch_features, batch_labels)
        opt.update(module, grads)
        return loss

    key = jax.random.key(config.seed)
    loss_history: list[float] = []
    for epoch in range(config.n_epochs):
        batches = _epoch_batches(n_samples, config.batch_size, jax.random.fold_in(key, epoch))
        epoch_losses = [
            float(
                train_step(
                    model,
                    optimizer,
                    jax.tree.map(lambda leaf, rows=batch: leaf[rows], feature_tree),
                    label_array[batch],
                )
            )
            for batch in batches
        ]
        loss_history.append(float(np.mean(epoch_losses)))

    return MiniBatchResult(loss_history=tuple(loss_history))
