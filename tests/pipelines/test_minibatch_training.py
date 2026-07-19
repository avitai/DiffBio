"""Tests for the deterministic mini-batch trainer."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.pipelines.minibatch_training import (
    MiniBatchConfig,
    train_minibatch,
)


def _linear(input_dim: int, n_classes: int, seed: int) -> nnx.Linear:
    return nnx.Linear(input_dim, n_classes, rngs=nnx.Rngs(seed))


def _forward(model: nnx.Linear, features: jnp.ndarray) -> jnp.ndarray:
    return model(features)


def _separable_task(
    n_cells: int, n_features: int, n_classes: int, seed: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Linearly separable features: each class is a shifted Gaussian blob."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n_cells)
    centers = rng.normal(size=(n_classes, n_features)) * 5.0
    features = centers[labels] + rng.normal(size=(n_cells, n_features))
    return jnp.asarray(features, dtype=jnp.float32), jnp.asarray(labels, dtype=jnp.int32)


def _accuracy(model: nnx.Linear, features: jnp.ndarray, labels: jnp.ndarray) -> float:
    predictions = jnp.argmax(_forward(model, features), axis=-1)
    return float(jnp.mean(predictions == labels))


# --- Config validation ----------------------------------------------------------


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="n_epochs"):
        MiniBatchConfig(n_epochs=0)
    with pytest.raises(ValueError, match="batch_size"):
        MiniBatchConfig(batch_size=0)
    with pytest.raises(ValueError, match="learning_rate"):
        MiniBatchConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        MiniBatchConfig(weight_decay=-0.1)
    with pytest.raises(ValueError, match="grad_clip_norm"):
        MiniBatchConfig(grad_clip_norm=0.0)


# --- Core learning behavior -----------------------------------------------------


def test_loss_decreases_and_task_is_learned() -> None:
    features, labels = _separable_task(400, 8, 4, seed=0)
    model = _linear(8, 4, seed=1)
    config = MiniBatchConfig(batch_size=64, n_epochs=40, learning_rate=5e-2, seed=0)
    result = train_minibatch(model, _forward, features, labels, n_classes=4, config=config)
    assert result.loss_history[-1] < result.loss_history[0]
    assert _accuracy(model, features, labels) > 0.9


def test_parameters_change_from_initialization() -> None:
    features, labels = _separable_task(200, 6, 3, seed=2)
    model = _linear(6, 3, seed=3)
    before = np.asarray(model.kernel[...]).copy()
    config = MiniBatchConfig(batch_size=32, n_epochs=10, seed=0)
    train_minibatch(model, _forward, features, labels, n_classes=3, config=config)
    assert not np.allclose(before, np.asarray(model.kernel[...]))


# --- Determinism ----------------------------------------------------------------


def test_same_seed_gives_identical_training() -> None:
    features, labels = _separable_task(300, 6, 3, seed=4)
    config = MiniBatchConfig(batch_size=48, n_epochs=15, seed=7)

    model_a = _linear(6, 3, seed=5)
    result_a = train_minibatch(model_a, _forward, features, labels, n_classes=3, config=config)
    model_b = _linear(6, 3, seed=5)
    result_b = train_minibatch(model_b, _forward, features, labels, n_classes=3, config=config)

    assert result_a.loss_history == result_b.loss_history
    np.testing.assert_array_equal(np.asarray(model_a.kernel[...]), np.asarray(model_b.kernel[...]))


def test_different_shuffle_seed_changes_trajectory() -> None:
    features, labels = _separable_task(300, 6, 3, seed=6)
    model_a = _linear(6, 3, seed=8)
    model_b = _linear(6, 3, seed=8)
    result_a = train_minibatch(
        model_a,
        _forward,
        features,
        labels,
        n_classes=3,
        config=MiniBatchConfig(batch_size=48, seed=1),
    )
    result_b = train_minibatch(
        model_b,
        _forward,
        features,
        labels,
        n_classes=3,
        config=MiniBatchConfig(batch_size=48, seed=2),
    )
    assert result_a.loss_history != result_b.loss_history


# --- Batch-size regimes ---------------------------------------------------------


def test_full_batch_mode_trains() -> None:
    features, labels = _separable_task(200, 6, 3, seed=9)
    model = _linear(6, 3, seed=10)
    config = MiniBatchConfig(batch_size=None, n_epochs=60, learning_rate=5e-2, seed=0)
    result = train_minibatch(model, _forward, features, labels, n_classes=3, config=config)
    assert len(result.loss_history) == 60
    assert _accuracy(model, features, labels) > 0.9


def test_drop_last_handles_ragged_final_batch() -> None:
    # 250 cells, batch 64 -> 3 full batches (192 cells), ragged tail dropped.
    features, labels = _separable_task(250, 6, 3, seed=11)
    model = _linear(6, 3, seed=12)
    config = MiniBatchConfig(batch_size=64, n_epochs=5, seed=0)
    # Must run without a shape error despite 250 % 64 != 0.
    result = train_minibatch(model, _forward, features, labels, n_classes=3, config=config)
    assert len(result.loss_history) == 5


def test_gradients_stay_finite() -> None:
    features, labels = _separable_task(200, 6, 3, seed=13)
    model = _linear(6, 3, seed=14)
    config = MiniBatchConfig(batch_size=32, n_epochs=20, seed=0)
    result = train_minibatch(model, _forward, features, labels, n_classes=3, config=config)
    assert all(np.isfinite(loss) for loss in result.loss_history)


# --- Pytree (multi-tensor) features ---------------------------------------------


class _GatedLinear(nnx.Module):
    """Model whose forward consumes a dict of features (a gate mask plus values)."""

    def __init__(self, input_dim: int, n_classes: int, seed: int) -> None:
        self.linear = nnx.Linear(input_dim, n_classes, rngs=nnx.Rngs(seed))

    def logits(self, features: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Multiply the value tensor by an integer gate before the linear layer."""
        return self.linear(features["values"] * features["gate"].astype(jnp.float32))


def _dict_forward(model: _GatedLinear, features: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return model.logits(features)


def test_pytree_features_train_and_preserve_int_dtype() -> None:
    values, labels = _separable_task(240, 6, 3, seed=15)
    # An all-ones integer gate must reach forward_fn as an int tensor, not cast to float.
    features = {"values": values, "gate": jnp.ones((240, 6), dtype=jnp.int32)}
    model = _GatedLinear(6, 3, seed=16)
    config = MiniBatchConfig(batch_size=48, n_epochs=40, learning_rate=5e-2, seed=0)
    result = train_minibatch(model, _dict_forward, features, labels, n_classes=3, config=config)

    assert result.loss_history[-1] < result.loss_history[0]
    predictions = jnp.argmax(model.logits(features), axis=-1)
    assert float(jnp.mean(predictions == labels)) > 0.9
