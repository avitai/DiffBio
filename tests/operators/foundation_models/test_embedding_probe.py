"""Tests for the foundation-model embedding probe operator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.foundation_models import (
    EmbeddingProbeConfig,
    LinearEmbeddingProbe,
)


@pytest.fixture()
def probe_config() -> EmbeddingProbeConfig:
    """Default probe configuration."""
    return EmbeddingProbeConfig(input_dim=8, n_classes=3)


@pytest.fixture()
def embedding_batch() -> dict[str, jax.Array]:
    """Synthetic embedding batch."""
    embeddings = jnp.arange(40, dtype=jnp.float32).reshape(5, 8) / 10.0
    return {"embeddings": embeddings}


class TestEmbeddingProbeConfig:
    """Tests for EmbeddingProbeConfig."""

    def test_default_hidden_dim(self) -> None:
        config = EmbeddingProbeConfig(input_dim=16, n_classes=4)
        assert config.input_dim == 16
        assert config.n_classes == 4
        assert config.hidden_dim is None


class TestLinearEmbeddingProbe:
    """Tests for LinearEmbeddingProbe."""

    def test_output_keys(
        self,
        probe_config: EmbeddingProbeConfig,
        embedding_batch: dict[str, jax.Array],
    ) -> None:
        probe = LinearEmbeddingProbe(probe_config, rngs=nnx.Rngs(0))

        result, _, _ = probe.apply(embedding_batch, {}, None)

        assert "embeddings" in result
        assert "logits" in result
        assert "probabilities" in result
        assert "predicted_labels" in result

    def test_output_shapes(
        self,
        probe_config: EmbeddingProbeConfig,
        embedding_batch: dict[str, jax.Array],
    ) -> None:
        probe = LinearEmbeddingProbe(probe_config, rngs=nnx.Rngs(0))

        result, _, _ = probe.apply(embedding_batch, {}, None)

        assert result["logits"].shape == (5, 3)
        assert result["probabilities"].shape == (5, 3)
        assert result["predicted_labels"].shape == (5,)

    def test_probabilities_sum_to_one(
        self,
        probe_config: EmbeddingProbeConfig,
        embedding_batch: dict[str, jax.Array],
    ) -> None:
        probe = LinearEmbeddingProbe(probe_config, rngs=nnx.Rngs(0))

        result, _, _ = probe.apply(embedding_batch, {}, None)

        assert jnp.allclose(jnp.sum(result["probabilities"], axis=-1), 1.0, atol=1e-5)

    def test_probe_is_differentiable(
        self,
        probe_config: EmbeddingProbeConfig,
        embedding_batch: dict[str, jax.Array],
    ) -> None:
        probe = LinearEmbeddingProbe(probe_config, rngs=nnx.Rngs(0))
        labels = jnp.array([0, 1, 2, 1, 0], dtype=jnp.int32)

        def loss_fn(model: LinearEmbeddingProbe) -> jax.Array:
            result, _, _ = model.apply(embedding_batch, {}, None)
            log_probs = jax.nn.log_softmax(result["logits"], axis=-1)
            return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])

        grads = nnx.grad(loss_fn)(probe)

        assert jnp.any(grads.classifier.kernel[...] != 0.0)


def test_probe_with_hidden_layer_trains_under_nnx_grad() -> None:
    """Regression: a hidden-layer probe must survive an nnx-transformed grad step.

    A leading ``self.hidden = None`` used to register ``hidden`` as a static field, so
    the MLP head's Linear conflicted (data vs static) when nnx re-merged the module
    inside a training transform. The gradient step must now run and reach the head.
    """
    probe = LinearEmbeddingProbe(
        EmbeddingProbeConfig(input_dim=8, n_classes=3, hidden_dim=16),
        rngs=nnx.Rngs(0),
    )
    embeddings = jnp.ones((5, 8), dtype=jnp.float32)
    labels = jnp.array([0, 1, 2, 1, 0], dtype=jnp.int32)

    def loss(module: LinearEmbeddingProbe) -> jax.Array:
        logits = module.apply({"embeddings": embeddings}, {}, None)[0]["logits"]
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(5), labels])

    grads = nnx.grad(loss)(probe)
    leaves = jax.tree.leaves(nnx.state(grads))
    assert leaves  # gradient reached the hidden + classifier parameters
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
