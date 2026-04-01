"""Lightweight probing operators for foundation-model embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import PyTree


@dataclass(frozen=True, kw_only=True)
class EmbeddingProbeConfig(OperatorConfig):
    """Configuration for a lightweight embedding probe."""

    input_dim: int
    n_classes: int
    hidden_dim: int | None = None


class LinearEmbeddingProbe(OperatorModule):
    """Small classifier for probing embedding quality on downstream tasks."""

    def __init__(
        self,
        config: EmbeddingProbeConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.hidden = None
        if config.hidden_dim is not None:
            self.hidden = nnx.Linear(
                in_features=config.input_dim,
                out_features=config.hidden_dim,
                rngs=rngs,
            )
            classifier_in_dim = config.hidden_dim
        else:
            classifier_in_dim = config.input_dim

        self.classifier = nnx.Linear(
            in_features=classifier_in_dim,
            out_features=config.n_classes,
            rngs=rngs,
        )

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Predict class probabilities from input embeddings."""
        del random_params, stats

        embeddings = data["embeddings"]
        features = embeddings
        if self.hidden is not None:
            features = nnx.relu(self.hidden(features))

        logits = self.classifier(features)
        probabilities = jax.nn.softmax(logits, axis=-1)
        predicted_labels = jnp.argmax(probabilities, axis=-1)

        transformed_data = {
            **data,
            "logits": logits,
            "probabilities": probabilities,
            "predicted_labels": predicted_labels,
        }
        return transformed_data, state, metadata
