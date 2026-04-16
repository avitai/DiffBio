"""Contextual epigenomics operator for sequence, TF, and chromatin inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from artifex.generative_models.core.layers import TransformerEncoder
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx

from diffbio.operators._transformer_validation import TransformerEncoderShapeValidationMixin


@dataclass(frozen=True)
class _ContextualEncoderConfig:
    """Sequence encoder hyperparameters."""

    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    intermediate_dim: int = 256
    max_length: int = 512
    dropout_rate: float = 0.0


@dataclass(frozen=True)
class _ContextualTaskConfig:
    """Task and conditioning configuration."""

    num_tf_features: int = 8
    num_outputs: int = 1
    use_tf_context: bool = True
    use_chromatin_guidance: bool = False
    chromatin_guidance_weight: float = 0.1


@dataclass(frozen=True)
class ContextualEpigenomicsConfig(
    _ContextualEncoderConfig,
    _ContextualTaskConfig,
    TransformerEncoderShapeValidationMixin,
    OperatorConfig,
):
    """Configuration for the contextual epigenomics operator."""

    def __post_init__(self) -> None:
        """Validate the operator configuration."""
        super().__post_init__()
        if self.num_outputs < 1:
            raise ValueError("num_outputs must be at least 1.")
        if self.num_tf_features < 1:
            raise ValueError("num_tf_features must be at least 1.")
        if self.chromatin_guidance_weight < 0.0:
            raise ValueError("chromatin_guidance_weight must be non-negative.")


class ContextualEpigenomicsOperator(OperatorModule):
    """Single operator path for sequence-only and contextual epigenomics modes."""

    def __init__(
        self,
        config: ContextualEpigenomicsConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        super().__init__(config, rngs=rngs)

        if rngs is None:
            rngs = nnx.Rngs(0)
        if config.dropout_rate > 0 and "dropout" not in rngs:
            rngs = nnx.Rngs(params=rngs.params(), dropout=jax.random.key(1))

        self.config = config
        self.sequence_projection = nnx.Linear(4, config.hidden_dim, rngs=rngs)
        self.tf_scale = nnx.Linear(config.num_tf_features, config.hidden_dim, rngs=rngs)
        self.tf_shift = nnx.Linear(config.num_tf_features, config.hidden_dim, rngs=rngs)
        self.transformer = TransformerEncoder(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            mlp_ratio=config.intermediate_dim / config.hidden_dim,
            dropout_rate=config.dropout_rate,
            attention_dropout_rate=0.0,
            max_len=config.max_length,
            pos_encoding_type="sinusoidal",
            rngs=rngs,
        )
        self.output_head = nnx.Linear(config.hidden_dim, config.num_outputs, rngs=rngs)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Apply the contextual epigenomics operator to one batch."""
        del random_params, stats

        sequence = jnp.asarray(data["sequence"], dtype=jnp.float32)
        tf_context = data.get("tf_context")
        chromatin_contacts = data.get("chromatin_contacts")
        sequence_mask = data.get("sequence_mask")

        (
            sequence,
            tf_context,
            chromatin_contacts,
            sequence_mask,
            squeeze_batch,
        ) = _canonicalize_contextual_inputs(
            sequence=sequence,
            tf_context=tf_context,
            chromatin_contacts=chromatin_contacts,
            sequence_mask=sequence_mask,
        )

        hidden = self.sequence_projection(sequence)
        if self.config.use_tf_context:
            if tf_context is None:
                raise ValueError("tf_context is required when use_tf_context=True.")
            hidden = _apply_tf_conditioning(
                hidden=hidden,
                tf_context=jnp.asarray(tf_context, dtype=jnp.float32),
                tf_scale=self.tf_scale,
                tf_shift=self.tf_shift,
            )

        token_embeddings = self.transformer(
            hidden,
            mask=sequence_mask,
            deterministic=True,
        )
        masked_token_embeddings = token_embeddings * sequence_mask[..., None]
        pooled_embeddings = masked_token_embeddings.sum(axis=1) / jnp.maximum(
            sequence_mask.sum(axis=1, keepdims=True),
            1.0,
        )

        logits = self.output_head(token_embeddings)
        if self.config.num_outputs == 1:
            logits = logits.squeeze(-1)

        chromatin_guidance_loss = jnp.array(0.0, dtype=jnp.float32)
        if self.config.use_chromatin_guidance:
            if chromatin_contacts is None:
                raise ValueError("chromatin_contacts is required when use_chromatin_guidance=True.")
            chromatin_guidance_loss = compute_chromatin_guidance_loss(
                token_embeddings=token_embeddings,
                chromatin_contacts=jnp.asarray(chromatin_contacts, dtype=jnp.float32),
                sequence_mask=sequence_mask,
            )

        result = {
            **data,
            "embeddings": pooled_embeddings,
            "token_embeddings": token_embeddings,
            "logits": logits,
            "chromatin_guidance_loss": chromatin_guidance_loss,
        }
        if squeeze_batch:
            result["embeddings"] = pooled_embeddings.squeeze(0)
            result["token_embeddings"] = token_embeddings.squeeze(0)
            result["logits"] = logits.squeeze(0)

        return result, state, metadata


def _canonicalize_contextual_inputs(
    *,
    sequence: jnp.ndarray,
    tf_context: Any,
    chromatin_contacts: Any,
    sequence_mask: Any,
) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray, bool]:
    """Normalize contextual epigenomics inputs to batched tensors."""
    if sequence.ndim not in (2, 3) or sequence.shape[-1] != 4:
        raise ValueError("sequence must have shape (length, 4) or (batch, length, 4).")

    squeeze_batch = sequence.ndim == 2
    if squeeze_batch:
        sequence = sequence[None, ...]

    batch_size, sequence_length, _ = sequence.shape

    tf_tensor: jnp.ndarray | None = None
    if tf_context is not None:
        tf_tensor = jnp.asarray(tf_context, dtype=jnp.float32)
        if tf_tensor.ndim == 1:
            tf_tensor = tf_tensor[None, ...]
        if tf_tensor.ndim != 2 or tf_tensor.shape[0] != batch_size:
            raise ValueError("tf_context must have shape (features,) or (batch, features).")

    chromatin_tensor: jnp.ndarray | None = None
    if chromatin_contacts is not None:
        chromatin_tensor = jnp.asarray(chromatin_contacts, dtype=jnp.float32)
        if chromatin_tensor.ndim == 2:
            chromatin_tensor = chromatin_tensor[None, ...]
        if chromatin_tensor.ndim != 3 or chromatin_tensor.shape[0] != batch_size:
            raise ValueError(
                "chromatin_contacts must have shape (length, length) or (batch, length, length)."
            )
        if (
            chromatin_tensor.shape[1] != sequence_length
            or chromatin_tensor.shape[2] != sequence_length
        ):
            raise ValueError("chromatin_contacts must align with the sequence length.")

    if sequence_mask is None:
        mask_tensor = jnp.ones((batch_size, sequence_length), dtype=jnp.float32)
    else:
        mask_tensor = jnp.asarray(sequence_mask, dtype=jnp.float32)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor[None, ...]
        if mask_tensor.shape != (batch_size, sequence_length):
            raise ValueError("sequence_mask must have shape (length,) or (batch, length).")

    return sequence, tf_tensor, chromatin_tensor, mask_tensor, squeeze_batch


def _apply_tf_conditioning(
    *,
    hidden: jnp.ndarray,
    tf_context: jnp.ndarray,
    tf_scale: nnx.Linear,
    tf_shift: nnx.Linear,
) -> jnp.ndarray:
    """Apply FiLM-style TF conditioning to token embeddings."""
    scale = jnp.tanh(tf_scale(tf_context))[:, None, :]
    shift = tf_shift(tf_context)[:, None, :]
    return hidden * (1.0 + scale) + shift


def compute_chromatin_guidance_loss(
    *,
    token_embeddings: jnp.ndarray,
    chromatin_contacts: jnp.ndarray,
    sequence_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute a chromatin-consistency loss over token embeddings."""
    normalized_embeddings = token_embeddings / jnp.maximum(
        jnp.linalg.norm(token_embeddings, axis=-1, keepdims=True),
        1e-6,
    )
    similarity = jnp.einsum(
        "bld,bmd->blm",
        normalized_embeddings,
        normalized_embeddings,
    )
    predicted_contacts = jax.nn.sigmoid(similarity)
    pair_mask = sequence_mask[:, :, None] * sequence_mask[:, None, :]
    squared_error = jnp.square(predicted_contacts - chromatin_contacts) * pair_mask
    return squared_error.sum() / jnp.maximum(pair_mask.sum(), 1.0)


def compute_contextual_epigenomics_loss(
    model: ContextualEpigenomicsOperator,
    data: dict[str, Any],
) -> dict[str, jnp.ndarray]:
    """Compute supervised plus optional chromatin-guidance losses."""
    result = model.apply(data, {}, None)[0]
    logits = jnp.asarray(result["logits"], dtype=jnp.float32)
    targets = jnp.asarray(data["targets"])

    if logits.ndim == targets.ndim:
        supervised = optax.sigmoid_binary_cross_entropy(
            logits,
            targets.astype(jnp.float32),
        ).mean()
    else:
        supervised = optax.softmax_cross_entropy_with_integer_labels(
            logits,
            targets.astype(jnp.int32),
        ).mean()

    chromatin_guidance = jnp.asarray(result["chromatin_guidance_loss"], dtype=jnp.float32)
    total = supervised + model.config.chromatin_guidance_weight * chromatin_guidance
    return {
        "supervised": supervised,
        "chromatin_guidance": chromatin_guidance,
        "total": total,
    }
