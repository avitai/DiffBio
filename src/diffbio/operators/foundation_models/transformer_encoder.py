"""Transformer-based sequence encoder for DNA/RNA foundation models.

This module provides a differentiable transformer encoder following
DNABERT/RNA-FM architecture patterns. The encoder converts one-hot
encoded nucleotide sequences into dense embeddings suitable for
downstream bioinformatics tasks.

Key features:

- Multi-head self-attention for capturing sequence dependencies
- Sinusoidal positional encoding for position awareness
- Configurable architecture (layers, heads, dimensions)
- Multiple pooling strategies (mean, CLS token)
- Fully differentiable for gradient-based optimization

References:
    - DNABERT: Ji et al. (2021) Bioinformatics
    - RNA-FM: Chen et al. (2022) Nature Methods
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from artifex.generative_models.core.layers import TransformerEncoder
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.core.base_operators import SequenceOperator
from diffbio.operators.foundation_models.contracts import (
    FoundationEmbeddingMixin,
    FoundationEmbeddingOperatorConfig,
    FoundationModelKind,
    PoolingStrategy,
    register_foundation_model,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformerSequenceEncoderConfig(FoundationEmbeddingOperatorConfig):
    """Configuration for TransformerSequenceEncoder.

    Attributes:
        hidden_dim: Dimension of hidden states and embeddings.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        intermediate_dim: Dimension of feed-forward intermediate layer.
        max_length: Maximum sequence length for positional encoding.
        alphabet_size: Size of nucleotide alphabet (4 for DNA/RNA).
        dropout_rate: Dropout rate for regularization.
        pooling: Pooling strategy for sequence embedding ("mean" or "cls").
        adapter_mode: Integration mode for the encoder artifact.
        artifact_id: Identifier for the encoder artifact/version.
        preprocessing_version: Version tag for sequence preprocessing.
        input_embedding_type: Type of input embedding. "linear" projects
            one-hot encoded input via nnx.Linear. "token_embedding" uses
            nnx.Embed for integer token ID input.
        vocab_size: Vocabulary size for token embedding mode. Required
            when input_embedding_type is "token_embedding", ignored for "linear".
    """

    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    intermediate_dim: int = 1024
    max_length: int = 512
    alphabet_size: int = 4
    dropout_rate: float = 0.1
    pooling: Literal["mean", "cls"] = "mean"
    artifact_id: str = "diffbio.transformer_sequence_encoder"
    preprocessing_version: str = "one_hot_v1"
    input_embedding_type: Literal["linear", "token_embedding"] = "linear"
    vocab_size: int | None = None


class TransformerSequenceEncoder(FoundationEmbeddingMixin, SequenceOperator):
    """Transformer-based encoder for DNA/RNA sequences.

    This operator implements a BERT-style transformer encoder that
    converts nucleotide sequences into dense embeddings. The architecture
    follows DNABERT and RNA-FM patterns.

    Uses artifex's TransformerEncoder for the core transformer layers,
    following the DRY principle.

    Supports two input embedding modes:

    - "linear" (default): Projects one-hot encoded input (seq_len, alphabet_size)
      via nnx.Linear. This is the standard mode for continuous one-hot input.
    - "token_embedding": Embeds integer token IDs (seq_len,) via nnx.Embed.
      Useful for gene-token foundation models and tokenized input.

    The encoder produces:

    - Global sequence embedding via mean pooling or CLS token
    - Per-position embeddings for fine-grained analysis

    Args:
        config: TransformerSequenceEncoderConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = TransformerSequenceEncoderConfig(hidden_dim=256)
        encoder = TransformerSequenceEncoder(config, rngs=nnx.Rngs(42))
        data = {"sequence": one_hot_sequence}
        result, state, meta = encoder.apply(data, {}, None)
        embeddings = result["embeddings"]
        ```
    """

    foundation_model_kind = FoundationModelKind.SEQUENCE_TRANSFORMER

    def __init__(
        self,
        config: TransformerSequenceEncoderConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the transformer encoder.

        Args:
            config: Encoder configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        # Ensure dropout stream exists for artifex transformer
        if config.dropout_rate > 0 and "dropout" not in rngs:
            rngs = nnx.Rngs(params=rngs.params(), dropout=jax.random.key(1))

        # Input projection: alphabet_size -> hidden_dim (or token embedding)
        if config.input_embedding_type == "token_embedding":
            if config.vocab_size is None:
                raise ValueError(
                    "vocab_size must be specified when input_embedding_type is 'token_embedding'"
                )
            self.input_projection = nnx.Embed(
                num_embeddings=config.vocab_size,
                features=config.hidden_dim,
                rngs=rngs,
            )
        else:
            self.input_projection = nnx.Linear(
                config.alphabet_size,
                config.hidden_dim,
                rngs=rngs,
            )

        # CLS token embedding (learnable)
        self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (config.hidden_dim,)) * 0.02)

        # Compute MLP ratio from intermediate_dim
        mlp_ratio = config.intermediate_dim / config.hidden_dim

        # Use artifex's TransformerEncoder (DRY principle)
        self.transformer = TransformerEncoder(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=config.dropout_rate,
            attention_dropout_rate=0.0,
            max_len=config.max_length + 1,  # +1 for CLS token
            pos_encoding_type="sinusoidal",
            rngs=rngs,
        )

    def foundation_pooling_strategy(self) -> PoolingStrategy:
        """Return the pooling strategy for the global sequence embedding."""
        return PoolingStrategy(self.config.pooling)

    def get_positional_encoding(
        self,
        seq_len: int,
    ) -> Float[Array, "seq_len hidden_dim"]:
        """Generate sinusoidal positional encoding.

        This is provided for compatibility but the transformer uses
        internal positional encoding.

        Args:
            seq_len: Sequence length.

        Returns:
            Positional encoding matrix.
        """
        hidden_dim = self.config.hidden_dim
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, hidden_dim, 2) * -(jnp.log(10000.0) / hidden_dim))

        pe = jnp.zeros((seq_len, hidden_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        return pe

    def _encode_single(
        self,
        sequence: Array,
        mask: Float[Array, "seq_len"] | None = None,
    ) -> tuple[Float[Array, "hidden_dim"], Float[Array, "seq_len hidden_dim"]]:
        """Encode a single sequence.

        Args:
            sequence: Input sequence. One-hot encoded (seq_len, alphabet_size)
                for linear mode, or integer token IDs (seq_len,) for token
                embedding mode.
            mask: Optional attention mask.

        Returns:
            Tuple of (global_embedding, token_embeddings).
        """
        # Project input to hidden dimension
        hidden = self.input_projection(sequence)

        # Add batch dimension for transformer (expects [batch, seq, hidden])
        hidden = hidden[None, :, :]  # (1, seq_len, hidden_dim)

        # Prepend CLS token for CLS pooling
        if self.config.pooling == "cls":
            cls_token = self.cls_token[...][None, None, :]  # (1, 1, hidden_dim)
            hidden = jnp.concatenate([cls_token, hidden], axis=1)

            # Extend mask if provided
            if mask is not None:
                mask = jnp.concatenate([jnp.ones(1), mask], axis=0)

        # Ensure mask has batch dimension for artifex transformer
        if mask is not None:
            mask = mask[None, :]  # Add batch dim: (1, seq_len)

        # Apply transformer (deterministic=True for no dropout)
        hidden = self.transformer(hidden, mask=mask, deterministic=True)

        # Remove batch dimension
        hidden = hidden[0]  # (seq_len, hidden_dim)

        # Extract embeddings based on pooling strategy
        if self.config.pooling == "cls":
            # Use CLS token (first position)
            global_embedding = hidden[0]
            position_embeddings = hidden[1:]  # Remove CLS token
        else:
            # Mean pooling
            if mask is not None:
                # Mask is (1, seq_len), get the 1D version
                mask_1d = mask[0]
                mask_expanded = mask_1d[:, None]
                masked_hidden = hidden * mask_expanded
                global_embedding = jnp.sum(masked_hidden, axis=0) / (jnp.sum(mask_1d) + 1e-9)
            else:
                global_embedding = jnp.mean(hidden, axis=0)
            position_embeddings = hidden

        return global_embedding, position_embeddings

    def _encode_batch(
        self,
        sequences: Array,
        masks: Float[Array, "batch seq_len"] | None = None,
    ) -> tuple[
        Float[Array, "batch hidden_dim"],
        Float[Array, "batch seq_len hidden_dim"],
    ]:
        """Encode a batch of sequences.

        Args:
            sequences: Batch of input sequences. One-hot encoded
                (batch, seq_len, alphabet_size) for linear mode, or integer
                token IDs (batch, seq_len) for token embedding mode.
            masks: Optional attention masks.

        Returns:
            Tuple of (global_embeddings, token_embeddings).
        """
        batch_size = sequences.shape[0]

        # Project input to hidden dimension
        hidden = jax.vmap(self.input_projection)(sequences)

        # Prepend CLS token for CLS pooling
        if self.config.pooling == "cls":
            cls_token = self.cls_token[...][None, None, :]  # (1, 1, hidden_dim)
            cls_tokens = jnp.broadcast_to(cls_token, (batch_size, 1, self.config.hidden_dim))
            hidden = jnp.concatenate([cls_tokens, hidden], axis=1)

            # Extend masks if provided
            if masks is not None:
                mask_prefix = jnp.ones((batch_size, 1))
                masks = jnp.concatenate([mask_prefix, masks], axis=1)

        # Apply transformer
        hidden = self.transformer(hidden, mask=masks, deterministic=True)

        # Extract embeddings based on pooling strategy
        if self.config.pooling == "cls":
            global_embeddings = hidden[:, 0]
            position_embeddings = hidden[:, 1:]
        else:
            if masks is not None:
                mask_expanded = masks[:, :, None]
                masked_hidden = hidden * mask_expanded
                global_embeddings = jnp.sum(masked_hidden, axis=1) / (
                    jnp.sum(masks, axis=1, keepdims=True) + 1e-9
                )
            else:
                global_embeddings = jnp.mean(hidden, axis=1)
            position_embeddings = hidden

        return global_embeddings, position_embeddings

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply transformer encoding to sequence data.

        This method encodes DNA/RNA sequences into dense embeddings using
        a transformer encoder architecture.

        Input shape depends on ``input_embedding_type``:

        - "linear": one-hot ``(seq_len, alphabet_size)`` or
          ``(batch, seq_len, alphabet_size)``
        - "token_embedding": integer token IDs ``(seq_len,)`` or
          ``(batch, seq_len)``

        Args:
            data: Dictionary containing:
                - "sequence": Encoded sequence(s) (see above for shapes)
                - "attention_mask": Optional mask (seq_len,) or (batch, seq_len)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - All original keys from data
                    - "embeddings": Global sequence embedding
                    - "token_embeddings": Per-position hidden states
                    - "foundation_model": Canonical artifact metadata
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        del random_params, stats  # Unused

        sequence = data["sequence"]
        mask = data.get("attention_mask", None)

        is_token_mode = self.config.input_embedding_type == "token_embedding"

        # Determine single vs batch based on input dimensionality:
        # - token mode: single=(seq_len,) ndim=1, batch=(batch, seq_len) ndim=2
        # - linear mode: single=(seq_len, alphabet) ndim=2, batch=(batch, seq_len, alphabet) ndim=3
        single_ndim = 1 if is_token_mode else 2

        if sequence.ndim == single_ndim:
            embeddings, token_embeddings = self._encode_single(sequence, mask)
        else:
            embeddings, token_embeddings = self._encode_batch(sequence, mask)

        transformed_data = self.foundation_result(
            data,
            embeddings,
            token_embeddings=token_embeddings,
        )

        return transformed_data, state, metadata


def _create_sequence_encoder(
    alphabet_size: int,
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    intermediate_dim: int | None = None,
    max_length: int = 512,
    dropout_rate: float = 0.1,
    pooling: Literal["mean", "cls"] = "mean",
    *,
    rngs: nnx.Rngs | None = None,
) -> TransformerSequenceEncoder:
    """Create a transformer sequence encoder with given alphabet size.

    Args:
        alphabet_size: Size of input alphabet (e.g., 4 for DNA/RNA).
        hidden_dim: Dimension of hidden states.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        intermediate_dim: FFN intermediate dimension (default: 4 * hidden_dim).
        max_length: Maximum sequence length.
        dropout_rate: Dropout rate.
        pooling: Pooling strategy.
        rngs: Random number generators.

    Returns:
        Configured TransformerSequenceEncoder.
    """
    if intermediate_dim is None:
        intermediate_dim = 4 * hidden_dim

    if rngs is None:
        rngs = nnx.Rngs(0)

    config = TransformerSequenceEncoderConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        max_length=max_length,
        alphabet_size=alphabet_size,
        dropout_rate=dropout_rate,
        pooling=pooling,
    )

    return TransformerSequenceEncoder(config, rngs=rngs)


def create_dna_encoder(
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    intermediate_dim: int | None = None,
    max_length: int = 512,
    dropout_rate: float = 0.1,
    pooling: Literal["mean", "cls"] = "mean",
    *,
    rngs: nnx.Rngs | None = None,
) -> TransformerSequenceEncoder:
    """Create a transformer encoder for DNA sequences.

    Factory function for creating a DNA sequence encoder with
    sensible defaults for DNA processing.

    Args:
        hidden_dim: Dimension of hidden states.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        intermediate_dim: FFN intermediate dimension (default: 4 * hidden_dim).
        max_length: Maximum sequence length.
        dropout_rate: Dropout rate.
        pooling: Pooling strategy.
        rngs: Random number generators.

    Returns:
        Configured TransformerSequenceEncoder for DNA.

    Example:
        ```python
        encoder = create_dna_encoder(hidden_dim=256, num_layers=6)
        data = {"sequence": dna_one_hot}
        result, _, _ = encoder.apply(data, {}, None)
        embeddings = result["embeddings"]
        ```
    """
    return _create_sequence_encoder(
        alphabet_size=4,  # A, C, G, T
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        max_length=max_length,
        dropout_rate=dropout_rate,
        pooling=pooling,
        rngs=rngs,
    )


register_foundation_model(
    FoundationModelKind.SEQUENCE_TRANSFORMER,
    TransformerSequenceEncoder,
)


def create_rna_encoder(
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    intermediate_dim: int | None = None,
    max_length: int = 512,
    dropout_rate: float = 0.1,
    pooling: Literal["mean", "cls"] = "mean",
    *,
    rngs: nnx.Rngs | None = None,
) -> TransformerSequenceEncoder:
    """Create a transformer encoder for RNA sequences.

    Factory function for creating an RNA sequence encoder with
    sensible defaults for RNA processing.

    Args:
        hidden_dim: Dimension of hidden states.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        intermediate_dim: FFN intermediate dimension (default: 4 * hidden_dim).
        max_length: Maximum sequence length.
        dropout_rate: Dropout rate.
        pooling: Pooling strategy.
        rngs: Random number generators.

    Returns:
        Configured TransformerSequenceEncoder for RNA.

    Example:
        ```python
        encoder = create_rna_encoder(hidden_dim=640, num_layers=12)
        data = {"sequence": rna_one_hot}
        result, _, _ = encoder.apply(data, {}, None)
        ```
    """
    return _create_sequence_encoder(
        alphabet_size=4,  # A, C, G, U
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        max_length=max_length,
        dropout_rate=dropout_rate,
        pooling=pooling,
        rngs=rngs,
    )
