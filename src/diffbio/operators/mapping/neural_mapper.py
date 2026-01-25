"""Neural network-based read mapper for differentiable alignment.

This module provides a neural network approach to read mapping that
enables gradient flow through the mapping process.

Key technique: Uses cross-attention between read and reference embeddings
to compute soft alignment scores, enabling end-to-end differentiable mapping.

Applications: Differentiable read mapping for joint optimization with
downstream variant calling or assembly pipelines.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.configs import TemperatureConfig
from diffbio.core.base_operators import TemperatureOperator


@dataclass
class NeuralReadMapperConfig(TemperatureConfig):
    """Configuration for NeuralReadMapper.

    Attributes:
        read_length: Expected read length.
        reference_window: Reference window size.
        embedding_dim: Dimension of sequence embeddings.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        dropout_rate: Dropout rate for regularization.
        temperature: Temperature for softmax operations.
    """

    read_length: int = 150
    reference_window: int = 500
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.1
    temperature: float = 1.0


class SequenceEncoder(nnx.Module):
    """Encoder for DNA sequences.

    Converts one-hot encoded sequences to dense embeddings
    with positional encoding.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_length: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the sequence encoder.

        Args:
            embedding_dim: Output embedding dimension.
            max_length: Maximum sequence length.
            rngs: Random number generators.
        """
        super().__init__()

        # Project from one-hot (4 bases) to embedding dim
        self.input_projection = nnx.Linear(
            in_features=4,
            out_features=embedding_dim,
            rngs=rngs,
        )

        # Learnable positional encoding
        key = rngs.params()
        self.positional_encoding = nnx.Param(
            jax.random.normal(key, (max_length, embedding_dim)) * 0.02
        )

    def __call__(
        self,
        sequence: Float[Array, "batch length 4"],
    ) -> Float[Array, "batch length embedding_dim"]:
        """Encode a one-hot sequence.

        Args:
            sequence: One-hot encoded DNA sequence.

        Returns:
            Dense sequence embeddings with positional encoding.
        """
        # Project to embedding dimension
        embeddings = self.input_projection(sequence)

        # Add positional encoding
        seq_len = sequence.shape[1]
        pos_enc = self.positional_encoding[:seq_len]
        embeddings = embeddings + pos_enc

        return embeddings


class CrossAttentionLayer(nnx.Module):
    """Cross-attention layer for read-reference alignment."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the cross-attention layer.

        Args:
            embedding_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout_rate: Dropout rate.
            rngs: Random number generators.
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Query, Key, Value projections
        self.query_proj = nnx.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            rngs=rngs,
        )
        self.key_proj = nnx.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            rngs=rngs,
        )
        self.value_proj = nnx.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            rngs=rngs,
        )
        self.output_proj = nnx.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            rngs=rngs,
        )

        # Dropout
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        query: Float[Array, "batch query_len dim"],
        key_value: Float[Array, "batch kv_len dim"],
        *,
        deterministic: bool = True,
    ) -> Float[Array, "batch query_len dim"]:
        """Apply cross-attention.

        Args:
            query: Query embeddings (read).
            key_value: Key/Value embeddings (reference).
            deterministic: If True, disable dropout.

        Returns:
            Attended embeddings.
        """
        batch_size = query.shape[0]
        query_len = query.shape[1]
        kv_len = key_value.shape[1]

        # Project Q, K, V
        Q = self.query_proj(query)
        K = self.key_proj(key_value)
        V = self.value_proj(key_value)

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, query_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, kv_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, kv_len, self.num_heads, self.head_dim)

        # Transpose to (batch, heads, length, dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Compute attention scores
        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", Q, K) * self.scale
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)

        # Apply dropout to attention
        if self.dropout is not None and not deterministic:
            attn_probs = self.dropout(attn_probs)

        # Compute attended values
        attended = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, V)

        # Reshape back
        attended = attended.transpose(0, 2, 1, 3)
        attended = attended.reshape(batch_size, query_len, -1)

        # Output projection
        output = self.output_proj(attended)

        return output


class TransformerBlock(nnx.Module):
    """Transformer block with cross-attention and feedforward."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the transformer block.

        Args:
            embedding_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout_rate: Dropout rate.
            rngs: Random number generators.
        """
        super().__init__()

        self.cross_attention = CrossAttentionLayer(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

        self.layer_norm1 = nnx.LayerNorm(
            num_features=embedding_dim,
            rngs=rngs,
        )
        self.layer_norm2 = nnx.LayerNorm(
            num_features=embedding_dim,
            rngs=rngs,
        )

        # Feedforward
        self.ff_linear1 = nnx.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim * 4,
            rngs=rngs,
        )
        self.ff_linear2 = nnx.Linear(
            in_features=embedding_dim * 4,
            out_features=embedding_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        read_embeddings: Float[Array, "batch read_len dim"],
        ref_embeddings: Float[Array, "batch ref_len dim"],
        *,
        deterministic: bool = True,
    ) -> Float[Array, "batch read_len dim"]:
        """Apply transformer block.

        Args:
            read_embeddings: Read embeddings.
            ref_embeddings: Reference embeddings.
            deterministic: If True, disable dropout.

        Returns:
            Transformed read embeddings.
        """
        # Cross-attention with residual
        attended = self.cross_attention(
            read_embeddings, ref_embeddings, deterministic=deterministic
        )
        x = self.layer_norm1(read_embeddings + attended)

        # Feedforward with residual
        ff_out = self.ff_linear2(nnx.gelu(self.ff_linear1(x)))
        x = self.layer_norm2(x + ff_out)

        return x


class NeuralReadMapper(TemperatureOperator):
    """Neural network-based read mapper.

    This operator uses cross-attention between read and reference
    embeddings to compute soft alignment scores, enabling fully
    differentiable read mapping.

    Algorithm:
    1. Encode read and reference with positional embeddings
    2. Apply transformer layers with cross-attention
    3. Compute position-wise alignment scores
    4. Apply softmax for position probabilities
    5. Compute mapping quality from confidence

    Inherits from TemperatureOperator to get:

    - _temperature property for temperature-controlled smoothing
    - soft_max() for logsumexp-based smooth maximum
    - soft_argmax() for soft position selection

    Args:
        config: NeuralReadMapperConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = NeuralReadMapperConfig(embedding_dim=64)
        mapper = NeuralReadMapper(config, rngs=nnx.Rngs(42))
        data = {"read": read_onehot, "reference": ref_onehot}
        result, state, meta = mapper.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: NeuralReadMapperConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the neural read mapper.

        Args:
            config: Mapper configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        # Temperature is now managed by TemperatureOperator via self._temperature

        # Read encoder
        self.read_encoder = SequenceEncoder(
            embedding_dim=config.embedding_dim,
            max_length=config.read_length,
            rngs=rngs,
        )

        # Reference encoder
        self.ref_encoder = SequenceEncoder(
            embedding_dim=config.embedding_dim,
            max_length=config.reference_window,
            rngs=rngs,
        )

        # Transformer layers
        self.transformer_layers = nnx.List(
            [
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    dropout_rate=config.dropout_rate,
                    rngs=rngs,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Output projections
        self.score_projection = nnx.Linear(
            in_features=config.embedding_dim,
            out_features=1,
            rngs=rngs,
        )

        self.quality_projection = nnx.Linear(
            in_features=config.embedding_dim,
            out_features=1,
            rngs=rngs,
        )

    def compute_alignment_scores(
        self,
        read: Float[Array, "batch read_len 4"],
        reference: Float[Array, "batch ref_len 4"],
        *,
        deterministic: bool = True,
    ) -> tuple[
        Float[Array, "batch ref_len"],
        Float[Array, "batch embedding_dim"],
    ]:
        """Compute alignment scores for each reference position.

        Args:
            read: One-hot encoded read.
            reference: One-hot encoded reference.
            deterministic: If True, disable dropout.

        Returns:
            Tuple of (position scores, read summary embedding).
        """
        # Encode sequences
        read_emb = self.read_encoder(read)
        ref_emb = self.ref_encoder(reference)

        # Apply transformer layers
        for layer in self.transformer_layers:
            read_emb = layer(read_emb, ref_emb, deterministic=deterministic)

        # Global read representation (mean pooling)
        read_summary = jnp.mean(read_emb, axis=1)  # (batch, dim)

        # Compute scores: dot product of read summary with each reference position
        # (batch, dim) @ (batch, ref_len, dim).T -> (batch, ref_len)
        scores = jnp.einsum("bd,brd->br", read_summary, ref_emb)

        return scores, read_summary

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply neural read mapping.

        Args:
            data: Dictionary containing:
                - "read": One-hot encoded read (batch, read_len, 4)
                - "reference": One-hot encoded reference (batch, ref_len, 4)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "read": Original read
                    - "reference": Original reference
                    - "alignment_scores": Scores for each reference position
                    - "position_probs": Softmax probabilities over positions
                    - "best_position": Most likely mapping position
                    - "mapping_quality": Confidence score for mapping
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        read = data["read"]
        reference = data["reference"]

        # Compute alignment scores
        deterministic = not self.config.stochastic
        scores, read_summary = self.compute_alignment_scores(
            read, reference, deterministic=deterministic
        )

        # Position probabilities via softmax
        # Use inherited _temperature property from TemperatureOperator
        position_probs = jax.nn.softmax(scores / self._temperature, axis=-1)

        # Best position (argmax)
        best_position = jnp.argmax(position_probs, axis=-1)

        # Mapping quality: higher when distribution is peaked
        # Use negative entropy as quality measure
        entropy = -jnp.sum(position_probs * jnp.log(position_probs + 1e-10), axis=-1)
        max_entropy = jnp.log(jnp.array(reference.shape[1], dtype=jnp.float32))
        mapping_quality = 1.0 - (entropy / max_entropy)

        # Build output
        transformed_data = {
            "read": read,
            "reference": reference,
            "alignment_scores": scores,
            "position_probs": position_probs,
            "best_position": best_position,
            "mapping_quality": mapping_quality,
        }

        return transformed_data, state, metadata
