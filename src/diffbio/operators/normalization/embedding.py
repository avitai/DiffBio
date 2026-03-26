"""Sequence embedding operators.

This module provides operators for converting one-hot encoded
DNA sequences into dense embeddings using convolutional networks.

Key technique: Use 1D convolutions to extract local sequence features,
then aggregate into a fixed-size representation.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SequenceEmbeddingConfig(OperatorConfig):
    """Configuration for SequenceEmbedding.

    Attributes:
        embedding_dim: Dimension of output embedding.
        method: Embedding method ("conv" for convolutional).
        kernel_size: Convolution kernel size.
        num_conv_layers: Number of convolutional layers.
    """

    embedding_dim: int = 64
    method: str = "conv"
    kernel_size: int = 7
    num_conv_layers: int = 3


class SequenceEmbedding(OperatorModule):
    """Convolutional sequence embedding operator.

    This operator converts one-hot encoded DNA sequences into dense
    embeddings using a stack of 1D convolutions followed by global
    average pooling.

    The architecture:
    1. Input: one-hot sequence (length, 4)
    2. 1D convolutions with ReLU activation
    3. Per-position features (length, embedding_dim)
    4. Global average pooling -> fixed embedding (embedding_dim,)

    Args:
        config: SequenceEmbeddingConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = SequenceEmbeddingConfig(embedding_dim=64)
        embedder = SequenceEmbedding(config, rngs=nnx.Rngs(42))
        data = {"sequence": encoded_seq}
        result, state, meta = embedder.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: SequenceEmbeddingConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the sequence embedding operator.

        Args:
            config: Embedding configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.embedding_dim = config.embedding_dim
        self.kernel_size = config.kernel_size

        # Build convolutional layers
        # Using Linear layers to simulate 1D convolution via sliding windows
        # This is more compatible with varying sequence lengths
        alphabet_size = 4
        conv_layers: list[nnx.Linear] = []

        # First layer: alphabet -> embedding_dim
        # Input features: kernel_size * alphabet_size
        first_in = config.kernel_size * alphabet_size
        conv_layers.append(
            nnx.Linear(in_features=first_in, out_features=config.embedding_dim, rngs=rngs)
        )

        # Subsequent layers: embedding_dim -> embedding_dim
        for _ in range(config.num_conv_layers - 1):
            conv_layers.append(
                nnx.Linear(
                    in_features=config.kernel_size * config.embedding_dim,
                    out_features=config.embedding_dim,
                    rngs=rngs,
                )
            )

        self.conv_layers = nnx.List(conv_layers)

    def _extract_windows(
        self,
        sequence: Float[Array, "length features"],
        kernel_size: int,
    ) -> Float[Array, "length window_features"]:
        """Extract sliding windows from sequence.

        Args:
            sequence: Input sequence (length, features).
            kernel_size: Size of the sliding window.

        Returns:
            Windows of shape (length, kernel_size * features).
        """
        seq_len, num_features = sequence.shape
        half_k = kernel_size // 2

        # Pad sequence for edge handling
        padded = jnp.pad(sequence, ((half_k, half_k), (0, 0)), mode="constant", constant_values=0.0)

        # Extract windows using vmap
        def extract_window(center: Array | int) -> Float[Array, "window_features"]:
            window = jax.lax.dynamic_slice(padded, (center, 0), (kernel_size, num_features))
            return window.flatten()

        windows = jax.vmap(extract_window)(jnp.arange(seq_len))
        return windows

    def _apply_conv_layer(
        self,
        sequence: Float[Array, "length features"],
        layer: nnx.Linear,
        kernel_size: int,
    ) -> Float[Array, "length out_features"]:
        """Apply a convolutional layer using sliding windows.

        Args:
            sequence: Input sequence.
            layer: Linear layer to apply to each window.
            kernel_size: Window size.

        Returns:
            Output features at each position.
        """
        # Extract windows
        windows = self._extract_windows(sequence, kernel_size)

        # Apply linear transformation to each window
        output = jax.vmap(layer)(windows)

        return output

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply sequence embedding to sequence data.

        This method extracts dense embeddings from one-hot encoded
        DNA sequences using convolutional feature extraction.

        Args:
            data: Dictionary containing:
                - "sequence": One-hot encoded sequence (length, alphabet_size)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used (deterministic operator)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "sequence": Original sequence
                    - "embedding": Global sequence embedding (embedding_dim,)
                    - "position_embeddings": Per-position features (length, embedding_dim)
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        sequence = data["sequence"]

        # Apply first conv layer
        x = self._apply_conv_layer(sequence, self.conv_layers[0], self.kernel_size)
        x = nnx.relu(x)

        # Apply remaining conv layers
        for layer in self.conv_layers[1:]:
            x = self._apply_conv_layer(x, layer, self.kernel_size)
            x = nnx.relu(x)

        # x is now (length, embedding_dim)
        position_embeddings = x

        # Global average pooling to get fixed-size embedding
        embedding = jnp.mean(position_embeddings, axis=0)

        # Build output data
        transformed_data = {
            "sequence": sequence,
            "embedding": embedding,
            "position_embeddings": position_embeddings,
        }

        return transformed_data, state, metadata
