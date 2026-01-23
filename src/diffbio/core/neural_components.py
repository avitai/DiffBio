"""Reusable neural network components for DiffBio.

This module provides neural network building blocks that are specific to
bioinformatics applications and not available in Flax NNX built-ins.

IMPORTANT: For standard components, use Flax NNX built-ins:
- nnx.MultiHeadAttention for attention
- nnx.Linear for dense layers
- nnx.Conv for convolutions
- nnx.LayerNorm for normalization
- nnx.Dropout for dropout
- nnx.Sequential for layer composition

For reusable components, import from artifex:
- PositionalEncoding (sinusoidal, learned, RoPE)
- ResidualBlock1D, ResidualBlock2D
- TransformerBlock
- Various loss functions

This module only provides DiffBio-specific components:
- GumbelSoftmaxModule: Differentiable discrete sampling
- GraphMessagePassing: GNN message passing for graph-structured data
"""

from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

from diffbio.constants import DEFAULT_TEMPERATURE, EPSILON
from diffbio.core.differentiable_ops import gumbel_softmax
from diffbio.utils.nn_utils import get_rng_key

# =============================================================================
# Re-export from artifex (import when available, provide stubs otherwise)
# =============================================================================

from artifex.generative_models.core.layers.positional import (
    PositionalEncoding,
    RotaryPositionalEncoding as RoPE,
    SinusoidalPositionalEncoding,
)
from artifex.generative_models.core.layers.residual import (
    Conv1DResidualBlock as ResidualBlock1D,
    Conv2DResidualBlock as ResidualBlock2D,
)


__all__ = [
    # DiffBio-specific components
    "GumbelSoftmaxModule",
    "GraphMessagePassing",
    # Re-exported from artifex
    "PositionalEncoding",
    "SinusoidalPositionalEncoding",
    "RoPE",
    "ResidualBlock1D",
    "ResidualBlock2D",
]


class GumbelSoftmaxModule(nnx.Module):
    """Neural network module for Gumbel-softmax sampling.

    This module wraps the gumbel_softmax function as an nnx.Module,
    providing differentiable categorical sampling during forward pass.

    Useful for:
    - Discrete latent variable models (VQ-VAE variants)
    - Hard attention mechanisms
    - Discrete sequence generation

    Args:
        temperature: Initial temperature for sampling.
        hard: If True, use straight-through estimator for discrete samples.
        rngs: Flax NNX random number generators.

    Example:
        >>> module = GumbelSoftmaxModule(temperature=0.5, rngs=nnx.Rngs(42))
        >>> logits = jnp.array([[1.0, 2.0, 3.0]])
        >>> samples = module(logits)
    """

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        hard: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize GumbelSoftmaxModule.

        Args:
            temperature: Temperature for Gumbel-softmax.
            hard: Whether to use hard (one-hot) samples.
            rngs: Random number generators.
        """
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        self.rngs = rngs

    def __call__(self, logits: Float[Array, "... n"]) -> Float[Array, "... n"]:
        """Apply Gumbel-softmax sampling.

        Args:
            logits: Unnormalized log-probabilities of shape (..., n).

        Returns:
            Samples of same shape as logits.
        """
        key = get_rng_key(self.rngs, "dropout", fallback_seed=0)
        return gumbel_softmax(logits, key, self.temperature, self.hard)


class GraphMessagePassing(nnx.Module):
    """Graph neural network message passing layer.

    Implements a standard message passing scheme:
    1. Compute messages from source nodes and edge features
    2. Aggregate messages at destination nodes
    3. Update node features with aggregated messages

    Supports different aggregation functions (sum, mean, max).

    Args:
        node_features: Input node feature dimension.
        edge_features: Edge feature dimension.
        hidden_dim: Output hidden dimension.
        aggregation: Aggregation function ("sum", "mean", "max").
        rngs: Flax NNX random number generators.

    Example:
        >>> layer = GraphMessagePassing(
        ...     node_features=32, edge_features=8, hidden_dim=64,
        ...     rngs=nnx.Rngs(42)
        ... )
        >>> node_feat = jnp.ones((5, 32))  # 5 nodes
        >>> edge_feat = jnp.ones((8, 8))   # 8 edges
        >>> edge_index = jnp.array([[0,0,1,1,2,2,3,4], [1,2,2,3,3,4,4,0]])
        >>> output = layer(node_feat, edge_feat, edge_index)
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        aggregation: Literal["sum", "mean", "max"] = "sum",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize GraphMessagePassing layer.

        Args:
            node_features: Input node feature dimension.
            edge_features: Edge feature dimension.
            hidden_dim: Output dimension.
            aggregation: Aggregation method.
            rngs: Random number generators.
        """
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation

        # Message MLP: combines source node and edge features
        message_input_dim = node_features + edge_features
        self.message_linear1 = nnx.Linear(message_input_dim, hidden_dim, rngs=rngs)
        self.message_linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

        # Update MLP: combines node features with aggregated messages
        update_input_dim = node_features + hidden_dim
        self.update_linear1 = nnx.Linear(update_input_dim, hidden_dim, rngs=rngs)
        self.update_linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

    def __call__(
        self,
        node_features: Float[Array, "num_nodes node_feat"],
        edge_features: Float[Array, "num_edges edge_feat"],
        edge_index: Int[Array, "2 num_edges"],
    ) -> Float[Array, "num_nodes hidden_dim"]:
        """Apply message passing.

        Args:
            node_features: Node feature matrix (num_nodes, node_features).
            edge_features: Edge feature matrix (num_edges, edge_features).
            edge_index: Edge indices [source, dest] of shape (2, num_edges).

        Returns:
            Updated node features (num_nodes, hidden_dim).
        """
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]

        # Handle empty graph case
        if num_edges == 0:
            # No messages, just transform node features
            x = self.update_linear1(
                jnp.concatenate(
                    [node_features, jnp.zeros((num_nodes, self.hidden_dim))],
                    axis=-1,
                )
            )
            x = nnx.relu(x)
            return self.update_linear2(x)

        # Extract source and destination node indices
        source_idx = edge_index[0]
        dest_idx = edge_index[1]

        # Get source node features for each edge
        source_features = node_features[source_idx]  # (num_edges, node_feat)

        # Compute messages: MLP(concat(source_features, edge_features))
        message_input = jnp.concatenate([source_features, edge_features], axis=-1)
        messages = self.message_linear1(message_input)
        messages = nnx.relu(messages)
        messages = self.message_linear2(messages)  # (num_edges, hidden_dim)

        # Aggregate messages at destination nodes
        if self.aggregation == "sum":
            aggregated = jax.ops.segment_sum(
                messages, dest_idx, num_segments=num_nodes
            )
        elif self.aggregation == "mean":
            sum_messages = jax.ops.segment_sum(
                messages, dest_idx, num_segments=num_nodes
            )
            counts = jax.ops.segment_sum(
                jnp.ones(num_edges), dest_idx, num_segments=num_nodes
            )
            aggregated = sum_messages / (counts[:, None] + EPSILON)
        elif self.aggregation == "max":
            # segment_max with default of -inf for empty segments
            aggregated = jax.ops.segment_max(
                messages,
                dest_idx,
                num_segments=num_nodes,
                indices_are_sorted=False,
            )
            # Replace -inf with 0 for nodes with no incoming edges
            aggregated = jnp.where(
                jnp.isinf(aggregated), jnp.zeros_like(aggregated), aggregated
            )
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Update node features: MLP(concat(node_features, aggregated))
        update_input = jnp.concatenate([node_features, aggregated], axis=-1)
        x = self.update_linear1(update_input)
        x = nnx.relu(x)
        return self.update_linear2(x)
