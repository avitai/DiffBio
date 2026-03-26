"""Message passing neural network layers for molecular graphs.

This module implements directed message passing neural network (D-MPNN)
layers following the ChemProp architecture for molecular property prediction.
"""

import logging

import jax.numpy as jnp
from flax import nnx

logger = logging.getLogger(__name__)


class MessagePassingLayer(nnx.Module):
    """Directed message passing layer for molecular graphs.

    Implements the D-MPNN message passing scheme where messages are passed
    along directed edges. Each node aggregates messages from its neighbors
    and updates its representation.

    Attributes:
        hidden_dim: Dimension of hidden node representations.
        in_features: Number of input node features.
        num_edge_features: Number of edge features (default 4 for bond types).
    """

    def __init__(
        self,
        hidden_dim: int,
        in_features: int = 4,
        num_edge_features: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize message passing layer.

        Args:
            hidden_dim: Dimension of hidden representations.
            in_features: Number of input node features (default 4 for tests).
            num_edge_features: Number of edge/bond features.
            rngs: Flax NNX random number generators.
        """
        self.hidden_dim = hidden_dim
        self.in_features = in_features
        self.num_edge_features = num_edge_features

        # Node encoder - eagerly initialized with specified in_features
        self.node_encoder = nnx.Linear(
            in_features=in_features,
            out_features=hidden_dim,
            rngs=rngs,
        )

        # Edge encoder
        self.edge_encoder = nnx.Linear(
            in_features=num_edge_features,
            out_features=hidden_dim,
            rngs=rngs,
        )

        # Message transformation
        self.message_layer = nnx.Linear(
            in_features=hidden_dim * 3,  # src_node + edge + dst_node
            out_features=hidden_dim,
            rngs=rngs,
        )

        # Update function (GRU-like update)
        self.update_layer = nnx.Linear(
            in_features=hidden_dim * 2,  # current + aggregated
            out_features=hidden_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        node_features: jnp.ndarray,
        adjacency: jnp.ndarray,
        edge_features: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Perform one step of message passing.

        Args:
            node_features: Node features of shape (num_nodes, in_features).
            adjacency: Adjacency matrix of shape (num_nodes, num_nodes).
            edge_features: Optional edge features of shape
                (num_nodes, num_nodes, num_edge_features).

        Returns:
            Updated node features of shape (num_nodes, hidden_dim).
        """
        num_nodes = node_features.shape[0]

        # Encode node features to hidden dimension
        node_hidden = nnx.relu(self.node_encoder(node_features))

        # Handle edge features
        if edge_features is not None:
            edge_hidden = nnx.relu(self.edge_encoder(edge_features))
        else:
            # Use zeros if no edge features provided
            edge_hidden = jnp.zeros((num_nodes, num_nodes, self.hidden_dim), dtype=jnp.float32)

        # Compute messages for all pairs
        # For each edge (i, j), message = f(node_i, edge_ij, node_j)
        # Expand dimensions for broadcasting
        src_nodes = node_hidden[:, None, :]  # (N, 1, H)
        dst_nodes = node_hidden[None, :, :]  # (1, N, H)

        # Broadcast to (N, N, H)
        src_expanded = jnp.broadcast_to(src_nodes, (num_nodes, num_nodes, self.hidden_dim))
        dst_expanded = jnp.broadcast_to(dst_nodes, (num_nodes, num_nodes, self.hidden_dim))

        # Concatenate [src, edge, dst]
        message_input = jnp.concatenate([src_expanded, edge_hidden, dst_expanded], axis=-1)

        # Compute messages
        messages = nnx.relu(self.message_layer(message_input))

        # Mask messages by adjacency (only neighbors contribute)
        masked_messages = messages * adjacency[:, :, None]

        # Aggregate messages (sum over neighbors)
        aggregated = jnp.sum(masked_messages, axis=1)  # (N, H)

        # Update node representations
        update_input = jnp.concatenate([node_hidden, aggregated], axis=-1)
        updated = nnx.relu(self.update_layer(update_input))

        return updated


class StackedMessagePassing(nnx.Module):
    """Stack of message passing layers.

    Applies multiple rounds of message passing to capture higher-order
    neighborhood information.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        in_features: int = 4,
        num_edge_features: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize stacked message passing.

        Args:
            hidden_dim: Hidden dimension for all layers.
            num_layers: Number of message passing iterations.
            in_features: Number of input node features (default 4 for tests).
            num_edge_features: Number of edge features.
            rngs: Flax NNX random number generators.
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.in_features = in_features

        # Build layers with proper input dimensions:
        # - First layer: in_features -> hidden_dim
        # - Subsequent layers: hidden_dim -> hidden_dim
        layers = []
        for i in range(num_layers):
            layer_in_features = in_features if i == 0 else hidden_dim
            layers.append(
                MessagePassingLayer(
                    hidden_dim=hidden_dim,
                    in_features=layer_in_features,
                    num_edge_features=num_edge_features,
                    rngs=rngs,
                )
            )
        self.layers = nnx.List(layers)

    def __call__(
        self,
        node_features: jnp.ndarray,
        adjacency: jnp.ndarray,
        edge_features: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Apply multiple rounds of message passing.

        Args:
            node_features: Initial node features.
            adjacency: Adjacency matrix.
            edge_features: Optional edge features.

        Returns:
            Final node representations.
        """
        h = node_features

        for layer in self.layers:
            h = layer(h, adjacency, edge_features)

        return h
