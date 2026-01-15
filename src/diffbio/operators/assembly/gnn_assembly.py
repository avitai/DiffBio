"""Graph Neural Network-based assembly navigator.

This module provides a differentiable approach to assembly graph
traversal using message passing neural networks.

Key technique: Uses graph attention for message passing between nodes,
then scores edges for soft path selection, enabling gradient flow
through the assembly process.

Applications: Differentiable genome assembly, assembly polishing,
scaffolding optimization.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree


@dataclass
class GNNAssemblyNavigatorConfig(OperatorConfig):
    """Configuration for GNNAssemblyNavigator.

    Attributes:
        node_features: Dimension of input node features.
        hidden_dim: Hidden dimension for GNN layers.
        num_layers: Number of GNN layers.
        num_heads: Number of attention heads.
        edge_features: Dimension of edge features.
        dropout_rate: Dropout rate for regularization.
        temperature: Temperature for softmax operations.
        stochastic: Whether the operator uses randomness.
        stream_name: RNG stream name.
    """

    node_features: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    edge_features: int = 8
    dropout_rate: float = 0.1
    temperature: float = 1.0
    stochastic: bool = False
    stream_name: str | None = None


class GraphAttentionLayer(nnx.Module):
    """Graph attention layer for message passing.

    Uses multi-head attention to aggregate neighbor messages.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        edge_features: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the graph attention layer.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            num_heads: Number of attention heads.
            edge_features: Edge feature dimension.
            dropout_rate: Dropout rate.
            rngs: Random number generators.
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.scale = self.head_dim**-0.5

        # Node feature projections
        self.query_proj = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs,
        )
        self.key_proj = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs,
        )
        self.value_proj = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs,
        )

        # Edge feature projection
        self.edge_proj = nnx.Linear(
            in_features=edge_features,
            out_features=num_heads,
            rngs=rngs,
        )

        # Output projection
        self.output_proj = nnx.Linear(
            in_features=out_features,
            out_features=out_features,
            rngs=rngs,
        )

        # Dropout
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        node_features: Float[Array, "n_nodes in_features"],
        edge_index: Int[Array, "2 n_edges"],
        edge_features: Float[Array, "n_edges edge_features"],
        *,
        deterministic: bool = True,
    ) -> Float[Array, "n_nodes out_features"]:
        """Apply graph attention.

        Args:
            node_features: Node feature matrix.
            edge_index: Edge indices (source, target).
            edge_features: Edge feature matrix.
            deterministic: If True, disable dropout.

        Returns:
            Updated node features.
        """
        n_nodes = node_features.shape[0]
        n_edges = edge_index.shape[1]

        sources = edge_index[0]  # (n_edges,)
        targets = edge_index[1]  # (n_edges,)

        # Project all nodes
        Q = self.query_proj(node_features)  # (n_nodes, out_features)
        K = self.key_proj(node_features)
        V = self.value_proj(node_features)

        # Reshape for multi-head attention
        Q = Q.reshape(n_nodes, self.num_heads, self.head_dim)
        K = K.reshape(n_nodes, self.num_heads, self.head_dim)
        V = V.reshape(n_nodes, self.num_heads, self.head_dim)

        # Get source and target features for each edge
        Q_targets = Q[targets]  # (n_edges, num_heads, head_dim)
        K_sources = K[sources]  # (n_edges, num_heads, head_dim)
        V_sources = V[sources]  # (n_edges, num_heads, head_dim)

        # Compute attention scores
        attn_scores = jnp.sum(Q_targets * K_sources, axis=-1) * self.scale  # (n_edges, num_heads)

        # Add edge feature bias
        edge_bias = self.edge_proj(edge_features)  # (n_edges, num_heads)
        attn_scores = attn_scores + edge_bias

        # Normalize attention per target node using segment_max/segment_sum
        # This is equivalent to softmax over incoming edges per node
        max_scores = jax.ops.segment_max(
            attn_scores, targets, num_segments=n_nodes
        )  # (n_nodes, num_heads)
        attn_scores = attn_scores - max_scores[targets]  # Stability
        attn_exp = jnp.exp(attn_scores)

        # Sum of exp scores per target node
        attn_sum = (
            jax.ops.segment_sum(attn_exp, targets, num_segments=n_nodes) + 1e-10
        )  # (n_nodes, num_heads)

        # Normalize
        attn_probs = attn_exp / attn_sum[targets]  # (n_edges, num_heads)

        # Apply dropout
        if self.dropout is not None and not deterministic:
            attn_probs = self.dropout(attn_probs)

        # Weighted sum of values
        weighted_V = attn_probs[:, :, None] * V_sources  # (n_edges, num_heads, head_dim)

        # Aggregate to target nodes
        aggregated = jax.ops.segment_sum(
            weighted_V.reshape(n_edges, -1), targets, num_segments=n_nodes
        )  # (n_nodes, out_features)

        # Output projection
        output = self.output_proj(aggregated)

        return output


class GNNLayer(nnx.Module):
    """Full GNN layer with attention, normalization, and feedforward."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_features: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the GNN layer.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            edge_features: Edge feature dimension.
            dropout_rate: Dropout rate.
            rngs: Random number generators.
        """
        super().__init__()

        self.attention = GraphAttentionLayer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_heads=num_heads,
            edge_features=edge_features,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

        self.layer_norm1 = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)

        # Feedforward
        self.ff_linear1 = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim * 4,
            rngs=rngs,
        )
        self.ff_linear2 = nnx.Linear(
            in_features=hidden_dim * 4,
            out_features=hidden_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        node_features: Float[Array, "n_nodes hidden_dim"],
        edge_index: Int[Array, "2 n_edges"],
        edge_features: Float[Array, "n_edges edge_features"],
        *,
        deterministic: bool = True,
    ) -> Float[Array, "n_nodes hidden_dim"]:
        """Apply GNN layer.

        Args:
            node_features: Node features.
            edge_index: Edge indices.
            edge_features: Edge features.
            deterministic: If True, disable dropout.

        Returns:
            Updated node features.
        """
        # Attention with residual
        attended = self.attention(
            node_features, edge_index, edge_features, deterministic=deterministic
        )
        x = self.layer_norm1(node_features + attended)

        # Feedforward with residual
        ff_out = self.ff_linear2(nnx.gelu(self.ff_linear1(x)))
        x = self.layer_norm2(x + ff_out)

        return x


class GNNAssemblyNavigator(OperatorModule):
    """Graph Neural Network for assembly graph traversal.

    This operator uses message passing with graph attention to update
    node embeddings and predict edge traversal probabilities for
    differentiable assembly.

    Algorithm:
    1. Project input node features to hidden dimension
    2. Apply multiple GNN layers with graph attention
    3. Compute edge scores from source/target node embeddings
    4. Apply sigmoid for traversal probabilities
    5. Compute path confidence from edge scores

    Args:
        config: GNNAssemblyNavigatorConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = GNNAssemblyNavigatorConfig(hidden_dim=128)
        >>> navigator = GNNAssemblyNavigator(config, rngs=nnx.Rngs(42))
        >>> data = {"node_features": nodes, "edge_index": edges, "edge_features": edge_attr}
        >>> result, state, meta = navigator.apply(data, {}, None)
    """

    def __init__(
        self,
        config: GNNAssemblyNavigatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the GNN assembly navigator.

        Args:
            config: Navigator configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.hidden_dim = config.hidden_dim
        self.temperature = config.temperature

        # Input projection
        self.input_projection = nnx.Linear(
            in_features=config.node_features,
            out_features=config.hidden_dim,
            rngs=rngs,
        )

        # GNN layers
        self.gnn_layers = nnx.List(
            [
                GNNLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    edge_features=config.edge_features,
                    dropout_rate=config.dropout_rate,
                    rngs=rngs,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Edge scoring MLP
        self.edge_mlp = nnx.Linear(
            in_features=config.hidden_dim * 2,
            out_features=1,
            rngs=rngs,
        )

    def compute_edge_scores(
        self,
        node_embeddings: Float[Array, "n_nodes hidden_dim"],
        edge_index: Int[Array, "2 n_edges"],
    ) -> Float[Array, "n_edges"]:
        """Compute edge scores from node embeddings.

        Args:
            node_embeddings: Node embedding matrix.
            edge_index: Edge indices (source, target).

        Returns:
            Score for each edge.
        """
        sources = edge_index[0]
        targets = edge_index[1]

        # Get source and target embeddings
        source_emb = node_embeddings[sources]  # (n_edges, hidden_dim)
        target_emb = node_embeddings[targets]  # (n_edges, hidden_dim)

        # Concatenate and score
        edge_repr = jnp.concatenate([source_emb, target_emb], axis=-1)
        scores = self.edge_mlp(edge_repr).squeeze(-1)  # (n_edges,)

        return scores

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply GNN assembly navigation.

        Args:
            data: Dictionary containing:
                - "node_features": Node features (n_nodes, node_features)
                - "edge_index": Edge indices (2, n_edges)
                - "edge_features": Edge features (n_edges, edge_features)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:
                    - "node_features": Original node features
                    - "edge_index": Original edge indices
                    - "edge_features": Original edge features
                    - "node_embeddings": Updated node embeddings
                    - "edge_scores": Scores for each edge
                    - "traversal_probs": Sigmoid probabilities for traversal
                    - "path_confidence": Confidence score for paths
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        node_features = data["node_features"]
        edge_index = data["edge_index"]
        edge_features = data["edge_features"]

        # Project to hidden dimension
        node_emb = self.input_projection(node_features)

        # Apply GNN layers
        deterministic = not self.config.stochastic
        for layer in self.gnn_layers:
            node_emb = layer(node_emb, edge_index, edge_features, deterministic=deterministic)

        # Compute edge scores
        edge_scores = self.compute_edge_scores(node_emb, edge_index)

        # Traversal probabilities via sigmoid
        traversal_probs = jax.nn.sigmoid(edge_scores / self.temperature)

        # Path confidence: mean probability weighted by score magnitude
        path_confidence = jnp.mean(traversal_probs * jnp.abs(edge_scores))

        # Build output
        transformed_data = {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "node_embeddings": node_emb,
            "edge_scores": edge_scores,
            "traversal_probs": traversal_probs,
            "path_confidence": path_confidence,
        }

        return transformed_data, state, metadata
