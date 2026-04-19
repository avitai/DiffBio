"""Shared graph attention components for GNN-based operators.

Ownership note: DiffBio retains these sparse GAT/GATv2 layers because sibling
repos do not currently expose a GATv2-compatible graph-attention block with the
same edge-feature and segment-softmax contract. Generic model pieces should
still come from Artifex where an exact reusable layer exists.

This module provides reusable graph attention building blocks that are needed
by multiple downstream operators (assembly, cell-cell communication, GRN
inference, spatial domain identification, etc.).

Components:

- **GraphAttentionLayer**: Multi-head GAT-style attention (dot-product Q*K).
- **GraphAttentionBlock**: GAT attention + LayerNorm + residual + FFN.
- **GATv2Layer**: GATv2-style attention that applies LeakyReLU *before* the
  attention vector dot product, making it strictly more expressive than GAT.
- **GATv2Block**: GATv2 attention + LayerNorm + residual + FFN.

These are architecturally distinct from ``GraphMessagePassing`` in
``neural_components.py``, which uses simpler sum/mean/max aggregation
without attention.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

__all__ = [
    "GraphAttentionLayer",
    "GraphAttentionBlock",
    "GATv2Layer",
    "GATv2Block",
]


class _BidirectionalProjection(nnx.Module):
    """Pair of linear projections used by attention mechanisms."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        negative_slope: float | None = None,
    ) -> None:
        """Initialize the paired projection module."""
        super().__init__()
        self.left = nnx.Linear(in_features=in_features, out_features=out_features, rngs=rngs)
        self.right = nnx.Linear(in_features=in_features, out_features=out_features, rngs=rngs)
        self.negative_slope = negative_slope

    def __call__(
        self,
        node_features: Float[Array, "n_nodes in_features"],
    ) -> tuple[
        Float[Array, "n_nodes out_features"],
        Float[Array, "n_nodes out_features"],
    ]:
        """Project the same node tensor through both linear paths."""
        return self.left(node_features), self.right(node_features)


class GraphAttentionLayer(nnx.Module):
    """Multi-head graph attention layer for message passing.

    Computes attention-weighted message aggregation over graph edges.
    Each attention head independently computes query/key/value projections,
    adds an edge-feature bias to attention scores, normalizes via
    segment-softmax, and aggregates weighted values per target node.

    Args:
        in_features: Input node feature dimension.
        out_features: Output feature dimension (must be divisible by num_heads).
        num_heads: Number of parallel attention heads.
        edge_features: Edge feature dimension.
        dropout_rate: Dropout rate applied to attention weights.
        rngs: Flax NNX random number generators.
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
    ) -> None:
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
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if out_features % num_heads != 0:
            raise ValueError("out_features must be divisible by num_heads")
        if edge_features <= 0:
            raise ValueError("edge_features must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")

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

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs) if dropout_rate > 0 else None

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return self.edge_proj.out_features

    @property
    def head_dim(self) -> int:
        """Per-head hidden dimension."""
        return self.query_proj.out_features // self.num_heads

    @property
    def scale(self) -> float:
        """Dot-product attention scaling factor."""
        return self.head_dim**-0.5

    def __call__(
        self,
        node_features: Float[Array, "n_nodes in_features"],
        edge_index: Int[Array, "2 n_edges"],
        edge_features: Float[Array, "n_edges edge_features"],
        *,
        deterministic: bool = True,
    ) -> Float[Array, "n_nodes out_features"]:
        """Run one graph-attention update step.

        Args:
            node_features: Node feature matrix of shape ``(n_nodes, in_features)``.
            edge_index: Edge indices ``(source, target)`` of shape ``(2, n_edges)``.
            edge_features: Edge feature matrix of shape ``(n_edges, edge_features)``.
            deterministic: Whether to disable stochastic dropout.

        Returns:
            Updated node features of shape ``(n_nodes, out_features)``.
        """
        n_nodes = node_features.shape[0]
        n_edges = edge_index.shape[1]
        out_features = self.num_heads * self.head_dim

        # Handle empty graph: no edges means no messages to aggregate
        if n_edges == 0:
            return self.output_proj(jnp.zeros((n_nodes, out_features)))

        sources = edge_index[0]  # (n_edges,)
        targets = edge_index[1]  # (n_edges,)

        # Project all nodes
        queries = self.query_proj(node_features)  # (n_nodes, out_features)
        keys = self.key_proj(node_features)
        values = self.value_proj(node_features)

        # Reshape for multi-head attention
        queries = queries.reshape(n_nodes, self.num_heads, self.head_dim)
        keys = keys.reshape(n_nodes, self.num_heads, self.head_dim)
        values = values.reshape(n_nodes, self.num_heads, self.head_dim)

        # Get source and target features for each edge
        query_targets = queries[targets]  # (n_edges, num_heads, head_dim)
        key_sources = keys[sources]  # (n_edges, num_heads, head_dim)
        value_sources = values[sources]  # (n_edges, num_heads, head_dim)

        # Compute attention scores
        attn_scores = (
            jnp.sum(query_targets * key_sources, axis=-1) * self.scale
        )  # (n_edges, num_heads)

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
        weighted_values = attn_probs[:, :, None] * value_sources  # (n_edges, num_heads, head_dim)

        # Aggregate to target nodes
        aggregated = jax.ops.segment_sum(
            weighted_values.reshape(n_edges, -1), targets, num_segments=n_nodes
        )  # (n_nodes, out_features)

        # Output projection
        return self.output_proj(aggregated)


class GraphAttentionBlock(nnx.Module):
    """Full GNN block: graph attention + LayerNorm + residual + feedforward.

    Combines a :class:`GraphAttentionLayer` with pre-norm residual connections
    and a two-layer feedforward network (4x expansion), following the
    standard Transformer block pattern adapted for graphs.

    Architecture::

        x -> GraphAttentionLayer -> + -> LayerNorm -> FFN -> + -> LayerNorm -> out
        |___________________________|              |_________|

    Args:
        hidden_dim: Hidden dimension (both input and output).
        num_heads: Number of attention heads.
        edge_features: Edge feature dimension.
        dropout_rate: Dropout rate.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_features: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the GNN block.

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

        # Feedforward with 4x expansion
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
        """Apply the GNN block.

        Args:
            node_features: Node features of shape ``(n_nodes, hidden_dim)``.
            edge_index: Edge indices of shape ``(2, n_edges)``.
            edge_features: Edge features of shape ``(n_edges, edge_features)``.
            deterministic: If True, disable dropout.

        Returns:
            Updated node features of shape ``(n_nodes, hidden_dim)``.
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


class GATv2Layer(nnx.Module):
    """GATv2 multi-head graph attention layer.

    Unlike the original GAT (``GraphAttentionLayer``), GATv2 applies LeakyReLU
    *before* computing the attention scalar, which makes the attention function
    strictly more expressive (it can represent any monotonic scoring function
    over concatenated source/target features).

    GATv2 attention::

        e_{ij} = a^T * LeakyReLU(W_l * h_i + W_r * h_j + edge_bias)

    This is the key difference from GAT, where the nonlinearity is applied
    *after* the attention dot product.

    Reference: Brody, Alon, Yahav. "How Attentive are Graph Attention
    Networks?" (ICLR 2022).

    Args:
        in_features: Input node feature dimension.
        out_features: Output feature dimension (must be divisible by num_heads).
        num_heads: Number of parallel attention heads.
        edge_features: Edge feature dimension.
        dropout_rate: Dropout rate applied to attention weights.
        negative_slope: Negative slope for LeakyReLU (default 0.2).
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        edge_features: int,
        dropout_rate: float,
        negative_slope: float = 0.2,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the GATv2 layer.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            num_heads: Number of attention heads.
            edge_features: Edge feature dimension.
            dropout_rate: Dropout rate.
            negative_slope: Negative slope for LeakyReLU.
            rngs: Random number generators.
        """
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if out_features % num_heads != 0:
            raise ValueError("out_features must be divisible by num_heads")
        if edge_features <= 0:
            raise ValueError("edge_features must be positive")
        if negative_slope < 0.0:
            raise ValueError("negative_slope must be non-negative")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")

        # GATv2 uses separate left/right projections (not Q/K/V)
        self.attn_projections = _BidirectionalProjection(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs,
            negative_slope=negative_slope,
        )

        # Edge feature projection to per-head bias
        self.edge_proj = nnx.Linear(
            in_features=edge_features,
            out_features=num_heads,
            rngs=rngs,
        )

        # Per-head attention vector a^T (applied after LeakyReLU)
        self.attn_vector = nnx.Param(
            jax.random.normal(rngs.params(), (num_heads, self.head_dim)) * 0.01
        )

        # Value projection and output projection
        self.value_proj = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs,
        )
        self.output_proj = nnx.Linear(
            in_features=out_features,
            out_features=out_features,
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs) if dropout_rate > 0 else None

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return self.edge_proj.out_features

    @property
    def head_dim(self) -> int:
        """Per-head hidden dimension."""
        return self.attn_projections.left.out_features // self.num_heads

    @property
    def negative_slope(self) -> float:
        """Negative slope used by the LeakyReLU attention scorer."""
        if self.attn_projections.negative_slope is None:
            raise ValueError("negative_slope is not configured")
        return self.attn_projections.negative_slope

    def __call__(
        self,
        node_features: Float[Array, "n_nodes in_features"],
        edge_index: Int[Array, "2 n_edges"],
        edge_features: Float[Array, "n_edges edge_features"],
        *,
        deterministic: bool = True,
    ) -> Float[Array, "n_nodes out_features"]:
        """Run one GATv2 attention update step.

        Args:
            node_features: Node feature matrix ``(n_nodes, in_features)``.
            edge_index: Edge indices ``(source, target)`` of shape ``(2, n_edges)``.
            edge_features: Edge feature matrix ``(n_edges, edge_features)``.
            deterministic: Whether to disable stochastic dropout.

        Returns:
            Updated node features of shape ``(n_nodes, out_features)``.
        """
        n_nodes = node_features.shape[0]
        n_edges = edge_index.shape[1]
        out_features = self.num_heads * self.head_dim

        # Handle empty graph
        if n_edges == 0:
            return self.output_proj(jnp.zeros((n_nodes, out_features)))

        sources = edge_index[0]
        targets = edge_index[1]

        # Left/right projections for all nodes
        left, right = self.attn_projections(node_features)

        # Reshape to per-head: (n_nodes, num_heads, head_dim)
        left = left.reshape(n_nodes, self.num_heads, self.head_dim)
        right = right.reshape(n_nodes, self.num_heads, self.head_dim)

        # Gather per-edge: left for targets, right for sources
        left_targets = left[targets]  # (n_edges, num_heads, head_dim)
        right_sources = right[sources]

        # GATv2 key: LeakyReLU BEFORE the attention dot product
        combined = left_targets + right_sources  # (n_edges, num_heads, head_dim)
        activated = jax.nn.leaky_relu(combined, negative_slope=self.negative_slope)

        # Attention score: a^T * activated  =>  (n_edges, num_heads)
        attn_scores = jnp.sum(activated * self.attn_vector[...][None, :, :], axis=-1)

        # Add edge feature bias
        edge_bias = self.edge_proj(edge_features)  # (n_edges, num_heads)
        attn_scores = attn_scores + edge_bias

        # Segment-softmax normalization over incoming edges per target node
        max_scores = jax.ops.segment_max(attn_scores, targets, num_segments=n_nodes)
        attn_scores = attn_scores - max_scores[targets]
        attn_exp = jnp.exp(attn_scores)

        attn_sum = jax.ops.segment_sum(attn_exp, targets, num_segments=n_nodes) + 1e-10
        attn_probs = attn_exp / attn_sum[targets]  # (n_edges, num_heads)

        # Apply dropout
        if self.dropout is not None and not deterministic:
            attn_probs = self.dropout(attn_probs)

        # Value aggregation
        values = self.value_proj(node_features).reshape(n_nodes, self.num_heads, self.head_dim)
        value_sources = values[sources]  # (n_edges, num_heads, head_dim)
        weighted_values = attn_probs[:, :, None] * value_sources

        aggregated = jax.ops.segment_sum(
            weighted_values.reshape(n_edges, -1), targets, num_segments=n_nodes
        )

        return self.output_proj(aggregated)


class GATv2Block(nnx.Module):
    """Full GNN block using GATv2 attention + LayerNorm + residual + FFN.

    Architecture::

        x -> GATv2Layer -> + -> LayerNorm -> FFN -> + -> LayerNorm -> out
        |__________________|              |_________|

    Args:
        hidden_dim: Hidden dimension (both input and output).
        num_heads: Number of attention heads.
        edge_features: Edge feature dimension.
        dropout_rate: Dropout rate.
        negative_slope: Negative slope for LeakyReLU in GATv2 attention.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_features: int,
        dropout_rate: float,
        negative_slope: float = 0.2,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the GATv2 block.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            edge_features: Edge feature dimension.
            dropout_rate: Dropout rate.
            negative_slope: Negative slope for LeakyReLU.
            rngs: Random number generators.
        """
        super().__init__()

        self.attention = GATv2Layer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_heads=num_heads,
            edge_features=edge_features,
            dropout_rate=dropout_rate,
            negative_slope=negative_slope,
            rngs=rngs,
        )

        self.layer_norm1 = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)

        # Feedforward with 4x expansion
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
        """Apply the GATv2 block.

        Args:
            node_features: Node features ``(n_nodes, hidden_dim)``.
            edge_index: Edge indices ``(2, n_edges)``.
            edge_features: Edge features ``(n_edges, edge_features)``.
            deterministic: If True, disable dropout.

        Returns:
            Updated node features ``(n_nodes, hidden_dim)``.
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
