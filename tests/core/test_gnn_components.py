"""Tests for shared GNN components.

Tests for GraphAttentionLayer, GraphAttentionBlock, GATv2Layer, and GATv2Block.
These tests define the expected behavior for the graph attention components
extracted from gnn_assembly.py into a reusable core module.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.core.gnn_components import (
    GATv2Block,
    GATv2Layer,
    GraphAttentionBlock,
    GraphAttentionLayer,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_graph():
    """Provide a small graph with 5 nodes and 8 edges."""
    key = jax.random.key(0)
    n_nodes = 5
    n_edges = 8
    node_feat_dim = 16
    edge_feat_dim = 4

    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    node_features = jax.random.normal(k1, (n_nodes, node_feat_dim))
    sources = jax.random.randint(k2, (n_edges,), 0, n_nodes)
    targets = jax.random.randint(k3, (n_edges,), 0, n_nodes)
    edge_index = jnp.stack([sources, targets], axis=0)
    edge_features = jax.random.normal(k4, (n_edges, edge_feat_dim))

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "node_feat_dim": node_feat_dim,
        "edge_feat_dim": edge_feat_dim,
    }


# =============================================================================
# TestGraphAttentionLayer
# =============================================================================


class TestGraphAttentionLayer:
    """Tests for GraphAttentionLayer initialization, forward pass, and properties."""

    def test_initialization(self, rngs):
        """Test layer initializes with correct attributes."""
        layer = GraphAttentionLayer(
            in_features=16,
            out_features=32,
            num_heads=4,
            edge_features=4,
            dropout_rate=0.1,
            rngs=rngs,
        )
        assert layer.num_heads == 4
        assert layer.head_dim == 8  # 32 // 4

    def test_initialization_no_dropout(self, rngs):
        """Test layer initializes without dropout when rate is 0."""
        layer = GraphAttentionLayer(
            in_features=16,
            out_features=32,
            num_heads=4,
            edge_features=4,
            dropout_rate=0.0,
            rngs=rngs,
        )
        assert layer.dropout is None

    def test_forward_pass_output_shape(self, rngs, small_graph):
        """Test forward pass produces correct output shape."""
        out_features = 32
        layer = GraphAttentionLayer(
            in_features=small_graph["node_feat_dim"],
            out_features=out_features,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        output = layer(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert output.shape == (small_graph["n_nodes"], out_features)

    def test_forward_pass_output_finite(self, rngs, small_graph):
        """Test that output values are finite."""
        layer = GraphAttentionLayer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        output = layer(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert jnp.isfinite(output).all()

    def test_multi_head_attention_produces_different_results(self, rngs, small_graph):
        """Test that different head counts produce different outputs."""
        layer_2h = GraphAttentionLayer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=2,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )
        layer_4h = GraphAttentionLayer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )
        out_2h = layer_2h(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        out_4h = layer_4h(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        # With different head structures (different projections), outputs should differ
        assert not jnp.allclose(out_2h, out_4h, atol=1e-3)

    def test_edge_features_influence_output(self, rngs, small_graph):
        """Test that edge features influence output via non-zero gradient."""
        layer = GraphAttentionLayer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        node_features = small_graph["node_features"]
        edge_index = small_graph["edge_index"]

        # Verify that the output depends on edge features by checking
        # that the Jacobian w.r.t. edge features is non-zero.
        def loss_fn(ef):
            return layer(node_features, edge_index, ef).sum()

        grad = jax.grad(loss_fn)(small_graph["edge_features"])
        assert jnp.any(grad != 0.0), "Edge features should influence output"

    def test_gradient_flow_through_node_features(self, rngs, small_graph):
        """Test that gradients flow through node features."""
        layer = GraphAttentionLayer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        edge_index = small_graph["edge_index"]
        edge_features = small_graph["edge_features"]

        def loss_fn(node_features):
            return layer(node_features, edge_index, edge_features).sum()

        grads = jax.grad(loss_fn)(small_graph["node_features"])
        assert grads.shape == small_graph["node_features"].shape
        assert jnp.isfinite(grads).all()

    def test_gradient_flow_through_edge_features(self, rngs, small_graph):
        """Test that gradients flow through edge features."""
        layer = GraphAttentionLayer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        node_features = small_graph["node_features"]
        edge_index = small_graph["edge_index"]

        def loss_fn(edge_features):
            return layer(node_features, edge_index, edge_features).sum()

        grads = jax.grad(loss_fn)(small_graph["edge_features"])
        assert grads.shape == small_graph["edge_features"].shape
        assert jnp.isfinite(grads).all()

    def test_gradient_flow_through_parameters(self, rngs, small_graph):
        """Test that gradients flow through layer parameters."""
        layer = GraphAttentionLayer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )

        @nnx.value_and_grad
        def loss_fn(model):
            return model(
                small_graph["node_features"],
                small_graph["edge_index"],
                small_graph["edge_features"],
            ).sum()

        _, grads = loss_fn(layer)
        assert hasattr(grads, "query_proj")
        assert hasattr(grads, "key_proj")
        assert hasattr(grads, "value_proj")
        assert hasattr(grads, "output_proj")
        assert hasattr(grads, "edge_proj")

    def test_jit_compatible(self, rngs, small_graph):
        """Test that the layer works under JIT compilation."""
        layer = GraphAttentionLayer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )

        @jax.jit
        def forward(node_features, edge_index, edge_features):
            return layer(node_features, edge_index, edge_features)

        output = forward(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert output.shape == (small_graph["n_nodes"], 32)
        assert jnp.isfinite(output).all()

    def test_deterministic_flag_consistency(self, rngs, small_graph):
        """Test deterministic mode produces consistent results."""
        layer = GraphAttentionLayer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.1,
            rngs=rngs,
        )
        out1 = layer(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
            deterministic=True,
        )
        out2 = layer(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
            deterministic=True,
        )
        assert jnp.allclose(out1, out2)


# =============================================================================
# TestGraphAttentionBlock
# =============================================================================


class TestGraphAttentionBlock:
    """Tests for GraphAttentionBlock (attention + LayerNorm + residual + FFN)."""

    def test_initialization(self, rngs):
        """Test block initializes correctly."""
        block = GraphAttentionBlock(
            hidden_dim=32,
            num_heads=4,
            edge_features=4,
            dropout_rate=0.1,
            rngs=rngs,
        )
        assert block.attention is not None
        assert block.layer_norm1 is not None
        assert block.layer_norm2 is not None
        assert block.ff_linear1 is not None
        assert block.ff_linear2 is not None

    def test_forward_pass_output_shape(self, rngs, small_graph):
        """Test forward pass preserves node feature dimension."""
        hidden_dim = small_graph["node_feat_dim"]
        block = GraphAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        output = block(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        # Output has same shape as input (residual connection preserves dims)
        assert output.shape == small_graph["node_features"].shape

    def test_forward_pass_output_finite(self, rngs, small_graph):
        """Test that output values are finite."""
        hidden_dim = small_graph["node_feat_dim"]
        block = GraphAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        output = block(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert jnp.isfinite(output).all()

    def test_residual_connection_effect(self, rngs, small_graph):
        """Test that residual connection means output is not zero for zero attention."""
        hidden_dim = small_graph["node_feat_dim"]
        block = GraphAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        output = block(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        # Output should not be all zeros thanks to residual + LayerNorm
        assert not jnp.allclose(output, 0.0)

    def test_feedforward_expansion(self, rngs):
        """Test that feedforward layer uses 4x expansion."""
        hidden_dim = 32
        block = GraphAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=4,
            dropout_rate=0.0,
            rngs=rngs,
        )
        # ff_linear1 should project to 4*hidden_dim
        assert block.ff_linear1.out_features == hidden_dim * 4
        # ff_linear2 should project back to hidden_dim
        assert block.ff_linear2.out_features == hidden_dim

    def test_gradient_flow_through_block(self, rngs, small_graph):
        """Test that gradients flow through the entire block."""
        hidden_dim = small_graph["node_feat_dim"]
        block = GraphAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        edge_index = small_graph["edge_index"]
        edge_features = small_graph["edge_features"]

        def loss_fn(node_features):
            return block(node_features, edge_index, edge_features).sum()

        grads = jax.grad(loss_fn)(small_graph["node_features"])
        assert grads.shape == small_graph["node_features"].shape
        assert jnp.isfinite(grads).all()

    def test_gradient_flow_through_parameters(self, rngs, small_graph):
        """Test that gradients flow to all block parameters."""
        hidden_dim = small_graph["node_feat_dim"]
        block = GraphAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )

        @nnx.value_and_grad
        def loss_fn(model):
            return model(
                small_graph["node_features"],
                small_graph["edge_index"],
                small_graph["edge_features"],
            ).sum()

        _, grads = loss_fn(block)
        assert hasattr(grads, "attention")
        assert hasattr(grads, "layer_norm1")
        assert hasattr(grads, "layer_norm2")
        assert hasattr(grads, "ff_linear1")
        assert hasattr(grads, "ff_linear2")

    def test_jit_compatible(self, rngs, small_graph):
        """Test that the block works under JIT compilation."""
        hidden_dim = small_graph["node_feat_dim"]
        block = GraphAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )

        @jax.jit
        def forward(node_features, edge_index, edge_features):
            return block(node_features, edge_index, edge_features)

        output = forward(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert output.shape == small_graph["node_features"].shape
        assert jnp.isfinite(output).all()

    def test_stacking_multiple_blocks(self, rngs, small_graph):
        """Test that multiple blocks can be stacked sequentially."""
        hidden_dim = small_graph["node_feat_dim"]
        blocks = [
            GraphAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=4,
                edge_features=small_graph["edge_feat_dim"],
                dropout_rate=0.0,
                rngs=rngs,
            )
            for _ in range(3)
        ]

        x = small_graph["node_features"]
        for block in blocks:
            x = block(x, small_graph["edge_index"], small_graph["edge_features"])

        assert x.shape == small_graph["node_features"].shape
        assert jnp.isfinite(x).all()


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases: empty edges, single node, large graphs."""

    def test_single_node_no_edges(self, rngs):
        """Test with a single node and no edges."""
        layer = GraphAttentionLayer(
            in_features=8,
            out_features=16,
            num_heads=2,
            edge_features=4,
            dropout_rate=0.0,
            rngs=rngs,
        )
        node_features = jax.random.normal(jax.random.key(0), (1, 8))
        edge_index = jnp.zeros((2, 0), dtype=jnp.int32)
        edge_features = jnp.zeros((0, 4))

        output = layer(node_features, edge_index, edge_features)
        assert output.shape == (1, 16)
        assert jnp.isfinite(output).all()

    def test_single_node_no_edges_block(self, rngs):
        """Test GraphAttentionBlock with a single node and no edges."""
        block = GraphAttentionBlock(
            hidden_dim=8,
            num_heads=2,
            edge_features=4,
            dropout_rate=0.0,
            rngs=rngs,
        )
        node_features = jax.random.normal(jax.random.key(0), (1, 8))
        edge_index = jnp.zeros((2, 0), dtype=jnp.int32)
        edge_features = jnp.zeros((0, 4))

        output = block(node_features, edge_index, edge_features)
        assert output.shape == (1, 8)
        assert jnp.isfinite(output).all()

    def test_self_loop_edges(self, rngs):
        """Test with self-loop edges."""
        layer = GraphAttentionLayer(
            in_features=8,
            out_features=16,
            num_heads=2,
            edge_features=4,
            dropout_rate=0.0,
            rngs=rngs,
        )
        n_nodes = 3
        node_features = jax.random.normal(jax.random.key(0), (n_nodes, 8))
        edge_index = jnp.array([[0, 1, 2], [0, 1, 2]])  # All self-loops
        edge_features = jax.random.normal(jax.random.key(1), (3, 4))

        output = layer(node_features, edge_index, edge_features)
        assert output.shape == (n_nodes, 16)
        assert jnp.isfinite(output).all()

    def test_large_graph(self, rngs):
        """Test with a larger graph (100 nodes, 500 edges)."""
        key = jax.random.key(0)
        n_nodes = 100
        n_edges = 500
        in_features = 32
        edge_feat_dim = 8

        layer = GraphAttentionLayer(
            in_features=in_features,
            out_features=64,
            num_heads=4,
            edge_features=edge_feat_dim,
            dropout_rate=0.0,
            rngs=rngs,
        )

        k1, k2, k3, k4 = jax.random.split(key, 4)
        node_features = jax.random.normal(k1, (n_nodes, in_features))
        sources = jax.random.randint(k2, (n_edges,), 0, n_nodes)
        targets = jax.random.randint(k3, (n_edges,), 0, n_nodes)
        edge_index = jnp.stack([sources, targets], axis=0)
        edge_features = jax.random.normal(k4, (n_edges, edge_feat_dim))

        output = layer(node_features, edge_index, edge_features)
        assert output.shape == (n_nodes, 64)
        assert jnp.isfinite(output).all()

    def test_large_graph_block(self, rngs):
        """Test GraphAttentionBlock with larger graph."""
        key = jax.random.key(1)
        n_nodes = 100
        n_edges = 500
        hidden_dim = 32
        edge_feat_dim = 8

        block = GraphAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=edge_feat_dim,
            dropout_rate=0.0,
            rngs=rngs,
        )

        k1, k2, k3, k4 = jax.random.split(key, 4)
        node_features = jax.random.normal(k1, (n_nodes, hidden_dim))
        sources = jax.random.randint(k2, (n_edges,), 0, n_nodes)
        targets = jax.random.randint(k3, (n_edges,), 0, n_nodes)
        edge_index = jnp.stack([sources, targets], axis=0)
        edge_features = jax.random.normal(k4, (n_edges, edge_feat_dim))

        output = block(node_features, edge_index, edge_features)
        assert output.shape == (n_nodes, hidden_dim)
        assert jnp.isfinite(output).all()

    def test_gradient_with_single_edge(self, rngs):
        """Test gradient flow with a minimal graph (2 nodes, 1 edge)."""
        layer = GraphAttentionLayer(
            in_features=8,
            out_features=16,
            num_heads=2,
            edge_features=4,
            dropout_rate=0.0,
            rngs=rngs,
        )
        node_features = jax.random.normal(jax.random.key(0), (2, 8))
        edge_index = jnp.array([[0], [1]])
        edge_features = jax.random.normal(jax.random.key(1), (1, 4))

        def loss_fn(nf):
            return layer(nf, edge_index, edge_features).sum()

        grads = jax.grad(loss_fn)(node_features)
        assert grads.shape == node_features.shape
        assert jnp.isfinite(grads).all()

    def test_nodes_with_no_incoming_edges(self, rngs):
        """Test that nodes with no incoming edges still get valid outputs."""
        layer = GraphAttentionLayer(
            in_features=8,
            out_features=16,
            num_heads=2,
            edge_features=4,
            dropout_rate=0.0,
            rngs=rngs,
        )
        # Node 2 has no incoming edges
        node_features = jax.random.normal(jax.random.key(0), (3, 8))
        edge_index = jnp.array([[0, 1], [1, 0]])  # 0->1, 1->0, node 2 isolated
        edge_features = jax.random.normal(jax.random.key(1), (2, 4))

        output = layer(node_features, edge_index, edge_features)
        assert output.shape == (3, 16)
        assert jnp.isfinite(output).all()


# =============================================================================
# TestGATv2Layer
# =============================================================================


class TestGATv2Layer:
    """Tests for GATv2Layer -- GATv2-style attention with LeakyReLU before attention."""

    def test_forward_shape(self, rngs, small_graph):
        """Test forward pass produces correct output shape."""
        out_features = 32
        layer = GATv2Layer(
            in_features=small_graph["node_feat_dim"],
            out_features=out_features,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        output = layer(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert output.shape == (small_graph["n_nodes"], out_features)

    def test_forward_finite(self, rngs, small_graph):
        """Test that output values are finite."""
        layer = GATv2Layer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        output = layer(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert jnp.isfinite(output).all()

    def test_multi_head_different_results(self, rngs, small_graph):
        """Test that different head counts produce different outputs."""
        layer_2h = GATv2Layer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=2,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )
        layer_4h = GATv2Layer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )
        out_2h = layer_2h(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        out_4h = layer_4h(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert not jnp.allclose(out_2h, out_4h, atol=1e-3)

    def test_leaky_relu_before_attention(self, rngs, small_graph):
        """Test that GATv2 produces different outputs from GAT (different architecture)."""
        gatv2 = GATv2Layer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )
        gat = GraphAttentionLayer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=nnx.Rngs(0),
        )
        out_v2 = gatv2(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        out_v1 = gat(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        # GATv2 has a fundamentally different attention mechanism, outputs differ
        assert not jnp.allclose(out_v1, out_v2, atol=1e-3)

    def test_edge_features_influence_output(self, rngs, small_graph):
        """Test that edge features influence output via non-zero gradient."""
        layer = GATv2Layer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        node_features = small_graph["node_features"]
        edge_index = small_graph["edge_index"]

        def loss_fn(ef):
            return layer(node_features, edge_index, ef).sum()

        grad = jax.grad(loss_fn)(small_graph["edge_features"])
        assert jnp.any(grad != 0.0), "Edge features should influence output"

    def test_gradient_flow_through_node_features(self, rngs, small_graph):
        """Test that gradients flow through node features."""
        layer = GATv2Layer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        edge_index = small_graph["edge_index"]
        edge_features = small_graph["edge_features"]

        def loss_fn(nf):
            return layer(nf, edge_index, edge_features).sum()

        grads = jax.grad(loss_fn)(small_graph["node_features"])
        assert grads.shape == small_graph["node_features"].shape
        assert jnp.isfinite(grads).all()

    def test_gradient_flow_through_parameters(self, rngs, small_graph):
        """Test that gradients flow through layer parameters."""
        layer = GATv2Layer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )

        @nnx.value_and_grad
        def loss_fn(model):
            return model(
                small_graph["node_features"],
                small_graph["edge_index"],
                small_graph["edge_features"],
            ).sum()

        _, grads = loss_fn(layer)
        assert hasattr(grads, "left_proj")
        assert hasattr(grads, "right_proj")
        assert hasattr(grads, "attn_vector")
        assert hasattr(grads, "value_proj")
        assert hasattr(grads, "output_proj")

    def test_jit_compatible(self, rngs, small_graph):
        """Test that the layer works under JIT compilation."""
        layer = GATv2Layer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )

        @jax.jit
        def forward(nf, ei, ef):
            return layer(nf, ei, ef)

        output = forward(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert output.shape == (small_graph["n_nodes"], 32)
        assert jnp.isfinite(output).all()

    def test_empty_graph(self, rngs):
        """Test with a single node and no edges."""
        layer = GATv2Layer(
            in_features=8,
            out_features=16,
            num_heads=2,
            edge_features=4,
            dropout_rate=0.0,
            rngs=rngs,
        )
        node_features = jax.random.normal(jax.random.key(0), (1, 8))
        edge_index = jnp.zeros((2, 0), dtype=jnp.int32)
        edge_features = jnp.zeros((0, 4))

        output = layer(node_features, edge_index, edge_features)
        assert output.shape == (1, 16)
        assert jnp.isfinite(output).all()

    def test_negative_slope_parameter(self, rngs, small_graph):
        """Test that negative_slope parameter is configurable."""
        layer = GATv2Layer(
            in_features=small_graph["node_feat_dim"],
            out_features=32,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            negative_slope=0.1,
            rngs=rngs,
        )
        output = layer(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert output.shape == (small_graph["n_nodes"], 32)
        assert jnp.isfinite(output).all()


# =============================================================================
# TestGATv2Block
# =============================================================================


class TestGATv2Block:
    """Tests for GATv2Block (GATv2 attention + LayerNorm + residual + FFN)."""

    def test_forward_shape(self, rngs, small_graph):
        """Test forward pass preserves node feature dimension."""
        hidden_dim = small_graph["node_feat_dim"]
        block = GATv2Block(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        output = block(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert output.shape == small_graph["node_features"].shape

    def test_forward_finite(self, rngs, small_graph):
        """Test that output values are finite."""
        hidden_dim = small_graph["node_feat_dim"]
        block = GATv2Block(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        output = block(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert jnp.isfinite(output).all()

    def test_residual_connection(self, rngs, small_graph):
        """Test that residual connection means output is not zero."""
        hidden_dim = small_graph["node_feat_dim"]
        block = GATv2Block(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        output = block(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert not jnp.allclose(output, 0.0)

    def test_gradient_flow_through_block(self, rngs, small_graph):
        """Test that gradients flow through the entire block."""
        hidden_dim = small_graph["node_feat_dim"]
        block = GATv2Block(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )
        edge_index = small_graph["edge_index"]
        edge_features = small_graph["edge_features"]

        def loss_fn(nf):
            return block(nf, edge_index, edge_features).sum()

        grads = jax.grad(loss_fn)(small_graph["node_features"])
        assert grads.shape == small_graph["node_features"].shape
        assert jnp.isfinite(grads).all()

    def test_gradient_flow_through_parameters(self, rngs, small_graph):
        """Test that gradients flow to all block parameters."""
        hidden_dim = small_graph["node_feat_dim"]
        block = GATv2Block(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )

        @nnx.value_and_grad
        def loss_fn(model):
            return model(
                small_graph["node_features"],
                small_graph["edge_index"],
                small_graph["edge_features"],
            ).sum()

        _, grads = loss_fn(block)
        assert hasattr(grads, "attention")
        assert hasattr(grads, "layer_norm1")
        assert hasattr(grads, "layer_norm2")
        assert hasattr(grads, "ff_linear1")
        assert hasattr(grads, "ff_linear2")

    def test_jit_compatible(self, rngs, small_graph):
        """Test that GATv2Block works under JIT compilation."""
        hidden_dim = small_graph["node_feat_dim"]
        block = GATv2Block(
            hidden_dim=hidden_dim,
            num_heads=4,
            edge_features=small_graph["edge_feat_dim"],
            dropout_rate=0.0,
            rngs=rngs,
        )

        @jax.jit
        def forward(node_features, edge_index, edge_features):
            return block(node_features, edge_index, edge_features)

        output = forward(
            small_graph["node_features"],
            small_graph["edge_index"],
            small_graph["edge_features"],
        )
        assert output.shape == small_graph["node_features"].shape
        assert jnp.isfinite(output).all()
