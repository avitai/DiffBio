"""Tests for diffbio.operators.assembly.gnn_assembly module.

These tests define the expected behavior of the GNNAssemblyNavigator
operator for differentiable assembly graph traversal.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.assembly.gnn_assembly import (
    GNNAssemblyNavigator,
    GNNAssemblyNavigatorConfig,
)


class TestGNNAssemblyNavigatorConfig:
    """Tests for GNNAssemblyNavigatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GNNAssemblyNavigatorConfig()
        assert config.node_features == 64
        assert config.hidden_dim == 128
        assert config.num_layers == 3
        assert config.num_heads == 4
        assert config.stochastic is False

    def test_custom_node_features(self):
        """Test custom node features dimension."""
        config = GNNAssemblyNavigatorConfig(node_features=128)
        assert config.node_features == 128

    def test_custom_hidden_dim(self):
        """Test custom hidden dimension."""
        config = GNNAssemblyNavigatorConfig(hidden_dim=256)
        assert config.hidden_dim == 256

    def test_custom_num_layers(self):
        """Test custom number of layers."""
        config = GNNAssemblyNavigatorConfig(num_layers=5)
        assert config.num_layers == 5


class TestGNNAssemblyNavigator:
    """Tests for GNNAssemblyNavigator operator."""

    @pytest.fixture
    def sample_graph(self):
        """Provide sample assembly graph data."""
        key = jax.random.key(0)
        n_nodes = 20
        n_edges = 40
        node_features = 32

        # Node features (n_nodes, node_features)
        key, subkey = jax.random.split(key)
        nodes = jax.random.normal(subkey, (n_nodes, node_features))

        # Edge indices (2, n_edges) - source and target nodes
        key, subkey = jax.random.split(key)
        sources = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        key, subkey = jax.random.split(key)
        targets = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        edge_index = jnp.stack([sources, targets], axis=0)

        # Edge features (n_edges, edge_features)
        key, subkey = jax.random.split(key)
        edge_features = jax.random.normal(subkey, (n_edges, 8))

        return {
            "node_features": nodes,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }

    @pytest.fixture
    def small_config(self):
        """Provide small config for faster tests."""
        return GNNAssemblyNavigatorConfig(
            node_features=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            edge_features=8,
        )

    def test_initialization(self, rngs, small_config):
        """Test operator initialization."""
        op = GNNAssemblyNavigator(small_config, rngs=rngs)
        assert op is not None

    def test_output_contains_updated_nodes(self, rngs, small_config, sample_graph):
        """Test that output contains updated node embeddings."""
        op = GNNAssemblyNavigator(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_graph, {}, None, None)

        assert "node_embeddings" in transformed
        assert transformed["node_embeddings"].shape == (20, 64)  # hidden_dim

    def test_output_contains_edge_scores(self, rngs, small_config, sample_graph):
        """Test that output contains edge scores."""
        op = GNNAssemblyNavigator(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_graph, {}, None, None)

        assert "edge_scores" in transformed
        assert transformed["edge_scores"].shape == (40,)  # n_edges

    def test_output_contains_traversal_probs(self, rngs, small_config, sample_graph):
        """Test that output contains traversal probabilities."""
        op = GNNAssemblyNavigator(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_graph, {}, None, None)

        assert "traversal_probs" in transformed
        assert transformed["traversal_probs"].shape == (40,)  # n_edges

    def test_traversal_probs_valid(self, rngs, small_config, sample_graph):
        """Test that traversal probabilities are valid (0-1)."""
        op = GNNAssemblyNavigator(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_graph, {}, None, None)

        probs = transformed["traversal_probs"]
        assert jnp.all(probs >= 0)
        assert jnp.all(probs <= 1)

    def test_edge_scores_finite(self, rngs, small_config, sample_graph):
        """Test that edge scores are finite."""
        op = GNNAssemblyNavigator(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_graph, {}, None, None)

        assert jnp.isfinite(transformed["edge_scores"]).all()

    def test_output_contains_path_confidence(self, rngs, small_config, sample_graph):
        """Test that output contains path confidence scores."""
        op = GNNAssemblyNavigator(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_graph, {}, None, None)

        assert "path_confidence" in transformed


class TestGradientFlow:
    """Tests for gradient flow through GNN assembly navigator."""

    @pytest.fixture
    def small_config(self):
        return GNNAssemblyNavigatorConfig(
            node_features=16,
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            edge_features=4,
        )

    def test_gradient_flows_through_gnn(self, rngs, small_config):
        """Test that gradients flow through GNN."""
        op = GNNAssemblyNavigator(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_nodes = 10
        n_edges = 20

        key, subkey = jax.random.split(key)
        nodes = jax.random.normal(subkey, (n_nodes, 16))

        key, subkey = jax.random.split(key)
        sources = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        key, subkey = jax.random.split(key)
        targets = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        edge_index = jnp.stack([sources, targets], axis=0)

        key, subkey = jax.random.split(key)
        edge_features = jax.random.normal(subkey, (n_edges, 4))

        def loss_fn(node_input):
            data = {
                "node_features": node_input,
                "edge_index": edge_index,
                "edge_features": edge_features,
            }
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["edge_scores"].sum()

        grad = jax.grad(loss_fn)(nodes)
        assert grad is not None
        assert grad.shape == nodes.shape
        assert jnp.isfinite(grad).all()

    def test_model_is_learnable(self, rngs, small_config):
        """Test that model parameters are learnable."""
        op = GNNAssemblyNavigator(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_nodes = 10
        n_edges = 20

        key, subkey = jax.random.split(key)
        nodes = jax.random.normal(subkey, (n_nodes, 16))

        key, subkey = jax.random.split(key)
        sources = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        key, subkey = jax.random.split(key)
        targets = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        edge_index = jnp.stack([sources, targets], axis=0)

        key, subkey = jax.random.split(key)
        edge_features = jax.random.normal(subkey, (n_edges, 4))

        data = {
            "node_features": nodes,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["edge_scores"].sum()

        loss, grads = loss_fn(op)

        # Check GNN layers have gradients
        assert hasattr(grads, "gnn_layers")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def small_config(self):
        return GNNAssemblyNavigatorConfig(
            node_features=16,
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            edge_features=4,
        )

    def test_apply_is_jit_compatible(self, rngs, small_config):
        """Test that apply method works with JIT."""
        op = GNNAssemblyNavigator(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_nodes = 10
        n_edges = 20

        key, subkey = jax.random.split(key)
        nodes = jax.random.normal(subkey, (n_nodes, 16))

        key, subkey = jax.random.split(key)
        sources = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        key, subkey = jax.random.split(key)
        targets = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        edge_index = jnp.stack([sources, targets], axis=0)

        key, subkey = jax.random.split(key)
        edge_features = jax.random.normal(subkey, (n_edges, 4))

        data = {
            "node_features": nodes,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["edge_scores"]).all()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_graph(self, rngs):
        """Test with small graph."""
        config = GNNAssemblyNavigatorConfig(
            node_features=8,
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            edge_features=4,
        )
        op = GNNAssemblyNavigator(config, rngs=rngs)

        key = jax.random.key(0)
        n_nodes = 5
        n_edges = 8

        key, subkey = jax.random.split(key)
        nodes = jax.random.normal(subkey, (n_nodes, 8))

        key, subkey = jax.random.split(key)
        sources = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        key, subkey = jax.random.split(key)
        targets = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        edge_index = jnp.stack([sources, targets], axis=0)

        key, subkey = jax.random.split(key)
        edge_features = jax.random.normal(subkey, (n_edges, 4))

        data = {
            "node_features": nodes,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }
        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["edge_scores"]).all()

    def test_large_graph(self, rngs):
        """Test with larger graph."""
        config = GNNAssemblyNavigatorConfig(
            node_features=16,
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            edge_features=4,
        )
        op = GNNAssemblyNavigator(config, rngs=rngs)

        key = jax.random.key(0)
        n_nodes = 100
        n_edges = 300

        key, subkey = jax.random.split(key)
        nodes = jax.random.normal(subkey, (n_nodes, 16))

        key, subkey = jax.random.split(key)
        sources = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        key, subkey = jax.random.split(key)
        targets = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        edge_index = jnp.stack([sources, targets], axis=0)

        key, subkey = jax.random.split(key)
        edge_features = jax.random.normal(subkey, (n_edges, 4))

        data = {
            "node_features": nodes,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }
        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["node_embeddings"].shape == (100, 32)

    def test_dense_graph(self, rngs):
        """Test with dense connectivity."""
        config = GNNAssemblyNavigatorConfig(
            node_features=8,
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            edge_features=4,
        )
        op = GNNAssemblyNavigator(config, rngs=rngs)

        key = jax.random.key(0)
        n_nodes = 10
        n_edges = 50  # Dense connections

        key, subkey = jax.random.split(key)
        nodes = jax.random.normal(subkey, (n_nodes, 8))

        key, subkey = jax.random.split(key)
        sources = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        key, subkey = jax.random.split(key)
        targets = jax.random.randint(subkey, (n_edges,), 0, n_nodes)
        edge_index = jnp.stack([sources, targets], axis=0)

        key, subkey = jax.random.split(key)
        edge_features = jax.random.normal(subkey, (n_edges, 4))

        data = {
            "node_features": nodes,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }
        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["edge_scores"]).all()
