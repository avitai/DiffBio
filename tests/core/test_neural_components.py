"""Tests for reusable neural network components.

Following TDD: These tests define the expected behavior for neural network
building blocks that are specific to DiffBio (not available in Flax NNX).

Note: PositionalEncoding and ResidualBlock are imported from artifex.
These tests only cover DiffBio-specific components.
"""

import jax
import jax.numpy as jnp

from diffbio.constants import DEFAULT_HIDDEN_DIM, DEFAULT_TEMPERATURE


class TestGumbelSoftmaxModule:
    """Tests for GumbelSoftmax neural network module."""

    def test_init(self, rngs):
        """Test GumbelSoftmax initialization."""
        from diffbio.core.neural_components import GumbelSoftmaxModule

        module = GumbelSoftmaxModule(temperature=DEFAULT_TEMPERATURE, rngs=rngs)
        assert module is not None

    def test_forward_shape(self, rngs):
        """Test output shape matches input."""
        from diffbio.core.neural_components import GumbelSoftmaxModule

        module = GumbelSoftmaxModule(temperature=DEFAULT_TEMPERATURE, rngs=rngs)
        num_classes = 3
        batch_size = 2
        logits = jnp.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
        output = module(logits)

        assert output.shape == (batch_size, num_classes)

    def test_output_sums_to_one(self, rngs):
        """Test output rows sum to 1."""
        from diffbio.core.neural_components import GumbelSoftmaxModule

        module = GumbelSoftmaxModule(temperature=DEFAULT_TEMPERATURE, rngs=rngs)
        logits = jnp.array([[1.0, 2.0, 3.0]])
        output = module(logits)

        row_sum = jnp.sum(output, axis=-1)
        assert jnp.allclose(row_sum, 1.0, atol=1e-5)

    def test_hard_mode(self, rngs):
        """Test hard mode produces one-hot output."""
        from diffbio.core.neural_components import GumbelSoftmaxModule

        low_temp = 0.1
        module = GumbelSoftmaxModule(temperature=low_temp, hard=True, rngs=rngs)
        logits = jnp.array([[1.0, 5.0, 2.0]])
        output = module(logits)

        # One value should be close to 1, others close to 0
        max_val_threshold = 0.9
        assert jnp.max(output) > max_val_threshold

    def test_differentiable(self, rngs):
        """Test gradients flow through module."""
        from diffbio.core.neural_components import GumbelSoftmaxModule

        module = GumbelSoftmaxModule(temperature=DEFAULT_TEMPERATURE, rngs=rngs)

        def loss_fn(logits):
            return jnp.sum(module(logits))

        logits = jnp.array([[1.0, 2.0, 3.0]])
        grads = jax.grad(loss_fn)(logits)

        assert grads is not None
        assert jnp.all(jnp.isfinite(grads))

    def test_jit_compatible(self, rngs):
        """Test module works with JIT."""
        from diffbio.core.neural_components import GumbelSoftmaxModule

        module = GumbelSoftmaxModule(temperature=DEFAULT_TEMPERATURE, rngs=rngs)

        @jax.jit
        def forward(logits):
            return module(logits)

        logits = jnp.array([[1.0, 2.0, 3.0]])
        output = forward(logits)
        assert output.shape == logits.shape


class TestGraphMessagePassing:
    """Tests for graph message passing layer."""

    def test_init(self, rngs):
        """Test GraphMessagePassing initialization."""
        from diffbio.core.neural_components import GraphMessagePassing

        node_feat_dim = 32
        edge_feat_dim = 8
        layer = GraphMessagePassing(
            node_features=node_feat_dim,
            edge_features=edge_feat_dim,
            hidden_dim=DEFAULT_HIDDEN_DIM,
            rngs=rngs,
        )
        assert layer is not None

    def test_forward_shape(self, rngs):
        """Test output shapes match expectations."""
        from diffbio.core.neural_components import GraphMessagePassing

        node_feat_dim = 4
        edge_feat_dim = 2
        hidden_dim = 8
        layer = GraphMessagePassing(
            node_features=node_feat_dim,
            edge_features=edge_feat_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

        num_nodes = 5
        num_edges = 8
        node_feat = jnp.ones((num_nodes, node_feat_dim))
        edge_feat = jnp.ones((num_edges, edge_feat_dim))
        edge_index = jnp.array([[0, 0, 1, 1, 2, 2, 3, 4], [1, 2, 2, 3, 3, 4, 4, 0]])

        output = layer(node_feat, edge_feat, edge_index)

        # Output should have same number of nodes with hidden_dim features
        assert output.shape == (num_nodes, hidden_dim)

    def test_differentiable(self, rngs):
        """Test gradients flow through layer."""
        from diffbio.core.neural_components import GraphMessagePassing

        node_feat_dim = 4
        edge_feat_dim = 2
        hidden_dim = 8
        layer = GraphMessagePassing(
            node_features=node_feat_dim,
            edge_features=edge_feat_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

        num_edges = 3  # Must match edge_index shape

        def loss_fn(node_feat):
            edge_feat = jnp.ones((num_edges, edge_feat_dim))
            edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])  # 3 edges
            return jnp.sum(layer(node_feat, edge_feat, edge_index))

        num_nodes = 3
        node_feat = jnp.ones((num_nodes, node_feat_dim))
        grads = jax.grad(loss_fn)(node_feat)

        assert grads is not None
        assert grads.shape == node_feat.shape

    def test_aggregation_types(self, rngs):
        """Test different aggregation methods."""
        from diffbio.core.neural_components import GraphMessagePassing

        node_feat_dim = 4
        edge_feat_dim = 2
        hidden_dim = 8
        num_nodes = 3
        num_edges = 4

        for agg in ["sum", "mean", "max"]:
            layer = GraphMessagePassing(
                node_features=node_feat_dim,
                edge_features=edge_feat_dim,
                hidden_dim=hidden_dim,
                aggregation=agg,  # pyright: ignore[reportArgumentType]
                rngs=rngs,
            )

            node_feat = jnp.ones((num_nodes, node_feat_dim))
            edge_feat = jnp.ones((num_edges, edge_feat_dim))
            edge_index = jnp.array([[0, 0, 1, 2], [1, 2, 2, 0]])

            output = layer(node_feat, edge_feat, edge_index)
            assert output.shape == (num_nodes, hidden_dim)

    def test_jit_compatible(self, rngs):
        """Test layer works with JIT."""
        from diffbio.core.neural_components import GraphMessagePassing

        node_feat_dim = 4
        edge_feat_dim = 2
        hidden_dim = 8
        layer = GraphMessagePassing(
            node_features=node_feat_dim,
            edge_features=edge_feat_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

        @jax.jit
        def forward(node_feat, edge_feat, edge_index):
            return layer(node_feat, edge_feat, edge_index)

        num_nodes = 3
        num_edges = 4
        node_feat = jnp.ones((num_nodes, node_feat_dim))
        edge_feat = jnp.ones((num_edges, edge_feat_dim))
        edge_index = jnp.array([[0, 0, 1, 2], [1, 2, 2, 0]])

        output = forward(node_feat, edge_feat, edge_index)
        assert output.shape == (num_nodes, hidden_dim)


class TestEdgeCases:
    """Test edge cases for neural components."""

    def test_gumbel_single_class(self, rngs):
        """Test GumbelSoftmax with single class."""
        from diffbio.core.neural_components import GumbelSoftmaxModule

        module = GumbelSoftmaxModule(temperature=DEFAULT_TEMPERATURE, rngs=rngs)
        logits = jnp.array([[1.0]])
        output = module(logits)

        assert output.shape == (1, 1)
        assert jnp.allclose(output, 1.0)

    def test_gumbel_batch_processing(self, rngs):
        """Test GumbelSoftmax with batched input."""
        from diffbio.core.neural_components import GumbelSoftmaxModule

        module = GumbelSoftmaxModule(temperature=DEFAULT_TEMPERATURE, rngs=rngs)
        batch_size = 10
        num_classes = 5
        batch_logits = jnp.ones((batch_size, num_classes))
        output = module(batch_logits)

        assert output.shape == (batch_size, num_classes)
        # Each row should sum to 1
        assert jnp.allclose(jnp.sum(output, axis=-1), 1.0, atol=1e-5)

    def test_graph_no_edges(self, rngs):
        """Test GraphMessagePassing with isolated nodes."""
        from diffbio.core.neural_components import GraphMessagePassing

        node_feat_dim = 4
        edge_feat_dim = 2
        hidden_dim = 8
        layer = GraphMessagePassing(
            node_features=node_feat_dim,
            edge_features=edge_feat_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

        num_nodes = 3
        node_feat = jnp.ones((num_nodes, node_feat_dim))
        edge_feat = jnp.ones((0, edge_feat_dim))  # No edges
        edge_index = jnp.zeros((2, 0), dtype=jnp.int32)  # Empty edge index

        output = layer(node_feat, edge_feat, edge_index)
        assert output.shape == (num_nodes, hidden_dim)

    def test_graph_self_loops(self, rngs):
        """Test GraphMessagePassing with self-loops."""
        from diffbio.core.neural_components import GraphMessagePassing

        node_feat_dim = 4
        edge_feat_dim = 2
        hidden_dim = 8
        layer = GraphMessagePassing(
            node_features=node_feat_dim,
            edge_features=edge_feat_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

        num_nodes = 3
        node_feat = jnp.ones((num_nodes, node_feat_dim))
        edge_feat = jnp.ones((num_nodes, edge_feat_dim))
        # Self-loops
        edge_index = jnp.array([[0, 1, 2], [0, 1, 2]])

        output = layer(node_feat, edge_feat, edge_index)
        assert output.shape == (num_nodes, hidden_dim)
