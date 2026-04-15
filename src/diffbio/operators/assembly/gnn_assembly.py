"""Graph Neural Network-based assembly navigator.

This module provides a differentiable approach to assembly graph
traversal using message passing neural networks.

Key technique: Uses graph attention for message passing between nodes,
then scores edges for soft path selection, enabling gradient flow
through the assembly process.

Applications: Differentiable genome assembly, assembly polishing,
scaffolding optimization.

Inherits from GraphOperator to get:

- scatter_aggregate() for message aggregation
- global_pool() for graph-level pooling
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.configs import TemperatureConfig
from diffbio.core import soft_ops
from diffbio.core.base_operators import GraphOperator
from diffbio.core.gnn_components import GraphAttentionBlock
from diffbio.utils.nn_utils import init_learnable_param

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GNNAssemblyNavigatorConfig(TemperatureConfig):
    """Configuration for GNNAssemblyNavigator.

    Attributes:
        node_features: Dimension of input node features.
        hidden_dim: Hidden dimension for GNN layers.
        num_layers: Number of GNN layers.
        num_heads: Number of attention heads.
        edge_features: Dimension of edge features.
        dropout_rate: Dropout rate for regularization.
        temperature: Temperature for softmax operations.
    """

    node_features: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    edge_features: int = 8
    dropout_rate: float = 0.1
    temperature: float = 1.0


class GNNAssemblyNavigator(GraphOperator):
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

    Inherits from GraphOperator to get:

    - scatter_aggregate() for message aggregation utilities
    - global_pool() for graph-level pooling

    Uses temperature-controlled smoothing:
    - _temperature property for temperature-controlled sigmoid

    Args:
        config: GNNAssemblyNavigatorConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = GNNAssemblyNavigatorConfig(hidden_dim=128)
        navigator = GNNAssemblyNavigator(config, rngs=nnx.Rngs(42))
        data = {"node_features": nodes, "edge_index": edges, "edge_features": edge_attr}
        result, state, meta = navigator.apply(data, {}, None)
        ```
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

        # Temperature management (similar to TemperatureOperator pattern)
        if config.learnable_temperature:
            self._temperature_param = init_learnable_param(config.temperature)
        else:
            self._temperature_param = None
            self._fixed_temperature = config.temperature

        # Input projection
        self.input_projection = nnx.Linear(
            in_features=config.node_features,
            out_features=config.hidden_dim,
            rngs=rngs,
        )

        # GNN layers
        self.gnn_layers = nnx.List(
            [
                GraphAttentionBlock(
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

    @property
    def _temperature(self) -> Array | float:
        """Get current temperature value."""
        if self._temperature_param is not None:
            return jnp.abs(self._temperature_param[...]) + 1e-6
        return self._fixed_temperature

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
        # Use inherited _temperature property for temperature-controlled sigmoid
        traversal_probs = soft_ops.greater(edge_scores, 0.0, softness=self._temperature)

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
