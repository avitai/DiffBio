"""Molecular property prediction operator.

This module implements a ChemProp-style molecular property predictor
using message passing neural networks.
"""

from dataclasses import dataclass
from typing import Any
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx

from diffbio.operators.drug_discovery._graph_utils import (
    graph_sum_readout,
    initialize_graph_encoder,
)


@dataclass
class MolecularPropertyConfig(OperatorConfig):
    """Configuration for molecular property predictor.

    Attributes:
        hidden_dim: Hidden dimension for message passing layers.
        num_message_passing_steps: Number of message passing iterations.
        num_output_tasks: Number of prediction tasks (multi-task learning).
        dropout_rate: Dropout rate for regularization.
        in_features: Number of input node features (default: DEFAULT_ATOM_FEATURES=34).
        num_edge_features: Number of edge/bond features.
        stochastic: Whether operator uses random sampling.
        stream_name: Optional stream name for data routing.
    """

    hidden_dim: int = 300
    num_message_passing_steps: int = 3
    num_output_tasks: int = 1
    dropout_rate: float = 0.0
    in_features: int = 4  # Default for tests; use DEFAULT_ATOM_FEATURES for real molecules
    num_edge_features: int = 4
    stochastic: bool = False
    stream_name: str | None = None


class MolecularPropertyPredictor(OperatorModule):
    """ChemProp-style molecular property predictor.

    Implements a directed message passing neural network (D-MPNN) for
    predicting molecular properties from graph representations.

    The architecture consists of:
    1. Message passing layers to compute atom representations
    2. Graph-level readout via sum pooling
    3. Feed-forward network for property prediction

    Example:
        ```python
        config = MolecularPropertyConfig(hidden_dim=64, num_output_tasks=3)
        predictor = MolecularPropertyPredictor(config, rngs=nnx.Rngs(42))
        data = {
            "node_features": node_features,
            "adjacency": adjacency,
            "node_mask": mask,
        }
        result, state, meta = predictor.apply(data, {}, None)
        predictions = result["predictions"]  # shape: (3,)
        ```
    """

    def __init__(self, config: MolecularPropertyConfig, *, rngs: nnx.Rngs | None = None):
        """Initialize molecular property predictor.

        Args:
            config: Predictor configuration.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)

        rngs = initialize_graph_encoder(
            self,
            rngs=rngs,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_message_passing_steps,
            in_features=config.in_features,
            num_edge_features=config.num_edge_features,
        )

        # Feed-forward network for prediction
        self.ffn = nnx.Sequential(
            nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs),
            nnx.Linear(config.hidden_dim, config.num_output_tasks, rngs=rngs),
        )

        # Dropout for regularization
        if config.dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Predict molecular properties from graph representation.

        Args:
            data: Input data containing:
                - node_features: (num_nodes, num_features) atom features
                - adjacency: (num_nodes, num_nodes) adjacency matrix
                - edge_features: Optional (num_nodes, num_nodes, num_edge_features)
                - node_mask: (num_nodes,) mask for valid nodes
            state: Per-element state (passed through).
            metadata: Optional metadata.
            random_params: Unused random parameters.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of:
                - data with added "predictions" key
                - unchanged state
                - unchanged metadata
        """
        graph_repr = graph_sum_readout(data, self.encoder, dropout=self.dropout)

        # Feed-forward prediction
        # Apply ReLU between layers
        h = nnx.relu(self.ffn.layers[0](graph_repr))
        if self.dropout is not None:
            h = self.dropout(h)
        predictions = self.ffn.layers[1](h)

        result = {
            **data,
            "predictions": predictions,
            "graph_representation": graph_repr,
        }

        return result, state, metadata


def create_property_predictor(
    hidden_dim: int = 300,
    num_layers: int = 3,
    num_tasks: int = 1,
    dropout_rate: float = 0.0,
    seed: int = 42,
) -> MolecularPropertyPredictor:
    """Create a molecular property predictor.

    Args:
        hidden_dim: Hidden dimension for message passing.
        num_layers: Number of message passing steps.
        num_tasks: Number of prediction tasks.
        dropout_rate: Dropout rate.
        seed: Random seed.

    Returns:
        Configured MolecularPropertyPredictor.
    """
    config = MolecularPropertyConfig(
        hidden_dim=hidden_dim,
        num_message_passing_steps=num_layers,
        num_output_tasks=num_tasks,
        dropout_rate=dropout_rate,
    )
    return MolecularPropertyPredictor(config, rngs=nnx.Rngs(seed))
