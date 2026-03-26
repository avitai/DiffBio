"""Shared helpers for graph-based drug-discovery operators."""

import logging
from typing import Any, Mapping

import jax.numpy as jnp
from flax import nnx

from diffbio.operators.drug_discovery.message_passing import StackedMessagePassing

logger = logging.getLogger(__name__)


def stabilize_operator_id(module: Any) -> None:
    """Mark operator unique ID as static for NNX/JAX transformations."""
    module._unique_id = nnx.static(module._unique_id)


def ensure_rngs(rngs: nnx.Rngs | None) -> nnx.Rngs:
    """Return a default RNG container when one is not provided."""
    if rngs is not None:
        return rngs
    return nnx.Rngs(0)


def build_encoder(
    *,
    hidden_dim: int,
    num_layers: int,
    in_features: int,
    rngs: nnx.Rngs,
    num_edge_features: int | None = None,
) -> StackedMessagePassing:
    """Create a message-passing encoder with optional edge features."""
    encoder_kwargs: dict[str, Any] = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "in_features": in_features,
        "rngs": rngs,
    }
    if num_edge_features is not None:
        encoder_kwargs["num_edge_features"] = num_edge_features
    return StackedMessagePassing(**encoder_kwargs)


def initialize_graph_encoder(
    module: Any,
    *,
    rngs: nnx.Rngs | None,
    hidden_dim: int,
    num_layers: int,
    in_features: int,
    num_edge_features: int | None = None,
    attr: str = "encoder",
) -> nnx.Rngs:
    """Stabilize ID, ensure RNGs, and attach a message-passing encoder."""
    stabilize_operator_id(module)
    resolved_rngs = ensure_rngs(rngs)
    setattr(
        module,
        attr,
        build_encoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            in_features=in_features,
            rngs=resolved_rngs,
            num_edge_features=num_edge_features,
        ),
    )
    return resolved_rngs


def unpack_graph_inputs(data: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Extract standard molecular graph tensors from an input dictionary."""
    return (
        data["node_features"],
        data["adjacency"],
        data.get("edge_features"),
        data.get("node_mask"),
    )


def graph_sum_readout(
    data: Mapping[str, Any],
    encoder: StackedMessagePassing,
    *,
    dropout: nnx.Dropout | None = None,
) -> jnp.ndarray:
    """Encode a graph and sum-pool node states into a graph representation."""
    node_features, adjacency, edge_features, node_mask = unpack_graph_inputs(data)

    node_hidden = encoder(node_features, adjacency, edge_features)
    if node_mask is not None:
        node_hidden = node_hidden * node_mask[:, None]

    graph_repr = jnp.sum(node_hidden, axis=0)
    if dropout is not None:
        graph_repr = dropout(graph_repr)
    return graph_repr


def attach_fingerprint(data: Mapping[str, Any], fingerprint: Any) -> dict[str, Any]:
    """Return a shallow copy with a standardized fingerprint output field."""
    return {**data, "fingerprint": fingerprint}
