"""Shared helpers for graph-based drug-discovery operators."""

import logging
from typing import Any, Mapping

import jax.numpy as jnp
from flax import nnx

from diffbio.operators.drug_discovery.message_passing import StackedMessagePassing
from diffbio.utils.nn_utils import ensure_rngs

logger = logging.getLogger(__name__)


def stabilize_operator_id(module: Any) -> None:
    """Mark operator unique ID as static for NNX/JAX transformations."""
    module._unique_id = nnx.static(module._unique_id)


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


def _require_config_attr(config: Any, attr: str) -> Any:
    """Read a required config attribute with a clear error message."""
    if not hasattr(config, attr):
        raise AttributeError(
            f"{type(config).__name__} must define '{attr}' for graph encoder initialization."
        )
    return getattr(config, attr)


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


def initialize_graph_encoder_from_config(
    module: Any,
    config: Any,
    *,
    rngs: nnx.Rngs | None,
    num_layers_attr: str = "num_message_passing_steps",
    hidden_dim_attr: str = "hidden_dim",
    in_features_attr: str = "in_features",
    num_edge_features_attr: str = "num_edge_features",
    attr: str = "encoder",
) -> nnx.Rngs:
    """Initialize a standard graph encoder from a config object."""
    hidden_dim = _require_config_attr(config, hidden_dim_attr)
    num_layers = _require_config_attr(config, num_layers_attr)
    in_features = _require_config_attr(config, in_features_attr)
    num_edge_features = getattr(config, num_edge_features_attr, None)

    return initialize_graph_encoder(
        module,
        rngs=rngs,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        in_features=in_features,
        num_edge_features=num_edge_features,
        attr=attr,
    )


def build_optional_dropout(rate: float, *, rngs: nnx.Rngs) -> nnx.Dropout | None:
    """Create a dropout module only when the configured rate is positive."""
    if rate <= 0:
        return None
    return nnx.Dropout(rate=rate, rngs=rngs)


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
