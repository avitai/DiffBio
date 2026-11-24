"""Shared helpers for graph-based drug-discovery operators."""

import logging
from typing import Any, Literal, Mapping

import jax.numpy as jnp
from flax import nnx

from diffbio.operators.drug_discovery.message_passing import StackedMessagePassing


logger = logging.getLogger(__name__)


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
    rngs: nnx.Rngs,
    hidden_dim: int,
    num_layers: int,
    in_features: int,
    num_edge_features: int | None = None,
    attr: str = "encoder",
) -> nnx.Rngs:
    """Ensure RNGs and attach a message-passing encoder."""
    setattr(
        module,
        attr,
        build_encoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            in_features=in_features,
            rngs=rngs,
            num_edge_features=num_edge_features,
        ),
    )
    return rngs


def initialize_graph_encoder_from_config(
    module: Any,
    config: Any,
    *,
    rngs: nnx.Rngs,
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


def graph_readout(
    data: Mapping[str, Any],
    encoder: StackedMessagePassing,
    *,
    aggregation: Literal["sum", "mean"] = "mean",
    dropout: nnx.Dropout | None = None,
) -> jnp.ndarray:
    """Encode a graph and pool its node states into one graph-level representation.

    The readout is the only route by which molecule size reaches the output: the atom features
    carry no atom count, and message passing is local. So the pooling choice decides whether the
    representation is extensive in the atom count (``"sum"``) or intensive (``"mean"``).

    Choose by the property being predicted, which is chemprop's documented rule: mean for a
    property that does not depend on the number of atoms, sum for one that does. Schweidtmann et
    al., *Computers and Chemical Engineering* 172 (2023) 108202, reach the same conclusion from
    the physics and add that for a size-independent property the poolings should be compared
    rather than assumed.

    ``"max"`` is deliberately absent: chemprop's registry offers mean, sum, norm and attentive
    aggregation and no max for graph readout, and ``scatter_aggregate``'s ``"max"`` in this
    package is *message* aggregation, a different operation. chemprop's ``norm`` (a sum divided
    by a constant) and its attention readout are also unimplemented here; both leave the output
    extensive or add parameters, and neither has been measured against this package's operators.

    Args:
        data: Graph arrays, as ``unpack_graph_inputs`` reads them. ``node_mask`` is optional;
            when absent every row counts as a real atom.
        encoder: Message-passing encoder producing one hidden state per row.
        aggregation: ``"mean"`` to divide by the number of unmasked atoms, ``"sum"`` to total
            them.
        dropout: Optional dropout applied to the pooled representation.

    Returns:
        The pooled graph representation, of shape ``(hidden_dim,)``.

    Raises:
        ValueError: If ``aggregation`` is not a supported mode.
    """
    if aggregation not in {"sum", "mean"}:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    node_features, adjacency, edge_features, node_mask = unpack_graph_inputs(data)

    node_hidden = encoder(node_features, adjacency, edge_features)
    if node_mask is not None:
        node_hidden = node_hidden * node_mask[:, None]

    graph_repr = jnp.sum(node_hidden, axis=0)
    if aggregation == "mean":
        # `jnp.mean(..., where=)` is the standard-library masked mean and divides by the mask's
        # own sum, but it has no guard for an empty mask -- `_var` returns NaN deliberately while
        # `_mean` does not guard at all -- so an all-padding graph would return NaN and poison
        # the gradient. Clamping the divisor is what chemprop (`sums / counts.clamp(min=1)`) and
        # jraph (`maximum(count, 1)`) both do, and it is exact for every non-empty graph.
        atom_count = jnp.shape(node_hidden)[0] if node_mask is None else jnp.sum(node_mask)
        graph_repr = graph_repr / jnp.maximum(atom_count, 1)

    if dropout is not None:
        graph_repr = dropout(graph_repr)
    return graph_repr


def attach_fingerprint(data: Mapping[str, Any], fingerprint: Any) -> dict[str, Any]:
    """Return a shallow copy with a standardized fingerprint output field."""
    return {**data, "fingerprint": fingerprint}
