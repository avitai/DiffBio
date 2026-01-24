"""Differentiable molecular fingerprint operator.

This module implements neural graph fingerprints that provide
differentiable alternatives to traditional molecular fingerprints.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx

from diffbio.operators.drug_discovery.message_passing import StackedMessagePassing


@dataclass
class MolecularFingerprintConfig(OperatorConfig):
    """Configuration for molecular fingerprint operator.

    Attributes:
        fingerprint_dim: Dimension of output fingerprint vector.
        hidden_dim: Hidden dimension for graph convolutions.
        num_layers: Number of graph convolution layers.
        in_features: Number of input node features (default: DEFAULT_ATOM_FEATURES=34).
        normalize: Whether to L2-normalize the fingerprint.
        stochastic: Whether operator uses random sampling.
        stream_name: Optional stream name for data routing.
    """

    fingerprint_dim: int = 256
    hidden_dim: int = 128
    num_layers: int = 3
    in_features: int = 4  # Default for tests; use DEFAULT_ATOM_FEATURES for real molecules
    normalize: bool = False
    stochastic: bool = False
    stream_name: str | None = None


class DifferentiableMolecularFingerprint(OperatorModule):
    """Neural graph fingerprint operator.

    Computes learned molecular fingerprints using graph neural networks.
    Unlike traditional fingerprints (e.g., ECFP/Morgan), these are fully
    differentiable and can be optimized for specific tasks.

    The fingerprint is computed by:
    1. Message passing to compute atom representations
    2. Sum pooling to get graph-level representation
    3. Linear projection to fingerprint dimension
    4. Optional L2 normalization

    Example:
        >>> config = MolecularFingerprintConfig(fingerprint_dim=128)
        >>> fp_op = DifferentiableMolecularFingerprint(config, rngs=nnx.Rngs(42))
        >>> data = {"node_features": nodes, "adjacency": adj, "node_mask": mask}
        >>> result, _, _ = fp_op.apply(data, {}, None)
        >>> fingerprint = result["fingerprint"]  # shape: (128,)
    """

    def __init__(
        self, config: MolecularFingerprintConfig, *, rngs: nnx.Rngs | None = None
    ):
        """Initialize fingerprint operator.

        Args:
            config: Fingerprint configuration.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)
        self.config: MolecularFingerprintConfig = config

        # Fix: wrap _unique_id as static for jax.grad compatibility
        # (datarax stores it as plain int which causes gradient errors)
        self._unique_id = nnx.static(self._unique_id)

        if rngs is None:
            rngs = nnx.Rngs(0)

        # Graph encoder
        self.encoder = StackedMessagePassing(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            in_features=config.in_features,
            rngs=rngs,
        )

        # Projection to fingerprint dimension
        self.projection = nnx.Linear(
            in_features=config.hidden_dim,
            out_features=config.fingerprint_dim,
            rngs=rngs,
        )

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Compute molecular fingerprint.

        Args:
            data: Input data containing:
                - node_features: (num_nodes, num_features) atom features
                - adjacency: (num_nodes, num_nodes) adjacency matrix
                - node_mask: (num_nodes,) mask for valid nodes
            state: Per-element state (passed through).
            metadata: Optional metadata.
            random_params: Unused random parameters.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of:
                - data with added "fingerprint" key
                - unchanged state
                - unchanged metadata
        """
        node_features = data["node_features"]
        adjacency = data["adjacency"]
        edge_features = data.get("edge_features")
        node_mask = data.get("node_mask")

        # Message passing
        node_hidden = self.encoder(node_features, adjacency, edge_features)

        # Apply mask
        if node_mask is not None:
            node_hidden = node_hidden * node_mask[:, None]

        # Sum pooling
        graph_repr = jnp.sum(node_hidden, axis=0)

        # Project to fingerprint dimension
        fingerprint = self.projection(graph_repr)

        # Optional normalization
        if self.config.normalize:
            fingerprint = fingerprint / (jnp.linalg.norm(fingerprint) + 1e-8)

        result = {
            **data,
            "fingerprint": fingerprint,
        }

        return result, state, metadata


def create_fingerprint_operator(
    fingerprint_dim: int = 256,
    num_layers: int = 3,
    normalize: bool = False,
    seed: int = 42,
) -> DifferentiableMolecularFingerprint:
    """Create a molecular fingerprint operator.

    Args:
        fingerprint_dim: Output fingerprint dimension.
        num_layers: Number of message passing layers.
        normalize: Whether to L2-normalize output.
        seed: Random seed.

    Returns:
        Configured DifferentiableMolecularFingerprint.
    """
    config = MolecularFingerprintConfig(
        fingerprint_dim=fingerprint_dim,
        num_layers=num_layers,
        normalize=normalize,
    )
    return DifferentiableMolecularFingerprint(config, rngs=nnx.Rngs(seed))
