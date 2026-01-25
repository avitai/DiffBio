"""AttentiveFP: Attention-based graph fingerprint for molecular property prediction.

This module implements the AttentiveFP architecture from Xiong et al. 2019,
which combines graph attention mechanisms with GRU cells for molecular
representation learning.

The architecture provides:

    - Interpretable attention weights showing atom importance
    - Two-level aggregation (atom-level and molecule-level)
    - GRU-based state updates for iterative refinement

References:
    - Xiong et al. "Pushing the Boundaries of Molecular Representation for
      Drug Discovery with the Graph Attention Mechanism" JCIM 2019
    - https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.AttentiveFP.html
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx


@dataclass
class AttentiveFPConfig(OperatorConfig):
    """Configuration for AttentiveFP operator.

    Attributes:
        hidden_dim: Hidden dimension for GNN layers (default: 200).
        out_dim: Output fingerprint dimension (default: 200).
        num_layers: Number of atom-level attention layers (default: 2).
        num_timesteps: Number of molecule-level GRU iterations (default: 2).
        dropout_rate: Dropout rate for regularization (default: 0.0).
        in_features: Number of input node features (default: 39).
        edge_dim: Edge feature dimension (default: 10).
        negative_slope: LeakyReLU negative slope (default: 0.2).
        stochastic: Whether operator uses random sampling.
        stream_name: Optional stream name for data routing.
    """

    hidden_dim: int = 200
    out_dim: int = 200
    num_layers: int = 2
    num_timesteps: int = 2
    dropout_rate: float = 0.0
    in_features: int = 39
    edge_dim: int = 10
    negative_slope: float = 0.2
    stochastic: bool = False
    stream_name: str | None = None


class GATEConv(nnx.Module):
    """Graph Attention with Edge features (GATE) convolution layer.

    Combines node features with edge features using attention mechanism.
    This is the core building block for AttentiveFP's atom-level processing.

    The attention mechanism computes:
        alpha_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j || We_ij]))

    where || denotes concatenation and e_ij are edge features.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        *,
        negative_slope: float = 0.2,
        rngs: nnx.Rngs,
    ):
        """Initialize GATE convolution.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            edge_dim: Edge feature dimension.
            negative_slope: LeakyReLU negative slope.
            rngs: Random number generators.
        """
        super().__init__()
        self.negative_slope = negative_slope

        # Linear transformations
        self.linear_src = nnx.Linear(in_dim, out_dim, rngs=rngs)
        self.linear_dst = nnx.Linear(in_dim, out_dim, rngs=rngs)

        # Edge feature projection (if edge_dim > 0)
        self.use_edge_features = edge_dim > 0
        if self.use_edge_features:
            self.linear_edge = nnx.Linear(edge_dim, out_dim, rngs=rngs)

        # Attention coefficients
        # Attention is computed as: a^T [src || dst || edge]
        attn_dim = out_dim * 3 if self.use_edge_features else out_dim * 2
        self.attn = nnx.Linear(attn_dim, 1, use_bias=False, rngs=rngs)

    def __call__(
        self,
        node_features: jnp.ndarray,
        adjacency: jnp.ndarray,
        edge_features: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply GATE convolution.

        Args:
            node_features: (num_nodes, in_dim) node features
            adjacency: (num_nodes, num_nodes) adjacency matrix
            edge_features: Optional (num_nodes, num_nodes, edge_dim)

        Returns:
            Tuple of (updated_features, attention_weights)
        """
        num_nodes = node_features.shape[0]

        # Transform source and destination features
        h_src = self.linear_src(node_features)  # (N, out_dim)
        h_dst = self.linear_dst(node_features)  # (N, out_dim)

        # Expand for pairwise computation
        h_src_exp = jnp.expand_dims(h_src, axis=1)  # (N, 1, out_dim)
        h_dst_exp = jnp.expand_dims(h_dst, axis=0)  # (1, N, out_dim)

        # Broadcast to (N, N, out_dim)
        h_src_broad = jnp.broadcast_to(h_src_exp, (num_nodes, num_nodes, h_src.shape[-1]))
        h_dst_broad = jnp.broadcast_to(h_dst_exp, (num_nodes, num_nodes, h_dst.shape[-1]))

        # Concatenate for attention
        if self.use_edge_features and edge_features is not None:
            h_edge = self.linear_edge(edge_features)  # (N, N, out_dim)
            attn_input = jnp.concatenate([h_src_broad, h_dst_broad, h_edge], axis=-1)
        else:
            attn_input = jnp.concatenate([h_src_broad, h_dst_broad], axis=-1)

        # Compute attention logits
        attn_logits = self.attn(attn_input).squeeze(-1)  # (N, N)

        # Apply LeakyReLU
        attn_logits = jnp.where(
            attn_logits >= 0,
            attn_logits,
            self.negative_slope * attn_logits,
        )

        # Mask non-edges with large negative value
        attn_logits = jnp.where(adjacency > 0, attn_logits, -1e9)

        # Softmax over neighbors
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # (N, N)

        # Mask attention weights for non-edges
        attn_weights = attn_weights * adjacency

        # Aggregate: weighted sum of transformed features
        out = jnp.einsum("ij,jd->id", attn_weights, h_dst)  # (N, out_dim)

        return out, attn_weights


class AttentiveFP(OperatorModule):
    """AttentiveFP: Attention-based molecular fingerprint.

    Implements the AttentiveFP architecture with:
        1. Atom-level attention layers with GRU refinement
        2. Molecule-level aggregation with attention and GRU
        3. Final projection to fingerprint dimension

    The model provides interpretable attention weights that indicate
    which atoms contribute most to the molecular representation.

    Example:
        ```python
        config = AttentiveFPConfig(hidden_dim=128, out_dim=256)
        afp = AttentiveFP(config, rngs=nnx.Rngs(42))
        data = {"node_features": nodes, "adjacency": adj, "edge_features": edges}
        result, _, _ = afp.apply(data, {}, None)
        fingerprint = result["fingerprint"]  # (256,)
        attn = result["attention_weights"]  # interpretability
        ```

    References:
        - Xiong et al. JCIM 2019
    """

    def __init__(
        self,
        config: AttentiveFPConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize AttentiveFP.

        Args:
            config: AttentiveFP configuration.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)
        self.config: AttentiveFPConfig = config

        # Fix: wrap _unique_id as static for jax.grad compatibility
        self._unique_id = nnx.static(self._unique_id)

        if rngs is None:
            rngs = nnx.Rngs(0)

        # Initial linear projection
        self.input_proj = nnx.Linear(config.in_features, config.hidden_dim, rngs=rngs)

        # Atom-level: GATE convolutions with GRU
        atom_convs = []
        atom_grus = []

        for _ in range(config.num_layers):
            in_dim = config.hidden_dim
            atom_convs.append(
                GATEConv(
                    in_dim=in_dim,
                    out_dim=config.hidden_dim,
                    edge_dim=config.edge_dim,
                    negative_slope=config.negative_slope,
                    rngs=rngs,
                )
            )
            atom_grus.append(
                nnx.GRUCell(
                    in_features=config.hidden_dim,
                    hidden_features=config.hidden_dim,
                    rngs=rngs,
                )
            )
        self.atom_convs = nnx.List(atom_convs)
        self.atom_grus = nnx.List(atom_grus)

        # Molecule-level aggregation
        # Attention for global pooling
        self.mol_attn = nnx.Linear(config.hidden_dim, 1, rngs=rngs)

        # GRU for molecule-level refinement
        mol_grus = []
        for _ in range(config.num_timesteps):
            mol_grus.append(
                nnx.GRUCell(
                    in_features=config.hidden_dim,
                    hidden_features=config.hidden_dim,
                    rngs=rngs,
                )
            )
        self.mol_grus = nnx.List(mol_grus)

        # Final projection
        self.output_proj = nnx.Linear(config.hidden_dim, config.out_dim, rngs=rngs)

        # Dropout
        self.dropout: nnx.Dropout | None = None
        if config.dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Compute AttentiveFP molecular fingerprint.

        Args:
            data: Input data containing:
                - node_features: (num_nodes, in_features) atom features
                - adjacency: (num_nodes, num_nodes) adjacency matrix
                - edge_features: Optional (num_nodes, num_nodes, edge_dim)
                - node_mask: (num_nodes,) optional mask for valid nodes
            state: Per-element state (passed through).
            metadata: Optional metadata.
            random_params: Unused random parameters.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of:
                - data with "fingerprint" and "attention_weights" keys
                - unchanged state
                - unchanged metadata
        """
        del random_params, stats  # Unused

        node_features = data["node_features"]
        adjacency = data["adjacency"]
        edge_features = data.get("edge_features")
        node_mask = data.get("node_mask")

        # Initial projection
        h = self.input_proj(node_features)  # (N, hidden_dim)

        if self.dropout is not None:
            h = self.dropout(h)

        # Collect attention weights for interpretability
        all_attention_weights = []

        # Atom-level message passing with GRU
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            # Graph attention convolution
            h_new, attn_weights = conv(h, adjacency, edge_features)
            all_attention_weights.append(attn_weights)

            if self.dropout is not None:
                h_new = self.dropout(h_new)

            # GRU update: h_new is input, h is hidden state
            # GRUCell returns (new_carry, output) tuple - we use new_carry as next h
            h, _ = gru(h, h_new)

        # Apply node mask
        if node_mask is not None:
            h = h * node_mask[:, None]

        # Molecule-level aggregation with attention
        # Compute attention scores for global pooling
        attn_scores = self.mol_attn(h).squeeze(-1)  # (N,)
        if node_mask is not None:
            attn_scores = jnp.where(node_mask > 0, attn_scores, -1e9)
        attn_probs = jax.nn.softmax(attn_scores)  # (N,)

        # Weighted sum for initial molecule representation
        mol_repr = jnp.einsum("n,nd->d", attn_probs, h)  # (hidden_dim,)

        # GRU refinement at molecule level
        for mol_gru in self.mol_grus:
            # Use atom representations as context
            context = jnp.einsum("n,nd->d", attn_probs, h)
            # GRUCell returns (new_carry, output) tuple
            mol_repr, _ = mol_gru(mol_repr, context)

        # Final projection
        fingerprint = self.output_proj(mol_repr)

        if self.dropout is not None:
            fingerprint = self.dropout(fingerprint)

        result = {
            **data,
            "fingerprint": fingerprint,
            "attention_weights": all_attention_weights,
            "molecule_attention": attn_probs,
        }

        return result, state, metadata


def create_attentive_fp(
    hidden_dim: int = 200,
    out_dim: int = 200,
    num_layers: int = 2,
    num_timesteps: int = 2,
    dropout_rate: float = 0.0,
    seed: int = 42,
) -> AttentiveFP:
    """Create an AttentiveFP operator.

    Args:
        hidden_dim: Hidden dimension for GNN layers.
        out_dim: Output fingerprint dimension.
        num_layers: Number of atom-level attention layers.
        num_timesteps: Number of molecule-level GRU iterations.
        dropout_rate: Dropout rate.
        seed: Random seed.

    Returns:
        Configured AttentiveFP.
    """
    config = AttentiveFPConfig(
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout_rate=dropout_rate,
    )
    return AttentiveFP(config, rngs=nnx.Rngs(seed))
