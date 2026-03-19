"""Differentiable cell-cell communication analysis.

This module provides two complementary operators for analysing cell-cell
communication in single-cell data:

1. **DifferentiableLigandReceptor** -- ligand-receptor co-expression scoring
   using fuzzy k-NN adjacency graphs and analytical z-score significance.
2. **DifferentiableCellCommunication** -- GNN-based communication analysis
   using GATv2 graph attention on a spatial cell graph with per-edge
   L-R expression features.

Key techniques:
- Soft adjacency weighting via fuzzy k-NN (L-R scoring)
- GATv2 message passing on spatial cell graphs (cell communication)
- Temperature-controlled smooth approximations throughout

Applications: CellChat/CellPhoneDB-style communication analysis, spatial
transcriptomics niche identification, pathway-level signaling inference.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.constants import DISTANCE_MASK_SENTINEL, EPSILON
from diffbio.core.base_operators import GraphOperator, TemperatureOperator
from diffbio.core.gnn_components import GATv2Layer
from diffbio.core.graph_utils import (
    compute_fuzzy_membership,
    compute_pairwise_distances,
    symmetrize_graph,
)
from diffbio.utils.nn_utils import ensure_rngs


@dataclass
class LRScoringConfig(OperatorConfig):
    """Configuration for ligand-receptor co-expression scoring.

    Attributes:
        n_neighbors: Number of nearest neighbors for k-NN graph.
        temperature: Temperature for soft p-value sigmoid.
        learnable_temperature: Whether the temperature is a learnable parameter.
        metric: Distance metric for k-NN, either ``"euclidean"`` or ``"cosine"``.
        kh: Hill function half-maximal constant (CellChat default 0.5).
        hill_n: Hill function cooperativity coefficient (CellChat default 1.0).
    """

    n_neighbors: int = 15
    temperature: float = 1.0
    learnable_temperature: bool = False
    metric: str = "euclidean"
    kh: float = 0.5
    hill_n: float = 1.0


class DifferentiableLigandReceptor(TemperatureOperator):
    """Differentiable ligand-receptor co-expression scoring operator.

    Scores cell-cell communication by computing adjacency-weighted co-expression
    of ligand-receptor gene pairs. For each pair, the score at each receiver
    cell is the sum of sender ligand expression times receiver receptor
    expression, weighted by a fuzzy k-NN adjacency graph.

    Algorithm:
        1. Build a symmetric fuzzy k-NN adjacency from the count matrix using
           ``compute_pairwise_distances``, ``compute_fuzzy_membership``, and
           ``symmetrize_graph``.
        2. For each L-R pair (ligand_idx, receptor_idx):
           - ``score_i = sum_j(adjacency[i,j] * L[j] * R[i])``
           where L[j] is the sender's ligand expression and R[i] is the
           receiver's receptor expression.
        3. Compute analytical soft p-values via z-score comparison against
           an expected null distribution.

    Inherits from TemperatureOperator to get:

    - _temperature property for temperature-controlled smoothing
    - soft_max() for logsumexp-based smooth maximum
    - soft_argmax() for soft position selection

    Args:
        config: LRScoringConfig with operator parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = LRScoringConfig(n_neighbors=15)
        >>> op = DifferentiableLigandReceptor(config, rngs=nnx.Rngs(0))
        >>> data = {"counts": counts, "lr_pairs": jnp.array([[0, 1]])}
        >>> result, state, meta = op.apply(data, {}, None)
        >>> result["lr_scores"].shape
        (n_cells, 1)
    """

    def __init__(
        self,
        config: LRScoringConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the ligand-receptor scoring operator.

        Args:
            config: L-R scoring configuration.
            rngs: Random number generators for parameter initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

    def _build_adjacency(
        self,
        counts: Float[Array, "n_cells n_genes"],
    ) -> Float[Array, "n_cells n_cells"]:
        """Build symmetric fuzzy k-NN adjacency from expression counts.

        Args:
            counts: Gene expression matrix.

        Returns:
            Symmetric adjacency matrix of shape ``(n_cells, n_cells)``.
        """
        # Mask self-distances with large sentinel before k-NN computation
        n_cells = counts.shape[0]
        distances = compute_pairwise_distances(counts, metric=self.config.metric)
        distances = distances + jnp.eye(n_cells) * DISTANCE_MASK_SENTINEL

        membership = compute_fuzzy_membership(distances, k=self.config.n_neighbors)
        return symmetrize_graph(membership)

    def _score_lr_pair(
        self,
        adjacency: Float[Array, "n_cells n_cells"],
        ligand_expression: Float[Array, "n_cells"],
        receptor_expression: Float[Array, "n_cells"],
    ) -> Float[Array, "n_cells"]:
        """Score L-R pair using Hill function (CellChat-style).

        For each receiver cell *i* the score is a saturating Hill function
        of the neighbor-averaged ligand signal times the receiver's own
        receptor expression:

        ``P = (L*R)^n / (Kh^n + (L*R)^n)``

        where ``Kh`` and ``n`` are configured via ``LRScoringConfig.kh``
        and ``LRScoringConfig.hill_n``.

        Args:
            adjacency: Symmetric adjacency matrix.
            ligand_expression: Ligand gene expression per cell.
            receptor_expression: Receptor gene expression per cell.

        Returns:
            Per-cell interaction score of shape ``(n_cells,)``.
        """
        # Neighbor-averaged ligand expression (sender perspective)
        neighbor_ligand = adjacency @ ligand_expression  # (n_cells,)

        # L-R product per cell (receiver's receptor * sender's ligand)
        lr_product = neighbor_ligand * receptor_expression

        # Hill function for saturation (CellChat: Kh=0.5, n=1)
        kh: float = self.config.kh
        n: float = self.config.hill_n
        score = lr_product**n / (kh**n + lr_product**n + EPSILON)

        return score

    def _compute_soft_pvalues(
        self,
        scores: Float[Array, "n_cells n_pairs"],
        adjacency: Float[Array, "n_cells n_cells"],
        counts: Float[Array, "n_cells n_genes"],
        lr_pairs: Int[Array, "n_pairs 2"],
    ) -> Float[Array, "n_pairs"]:
        """Compute soft p-values for each L-R pair via analytical z-score.

        Under a null where adjacency is independent of expression, the expected
        score for each cell is ``E[score_i] = mean(L) * R[i] * sum_j(A[i,j])``.
        The aggregate z-score is computed from the total observed vs. expected
        score, and converted to a soft p-value via sigmoid.

        Args:
            scores: Per-cell L-R scores of shape ``(n_cells, n_pairs)``.
            adjacency: Adjacency matrix.
            counts: Expression matrix for computing means.
            lr_pairs: L-R pair indices.

        Returns:
            Soft p-values per pair, shape ``(n_pairs,)``.
        """
        row_sums = jnp.sum(adjacency, axis=1)  # (n_cells,)
        temperature = self._temperature

        def _pvalue_for_pair(pair_idx: Int[Array, ""]) -> Float[Array, ""]:
            ligand_idx = lr_pairs[pair_idx, 0]
            receptor_idx = lr_pairs[pair_idx, 1]

            ligand_expr = counts[:, ligand_idx]
            receptor_expr = counts[:, receptor_idx]

            # Observed total score
            observed = jnp.sum(scores[:, pair_idx])

            # Expected under null: E[score_i] = mean(L) * R[i] * degree_i
            mean_ligand = jnp.mean(ligand_expr)
            mean_receptor = jnp.mean(receptor_expr)
            expected = jnp.sum(mean_ligand * receptor_expr * row_sums)

            # Correct variance for Var(sum_j A[i,j]*L[j]*R[i]) under
            # independence of L and R:
            #   Var(sum) = sum(A^2) * (Var(L)*Var(R)
            #              + Var(L)*E[R]^2 + Var(R)*E[L]^2)
            var_ligand = jnp.var(ligand_expr) + EPSILON
            var_receptor = jnp.var(receptor_expr) + EPSILON
            sum_adj_sq = jnp.sum(adjacency**2)
            std_approx = jnp.sqrt(
                sum_adj_sq
                * (
                    var_ligand * var_receptor
                    + var_ligand * mean_receptor**2
                    + var_receptor * mean_ligand**2
                )
                + EPSILON
            )

            z_score = (observed - expected) / (std_approx + EPSILON)

            # Soft p-value: high z-score -> low p-value
            return 1.0 - jax.nn.sigmoid(z_score / temperature)

        n_pairs = lr_pairs.shape[0]
        pair_indices = jnp.arange(n_pairs)
        return jax.vmap(_pvalue_for_pair)(pair_indices)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply ligand-receptor co-expression scoring.

        Args:
            data: Dictionary containing:
                - ``"counts"``: Gene expression matrix ``(n_cells, n_genes)``
                - ``"lr_pairs"``: L-R pair indices ``(n_pairs, 2)`` where each
                  row is ``[ligand_gene_idx, receptor_gene_idx]``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used (non-stochastic operator).
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - all original data keys
                    - ``"lr_scores"``: Per-cell interaction scores ``(n_cells, n_pairs)``
                    - ``"lr_pvalues"``: Soft p-values per pair ``(n_pairs,)``
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts = data["counts"]
        lr_pairs = data["lr_pairs"]

        # Step 1: Build k-NN adjacency graph
        adjacency = self._build_adjacency(counts)

        # Step 2: Score each L-R pair across all cells
        def _score_pair(pair: Int[Array, "2"]) -> Float[Array, "n_cells"]:
            ligand_expr = counts[:, pair[0]]
            receptor_expr = counts[:, pair[1]]
            return self._score_lr_pair(adjacency, ligand_expr, receptor_expr)

        lr_scores = jax.vmap(_score_pair)(lr_pairs).T  # (n_cells, n_pairs)

        # Step 3: Compute soft p-values
        lr_pvalues = self._compute_soft_pvalues(lr_scores, adjacency, counts, lr_pairs)

        transformed_data = {
            **data,
            "lr_scores": lr_scores,
            "lr_pvalues": lr_pvalues,
        }

        return transformed_data, state, metadata


# =============================================================================
# GNN-based cell-cell communication
# =============================================================================


@dataclass
class CellCommunicationConfig(OperatorConfig):
    """Configuration for GNN-based cell-cell communication analysis.

    Attributes:
        n_genes: Number of genes in the expression matrix.
        n_lr_pairs: Number of ligand-receptor pairs (determines edge feature
            projection input dimension and communication score output width).
        hidden_dim: Hidden dimension for GNN layers (must be divisible by num_heads).
        num_heads: Number of attention heads in GATv2 layers.
        edge_features_dim: Dimension of per-edge L-R features fed to GATv2.
        num_gnn_layers: Number of stacked GATv2 layers.
        n_pathways: Number of signaling pathways to infer.
        dropout_rate: Dropout rate for regularization.
    """

    n_genes: int = 2000
    n_lr_pairs: int = 10
    hidden_dim: int = 64
    num_heads: int = 4
    edge_features_dim: int = 8
    num_gnn_layers: int = 2
    n_pathways: int = 20
    dropout_rate: float = 0.1


class SpatialAttentionGNN(nnx.Module):
    """Stacked GATv2 layers with residual connections for spatial cell graphs.

    Each layer applies GATv2 attention followed by a LayerNorm and residual
    connection.  An input projection maps node features to the hidden dimension
    before the first GATv2 layer.

    Args:
        in_features: Dimension of input node features.
        hidden_dim: Hidden dimension (must be divisible by num_heads).
        num_heads: Number of attention heads per GATv2 layer.
        edge_features_dim: Edge feature dimension.
        num_layers: Number of GATv2 layers.
        dropout_rate: Dropout rate.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_heads: int,
        edge_features_dim: int,
        num_layers: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the spatial attention GNN.

        Args:
            in_features: Input feature dimension.
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            edge_features_dim: Edge feature dimension.
            num_layers: Number of GATv2 layers.
            dropout_rate: Dropout rate.
            rngs: Random number generators.
        """
        super().__init__()

        # Project input features to hidden dimension
        self.input_proj = nnx.Linear(
            in_features=in_features,
            out_features=hidden_dim,
            rngs=rngs,
        )

        self.gat_layers = nnx.List(
            [
                GATv2Layer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    edge_features=edge_features_dim,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norms = nnx.List(
            [nnx.LayerNorm(num_features=hidden_dim, rngs=rngs) for _ in range(num_layers)]
        )

    def __call__(
        self,
        node_features: Float[Array, "n_nodes in_features"],
        edge_index: Int[Array, "2 n_edges"],
        edge_features: Float[Array, "n_edges edge_features_dim"],
        *,
        deterministic: bool = True,
    ) -> Float[Array, "n_nodes hidden_dim"]:
        """Run stacked GATv2 attention with residual connections.

        Args:
            node_features: Input node features.
            edge_index: Edge indices ``(source, target)`` of shape ``(2, n_edges)``.
            edge_features: Per-edge features.
            deterministic: If True, disable dropout.

        Returns:
            Updated node embeddings of shape ``(n_nodes, hidden_dim)``.
        """
        x = self.input_proj(node_features)

        for gat_layer, norm in zip(self.gat_layers, self.layer_norms):
            residual = x
            x = gat_layer(x, edge_index, edge_features, deterministic=deterministic)
            x = norm(x + residual)

        return x


class SignalingDecoder(nnx.Module):
    """Map node embeddings to pathway activities and communication scores.

    Two heads:
    - **Pathway head**: ``Linear(hidden_dim, n_pathways)`` produces per-node
      pathway activity.
    - **Communication head**: ``Linear(hidden_dim, n_lr_pairs)`` produces
      per-node communication scores for each L-R pair.

    Args:
        hidden_dim: Input embedding dimension.
        n_pathways: Number of output pathways.
        n_lr_pairs: Number of L-R pairs (communication score outputs).
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_pathways: int,
        n_lr_pairs: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the signaling decoder.

        Args:
            hidden_dim: Hidden dimension.
            n_pathways: Number of signaling pathways.
            n_lr_pairs: Number of L-R pairs.
            rngs: Random number generators.
        """
        super().__init__()

        self.pathway_head = nnx.Linear(
            in_features=hidden_dim,
            out_features=n_pathways,
            rngs=rngs,
        )
        self.comm_head = nnx.Linear(
            in_features=hidden_dim,
            out_features=n_lr_pairs,
            rngs=rngs,
        )

    def __call__(
        self,
        node_embeddings: Float[Array, "n_nodes hidden_dim"],
    ) -> tuple[
        Float[Array, "n_nodes n_pathways"],
        Float[Array, "n_nodes n_lr_pairs"],
    ]:
        """Decode node embeddings into pathway activities and communication scores.

        Args:
            node_embeddings: Node embedding matrix.

        Returns:
            Tuple of (signaling_activity, communication_scores).
        """
        signaling_activity = self.pathway_head(node_embeddings)
        communication_scores = self.comm_head(node_embeddings)
        return signaling_activity, communication_scores


class DifferentiableCellCommunication(GraphOperator):
    """GNN-based differentiable cell-cell communication analysis.

    Analyses inter-cellular signaling by applying GATv2 graph attention on a
    spatial cell graph whose edges carry ligand-receptor expression features.

    Algorithm:
        1. Build per-edge L-R expression features from ``counts`` and
           ``lr_pairs``: for each edge (i, j) the feature vector is the
           concatenation of ``[L_expr[source], R_expr[target]]`` across all
           L-R pairs, projected to ``edge_features_dim``.
        2. Project per-node gene expression to initial node embeddings.
        3. Apply stacked GATv2 layers (``SpatialAttentionGNN``) for message
           passing on the spatial cell graph.
        4. Decode node embeddings into per-node pathway activity and per-node
           communication scores via ``SignalingDecoder``.

    Inherits from GraphOperator to get:

    - scatter_aggregate() for message aggregation
    - global_pool() for graph-level pooling

    Args:
        config: CellCommunicationConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = CellCommunicationConfig(n_genes=50, n_lr_pairs=3, hidden_dim=32)
        >>> op = DifferentiableCellCommunication(config, rngs=nnx.Rngs(0))
        >>> data = {"counts": counts, "spatial_graph": graph, "lr_pairs": pairs}
        >>> result, state, meta = op.apply(data, {}, None)
        >>> result["communication_scores"].shape
        (n_cells, 3)
    """

    def __init__(
        self,
        config: CellCommunicationConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the cell communication operator.

        Args:
            config: Cell communication configuration.
            rngs: Random number generators for parameter initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        self.hidden_dim = config.hidden_dim

        # Node feature projection: n_genes -> hidden_dim
        self.node_proj = nnx.Linear(
            in_features=config.n_genes,
            out_features=config.hidden_dim,
            rngs=rngs,
        )

        # Edge feature projection: raw LR features (2 * n_lr_pairs) -> edge_features_dim
        self.edge_proj = nnx.Linear(
            in_features=2 * config.n_lr_pairs,
            out_features=config.edge_features_dim,
            rngs=rngs,
        )

        # GATv2 stack
        self.spatial_gnn = SpatialAttentionGNN(
            in_features=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            edge_features_dim=config.edge_features_dim,
            num_layers=config.num_gnn_layers,
            dropout_rate=config.dropout_rate,
            rngs=rngs,
        )

        # Signaling decoder
        self.decoder = SignalingDecoder(
            hidden_dim=config.hidden_dim,
            n_pathways=config.n_pathways,
            n_lr_pairs=config.n_lr_pairs,
            rngs=rngs,
        )

    def _build_edge_features(
        self,
        counts: Float[Array, "n_cells n_genes"],
        spatial_graph: Int[Array, "2 n_edges"],
        lr_pairs: Int[Array, "n_pairs 2"],
    ) -> Float[Array, "n_edges edge_features_dim"]:
        """Compute per-edge L-R expression features.

        For each edge (i, j) and each L-R pair (l, r) the raw feature is
        ``[counts[source, l], counts[target, r]]``.  These are concatenated
        across all pairs then projected to ``edge_features_dim``.

        Args:
            counts: Gene expression matrix ``(n_cells, n_genes)``.
            spatial_graph: Edge indices ``(source, target)`` ``(2, n_edges)``.
            lr_pairs: L-R pair gene indices ``(n_pairs, 2)``.

        Returns:
            Projected edge features ``(n_edges, edge_features_dim)``.
        """
        sources = spatial_graph[0]  # (n_edges,)
        targets = spatial_graph[1]  # (n_edges,)

        # For each LR pair, gather ligand expression of source and receptor
        # expression of target.  Shape per pair: (n_edges, 2)
        def _pair_features(pair: Int[Array, "2"]) -> Float[Array, "n_edges 2"]:
            ligand_idx = pair[0]
            receptor_idx = pair[1]
            l_expr = counts[sources, ligand_idx]  # (n_edges,)
            r_expr = counts[targets, receptor_idx]  # (n_edges,)
            return jnp.stack([l_expr, r_expr], axis=-1)

        # (n_pairs, n_edges, 2)
        raw_features = jax.vmap(_pair_features)(lr_pairs)
        # Reshape to (n_edges, 2*n_pairs)
        n_edges = spatial_graph.shape[1]
        raw_features = raw_features.transpose(1, 0, 2).reshape(n_edges, -1)

        return self.edge_proj(raw_features)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply GNN-based cell-cell communication analysis.

        Args:
            data: Dictionary containing:
                - ``"counts"``: Gene expression matrix ``(n_cells, n_genes)``
                - ``"spatial_graph"``: Edge indices ``(2, n_edges)`` where
                  row 0 = source nodes, row 1 = target nodes
                - ``"lr_pairs"``: L-R pair gene indices ``(n_pairs, 2)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used (non-stochastic operator).
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains all original keys plus:

                    - ``"communication_scores"``: ``(n_cells, n_pairs)``
                    - ``"signaling_activity"``: ``(n_cells, n_pathways)``
                    - ``"niche_embeddings"``: ``(n_cells, hidden_dim)``
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts: Float[Array, "n_cells n_genes"] = data["counts"]
        spatial_graph: Int[Array, "2 n_edges"] = data["spatial_graph"]
        lr_pairs: Int[Array, "n_pairs 2"] = data["lr_pairs"]

        # Step 1: Build per-edge L-R features
        edge_features = self._build_edge_features(counts, spatial_graph, lr_pairs)

        # Step 2: Project per-node gene expression to hidden dim
        node_features = self.node_proj(counts)  # (n_cells, hidden_dim)

        # Step 3: GATv2 message passing
        niche_embeddings = self.spatial_gnn(
            node_features,
            spatial_graph,
            edge_features,
            deterministic=True,
        )

        # Step 4: Decode to pathway activities and communication scores
        signaling_activity, communication_scores = self.decoder(niche_embeddings)

        transformed_data = {
            **data,
            "communication_scores": communication_scores,
            "signaling_activity": signaling_activity,
            "niche_embeddings": niche_embeddings,
        }

        return transformed_data, state, metadata
