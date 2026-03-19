"""Spatial domain identification and slice alignment operators.

This module provides two complementary operators for spatial transcriptomics:

1. **DifferentiableSpatialDomain** -- STAGATE-inspired graph attention autoencoder
   that identifies spatial domains by combining gene expression with spatial
   coordinates. Uses GATv2 attention with dual-graph pruning (alpha-weighted
   combination of full and pruned adjacency) for encoding, followed by soft
   domain assignment via learned prototypes.

2. **DifferentiablePASTEAlignment** -- PASTE-inspired fused Gromov-Wasserstein
   optimal transport for aligning two spatial transcriptomics slices. Balances
   expression dissimilarity with spatial structure preservation via entropy-
   regularised Sinkhorn transport.

Key techniques:
- GATv2 graph attention on spatial k-NN graphs (STAGATE)
- Autoencoder reconstruction loss for representation learning (STAGATE)
- Fused expression + spatial Gromov-Wasserstein cost (PASTE)
- Sinkhorn optimal transport for differentiable alignment (PASTE)

References:
- Dong & Zhang, "STAGATE: Deciphering spatial domains from spatially resolved
  transcriptomics with graph attention auto-encoder", Nature Communications 2022.
- Zeira et al., "Alignment and integration of spatial transcriptomics data",
  Nature Methods 2022.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.constants import DISTANCE_MASK_SENTINEL, EPSILON
from diffbio.core.base_operators import GraphOperator
from diffbio.core.gnn_components import GATv2Layer
from diffbio.core.graph_utils import compute_knn_graph, compute_pairwise_distances
from diffbio.core.optimal_transport import SinkhornLayer
from diffbio.utils.nn_utils import ensure_rngs


# =============================================================================
# STAGATE-inspired spatial domain identification
# =============================================================================


@dataclass
class SpatialDomainConfig(OperatorConfig):
    """Configuration for STAGATE-style spatial domain identification.

    Attributes:
        n_genes: Number of input genes.
        hidden_dim: Latent embedding dimension. Must be divisible by num_heads.
        num_heads: Number of GATv2 attention heads.
        n_domains: Number of spatial domains to identify.
        alpha: Weight for pruned graph in dual-graph attention (STAGATE default 0.8).
            At alpha=0, only the full k-NN graph is used. At alpha=1, only the
            pruned (mutual k-NN) graph is used.
        n_neighbors: Number of nearest neighbors for spatial k-NN graph.
    """

    n_genes: int = 2000
    hidden_dim: int = 64
    num_heads: int = 4
    n_domains: int = 7
    alpha: float = 0.8
    n_neighbors: int = 15


class _SpatialGATEncoder(nnx.Module):
    """GATv2-based encoder for spatial transcriptomics (STAGATE-style).

    Encodes gene expression using dual-graph attention: a combination of
    attention on the full k-NN graph and a pruned (mutual k-NN) graph,
    weighted by alpha. This follows the STAGATE architecture where the
    pruned graph encourages attention to spatially similar neighbors.

    Args:
        n_genes: Input gene expression dimension.
        hidden_dim: Output embedding dimension.
        num_heads: Number of GATv2 attention heads.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the spatial GAT encoder.

        Args:
            n_genes: Input feature dimension.
            hidden_dim: Hidden / output dimension.
            num_heads: Number of attention heads.
            rngs: Random number generators.
        """
        super().__init__()

        # Input projection: n_genes -> hidden_dim
        self.input_proj = nnx.Linear(
            in_features=n_genes,
            out_features=hidden_dim,
            rngs=rngs,
        )

        # GATv2 layer for full k-NN graph
        self.gat_full = GATv2Layer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_heads=num_heads,
            edge_features=1,
            dropout_rate=0.0,
            rngs=rngs,
        )

        # GATv2 layer for pruned (mutual k-NN) graph
        self.gat_pruned = GATv2Layer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_heads=num_heads,
            edge_features=1,
            dropout_rate=0.0,
            rngs=rngs,
        )

        self.layer_norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)

    def __call__(
        self,
        node_features: Float[Array, "n_cells n_genes"],
        full_edge_index: Int[Array, "2 n_full_edges"],
        full_edge_weights: Float[Array, "n_full_edges 1"],
        pruned_edge_index: Int[Array, "2 n_pruned_edges"],
        pruned_edge_weights: Float[Array, "n_pruned_edges 1"],
        alpha: float,
    ) -> Float[Array, "n_cells hidden_dim"]:
        """Encode gene expression using dual-graph GATv2 attention.

        Args:
            node_features: Gene expression matrix.
            full_edge_index: Edge indices for full k-NN graph (2, n_full_edges).
            full_edge_weights: Edge weights for full graph.
            pruned_edge_index: Edge indices for pruned mutual k-NN graph.
            pruned_edge_weights: Edge weights for pruned graph.
            alpha: Weight for pruned graph (0 = full only, 1 = pruned only).

        Returns:
            Spatial embeddings of shape (n_cells, hidden_dim).
        """
        # Project input to hidden dim
        h = self.input_proj(node_features)

        # Dual-graph attention (STAGATE: (1-alpha)*full + alpha*pruned)
        h_full = self.gat_full(h, full_edge_index, full_edge_weights, deterministic=True)
        h_pruned = self.gat_pruned(h, pruned_edge_index, pruned_edge_weights, deterministic=True)

        h = (1.0 - alpha) * h_full + alpha * h_pruned

        # Apply ELU activation (following STAGATE) + LayerNorm
        h = nnx.elu(h)
        h = self.layer_norm(h)

        return h


class _ExpressionDecoder(nnx.Module):
    """Decoder that reconstructs gene expression from spatial embeddings.

    Simple linear decoder mirroring the STAGATE architecture where the
    decoder uses transposed weights for reconstruction.

    Args:
        hidden_dim: Input embedding dimension.
        n_genes: Output gene expression dimension.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_genes: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the expression decoder.

        Args:
            hidden_dim: Embedding dimension.
            n_genes: Number of output genes.
            rngs: Random number generators.
        """
        super().__init__()
        self.linear = nnx.Linear(
            in_features=hidden_dim,
            out_features=n_genes,
            rngs=rngs,
        )

    def __call__(
        self,
        embeddings: Float[Array, "n_cells hidden_dim"],
    ) -> Float[Array, "n_cells n_genes"]:
        """Reconstruct gene expression from embeddings.

        Args:
            embeddings: Spatial embeddings.

        Returns:
            Reconstructed gene expression.
        """
        return self.linear(embeddings)


class DifferentiableSpatialDomain(GraphOperator):
    """STAGATE-inspired differentiable spatial domain identification.

    Identifies spatial domains by combining gene expression with spatial
    coordinates through a graph attention autoencoder. The encoder uses
    dual-graph GATv2 attention (full + pruned k-NN graphs), and soft domain
    assignments are computed via learned prototypes with softmax.

    Algorithm:
        1. Build spatial k-NN graph from coordinates (full + pruned/mutual).
        2. Apply GATv2 encoder: counts -> spatial embeddings (dual-graph
           attention weighted by alpha).
        3. Decoder: reconstruct gene expression from embeddings (autoencoder).
        4. Soft domain assignment via softmax on learned domain prototypes.

    Inherits from GraphOperator to get:

    - scatter_aggregate() for message aggregation
    - global_pool() for graph-level pooling

    Args:
        config: SpatialDomainConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.
    """

    def __init__(
        self,
        config: SpatialDomainConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the spatial domain identification operator.

        Args:
            config: Spatial domain configuration.
            rngs: Random number generators for parameter initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        self.alpha = config.alpha
        self.n_neighbors = config.n_neighbors
        self.n_domains = config.n_domains
        self.hidden_dim = config.hidden_dim

        # Encoder: GATv2-based spatial graph attention
        self.encoder = _SpatialGATEncoder(
            n_genes=config.n_genes,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            rngs=rngs,
        )

        # Decoder: reconstruct gene expression
        self.decoder = _ExpressionDecoder(
            hidden_dim=config.hidden_dim,
            n_genes=config.n_genes,
            rngs=rngs,
        )

        # Domain prototypes for soft assignment
        key = rngs.params()
        init_prototypes = jax.random.normal(key, (config.n_domains, config.hidden_dim)) * 0.1
        self.domain_prototypes = nnx.Param(init_prototypes)

    def _build_spatial_graphs(
        self,
        spatial_coords: Float[Array, "n_cells 2"],
    ) -> tuple[
        Int[Array, "2 n_full_edges"],
        Float[Array, "n_full_edges 1"],
        Int[Array, "2 n_pruned_edges"],
        Float[Array, "n_pruned_edges 1"],
    ]:
        """Build full and pruned (mutual) k-NN graphs from spatial coordinates.

        The full graph connects each cell to its k nearest spatial neighbors.
        The pruned graph keeps only mutual neighbors (edges present in both
        directions), following the STAGATE strategy.

        Args:
            spatial_coords: Spatial coordinates of shape (n_cells, 2).

        Returns:
            Tuple of (full_edge_index, full_edge_weights,
            pruned_edge_index, pruned_edge_weights).
        """
        n_cells = spatial_coords.shape[0]

        # Compute pairwise spatial distances
        distances = compute_pairwise_distances(spatial_coords, metric="euclidean")
        # Mask self-distances
        distances = distances + jnp.eye(n_cells) * DISTANCE_MASK_SENTINEL

        # Full k-NN graph
        edge_indices, edge_weights = compute_knn_graph(distances, k=self.n_neighbors)
        # edge_indices: (n_edges, 2), edge_weights: (n_edges,)
        full_edge_index = edge_indices.T  # (2, n_edges)
        full_edge_weights = edge_weights[:, None]  # (n_edges, 1)

        # Pruned (mutual) k-NN graph: keep only mutual edges
        # Build adjacency indicator for fast mutual check
        adj_indicator = jnp.zeros((n_cells, n_cells))
        adj_indicator = adj_indicator.at[edge_indices[:, 0], edge_indices[:, 1]].set(1.0)

        # Edge is mutual if both (i,j) and (j,i) are in the k-NN graph
        mutual_mask = (
            adj_indicator[edge_indices[:, 0], edge_indices[:, 1]]
            * adj_indicator[edge_indices[:, 1], edge_indices[:, 0]]
        )

        # Use the same edges but weight by mutual membership
        # (soft pruning for differentiability)
        pruned_edge_index = full_edge_index
        pruned_edge_weights = full_edge_weights * mutual_mask[:, None]

        return full_edge_index, full_edge_weights, pruned_edge_index, pruned_edge_weights

    def _compute_domain_assignments(
        self,
        embeddings: Float[Array, "n_cells hidden_dim"],
    ) -> Float[Array, "n_cells n_domains"]:
        """Compute soft domain assignments via learned prototypes.

        Distance from each cell embedding to each domain prototype is computed,
        then converted to assignment probabilities via softmax over negative
        squared distances.

        Args:
            embeddings: Spatial embeddings.

        Returns:
            Soft domain assignment probabilities.
        """
        prototypes = self.domain_prototypes[...]  # (n_domains, hidden_dim)

        # Squared distances: ||embedding - prototype||^2
        # Using expansion: ||e||^2 + ||p||^2 - 2*e.p
        emb_sq = jnp.sum(embeddings**2, axis=-1, keepdims=True)  # (n, 1)
        proto_sq = jnp.sum(prototypes**2, axis=-1)  # (d,)
        dot = jnp.einsum("nf,df->nd", embeddings, prototypes)  # (n, d)
        distances_sq = emb_sq + proto_sq - 2.0 * dot

        # Soft assignment via softmax over negative distances
        return jax.nn.softmax(-distances_sq, axis=-1)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply spatial domain identification to spatial transcriptomics data.

        Args:
            data: Dictionary containing:
                - ``"counts"``: Gene expression matrix ``(n_cells, n_genes)``
                - ``"spatial_coords"``: Spatial coordinates ``(n_cells, 2)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used (non-stochastic operator).
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains all original keys plus:

                    - ``"domain_assignments"``: Soft domain probabilities
                      ``(n_cells, n_domains)``
                    - ``"spatial_embeddings"``: Latent embeddings
                      ``(n_cells, hidden_dim)``
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts: Float[Array, "n_cells n_genes"] = data["counts"]
        spatial_coords: Float[Array, "n_cells 2"] = data["spatial_coords"]

        # Step 1: Build spatial k-NN graphs (full + pruned)
        (
            full_edge_index,
            full_edge_weights,
            pruned_edge_index,
            pruned_edge_weights,
        ) = self._build_spatial_graphs(spatial_coords)

        # Step 2: Encode via dual-graph GATv2 attention
        embeddings = self.encoder(
            counts,
            full_edge_index,
            full_edge_weights,
            pruned_edge_index,
            pruned_edge_weights,
            self.alpha,
        )

        # Step 3: Decode (autoencoder reconstruction -- loss can be computed externally)
        _reconstructed = self.decoder(embeddings)  # noqa: F841

        # Step 4: Soft domain assignment via prototypes
        domain_assignments = self._compute_domain_assignments(embeddings)

        transformed_data = {
            **data,
            "domain_assignments": domain_assignments,
            "spatial_embeddings": embeddings,
        }

        return transformed_data, state, metadata


# =============================================================================
# PASTE-inspired slice alignment
# =============================================================================


@dataclass
class PASTEAlignmentConfig(OperatorConfig):
    """Configuration for PASTE-style spatial transcriptomics slice alignment.

    Attributes:
        alpha: Balance between expression dissimilarity (linear term) and
            spatial Gromov-Wasserstein cost (quadratic term). 0 = pure expression
            matching, 1 = pure spatial structure matching. PASTE default: 0.1.
        sinkhorn_epsilon: Entropy regularisation strength for the Sinkhorn
            optimal transport solver.
        sinkhorn_iters: Number of Sinkhorn iterations.
    """

    alpha: float = 0.1
    sinkhorn_epsilon: float = 0.1
    sinkhorn_iters: int = 100


class DifferentiablePASTEAlignment(GraphOperator):
    """PASTE-inspired differentiable spatial transcriptomics slice alignment.

    Aligns two spatial transcriptomics slices by computing a fused cost that
    balances expression dissimilarity with spatial structure (Gromov-Wasserstein)
    and solving for the optimal transport plan via differentiable Sinkhorn.

    Algorithm:
        1. Compute expression dissimilarity between slices (Euclidean distance).
        2. Compute intra-slice spatial distance matrices.
        3. Compute Gromov-Wasserstein spatial cost that penalizes distortion
           of pairwise spatial relationships.
        4. Fuse costs: alpha * expression_cost + (1 - alpha) * spatial_GW_cost.
        5. Solve OT via SinkhornLayer for the differentiable transport plan.
        6. Align slice 2 coordinates using the transport plan.

    Inherits from GraphOperator to get:

    - scatter_aggregate() for message aggregation
    - global_pool() for graph-level pooling

    Args:
        config: PASTEAlignmentConfig with alignment parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.
    """

    def __init__(
        self,
        config: PASTEAlignmentConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the PASTE alignment operator.

        Args:
            config: PASTE alignment configuration.
            rngs: Random number generators.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        self.alpha_cost = config.alpha
        self.sinkhorn = SinkhornLayer(
            epsilon=config.sinkhorn_epsilon,
            num_iters=config.sinkhorn_iters,
            rngs=rngs,
        )

    def _compute_expression_cost(
        self,
        counts1: Float[Array, "n1 g"],
        counts2: Float[Array, "n2 g"],
    ) -> Float[Array, "n1 n2"]:
        """Compute pairwise expression dissimilarity between two slices.

        Uses squared Euclidean distance, normalized by number of genes
        for numerical stability.

        Args:
            counts1: Expression matrix for slice 1.
            counts2: Expression matrix for slice 2.

        Returns:
            Expression cost matrix of shape (n1, n2).
        """
        n_genes = counts1.shape[1]
        # ||c1_i - c2_j||^2 = ||c1_i||^2 + ||c2_j||^2 - 2 * c1_i . c2_j
        sq1 = jnp.sum(counts1**2, axis=-1, keepdims=True)  # (n1, 1)
        sq2 = jnp.sum(counts2**2, axis=-1)  # (n2,)
        dot = jnp.dot(counts1, counts2.T)  # (n1, n2)
        cost = sq1 + sq2 - 2.0 * dot
        # Normalize by number of genes for stability
        return jnp.maximum(cost, 0.0) / (n_genes + EPSILON)

    def _compute_spatial_distances(
        self,
        coords: Float[Array, "n 2"],
    ) -> Float[Array, "n n"]:
        """Compute intra-slice pairwise spatial distance matrix.

        Args:
            coords: Spatial coordinates of shape (n, 2).

        Returns:
            Distance matrix of shape (n, n).
        """
        return compute_pairwise_distances(coords, metric="euclidean")

    def _compute_gromov_wasserstein_cost(
        self,
        dist_a: Float[Array, "n1 n1"],
        dist_b: Float[Array, "n2 n2"],
        transport_plan: Float[Array, "n1 n2"],
    ) -> Float[Array, "n1 n2"]:
        """Compute the Gromov-Wasserstein gradient term for the fused cost.

        The GW cost measures how well the transport plan preserves pairwise
        spatial relationships::

            L(D_A, D_B, T) = sum |D_A[i,k] - D_B[j,l]|^2 * T[i,j] * T[k,l]

        This computes the gradient of the GW cost with respect to T, which
        gives the linear cost matrix for the next Sinkhorn iteration.

        Args:
            dist_a: Spatial distance matrix for slice 1.
            dist_b: Spatial distance matrix for slice 2.
            transport_plan: Current transport plan estimate.

        Returns:
            GW cost gradient matrix of shape (n1, n2).
        """
        # Square loss GW: sum_{ijkl} (D_A[i,k] - D_B[j,l])^2 * T[k,l]
        # Gradient w.r.t. T[i,j] = 2 * (D_A^2 @ T @ 1 + 1 @ T @ D_B^2 - 2 * D_A @ T @ D_B)
        # Simplified constant parts + linear in T:
        da_sq = dist_a**2
        db_sq = dist_b**2

        # Term 1: D_A^2 @ T @ ones_n2 (broadcast) -> (n1, n2) contribution
        term1 = da_sq @ transport_plan  # (n1, n2)

        # Term 2: ones_n1^T @ T @ D_B^2 (broadcast) -> (n1, n2) contribution
        term2 = transport_plan @ db_sq  # (n1, n2)

        # Term 3: D_A @ T @ D_B^T (cross term)
        cross = dist_a @ transport_plan @ dist_b  # (n1, n2)

        gw_cost = term1 + term2 - 2.0 * cross
        return gw_cost

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply PASTE-style alignment between two spatial transcriptomics slices.

        Args:
            data: Dictionary containing:
                - ``"slice1_counts"``: Expression matrix for slice 1 ``(n1, g)``
                - ``"slice2_counts"``: Expression matrix for slice 2 ``(n2, g)``
                - ``"slice1_coords"``: Spatial coordinates for slice 1 ``(n1, 2)``
                - ``"slice2_coords"``: Spatial coordinates for slice 2 ``(n2, 2)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used (non-stochastic operator).
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains all original keys plus:

                    - ``"transport_plan"``: OT plan ``(n1, n2)``
                    - ``"aligned_coords"``: Aligned slice 2 coordinates ``(n2, 2)``
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts1: Float[Array, "n1 g"] = data["slice1_counts"]
        counts2: Float[Array, "n2 g"] = data["slice2_counts"]
        coords1: Float[Array, "n1 2"] = data["slice1_coords"]
        coords2: Float[Array, "n2 2"] = data["slice2_coords"]

        n1 = counts1.shape[0]
        n2 = counts2.shape[0]

        # Step 1: Expression dissimilarity cost
        expression_cost = self._compute_expression_cost(counts1, counts2)

        # Step 2: Intra-slice spatial distances
        dist_a = self._compute_spatial_distances(coords1)
        dist_b = self._compute_spatial_distances(coords2)

        # Step 3: Initial transport plan (uniform) for GW cost estimation
        init_plan = jnp.ones((n1, n2)) / (n1 * n2)

        # Step 4: Compute Gromov-Wasserstein spatial cost
        gw_cost = self._compute_gromov_wasserstein_cost(dist_a, dist_b, init_plan)

        # Normalize costs to comparable scales
        expr_max = jnp.max(expression_cost) + EPSILON
        gw_max = jnp.max(gw_cost) + EPSILON
        expression_cost_norm = expression_cost / expr_max
        gw_cost_norm = gw_cost / gw_max

        # Step 5: Fused cost = (1-alpha) * expression + alpha * GW_spatial
        fused_cost = (1.0 - self.alpha_cost) * expression_cost_norm + self.alpha_cost * gw_cost_norm

        # Step 6: Solve OT via Sinkhorn
        a = jnp.ones(n1) / n1  # uniform source marginal
        b = jnp.ones(n2) / n2  # uniform target marginal
        transport_plan = self.sinkhorn(fused_cost, a, b)

        # Step 7: Align slice 2 coordinates using the transport plan
        # Normalized plan rows: T_norm[i, :] = T[i, :] / sum_j T[i, j]
        plan_col_normalized = transport_plan / (
            jnp.sum(transport_plan, axis=0, keepdims=True) + EPSILON
        )
        # aligned_coords[j] = sum_i plan_col_norm[i, j] * coords1[i]
        aligned_coords = plan_col_normalized.T @ coords1

        transformed_data = {
            **data,
            "transport_plan": transport_plan,
            "aligned_coords": aligned_coords,
        }

        return transformed_data, state, metadata
