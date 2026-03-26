"""Differentiable gene regulatory network inference.

This module provides a differentiable alternative to GENIE3/SCENIC for gene
regulatory network (GRN) inference from single-cell expression data. Instead
of random forest feature importance, it uses GATv2 graph attention on a
TF-gene bipartite graph to learn regulatory strengths.

Key technique: attention weights on a dense bipartite graph between
transcription factors and target genes serve as a differentiable proxy for
regulatory importance scores. Soft L1 sparsity via sigmoid gating promotes
biologically realistic sparse networks.

Algorithm:
    1. Build a TF-gene bipartite graph (every TF connected to every gene).
    2. Compute per-edge features from expression: concatenation of TF
       expression, gene expression, and absolute expression difference.
    3. Apply GATv2 attention -- attention weights between TF-gene pairs
       represent regulatory strength.
    4. Extract attention weights as the GRN adjacency matrix.
    5. Apply soft L1 sparsity: ``grn * sigmoid(grn / temperature)``.
    6. Compute TF activity: ``counts[:, tf_indices] @ grn_matrix``.

Applications: SCENIC/GENIE3-style regulatory network reconstruction,
transcription factor activity estimation, regulon discovery.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.constants import EPSILON
from diffbio.core.gnn_components import GATv2Layer
from diffbio.utils.nn_utils import ensure_rngs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GRNInferenceConfig(OperatorConfig):
    """Configuration for differentiable GRN inference.

    Attributes:
        n_tfs: Number of transcription factors.
        n_genes: Number of genes in the expression matrix.
        hidden_dim: Hidden dimension for GATv2 attention (must be divisible
            by num_heads).
        num_heads: Number of attention heads in the GATv2 layer.
        sparsity_temperature: Temperature for soft L1 sparsity gating.
            Lower values produce sharper thresholding toward zero.
        sparsity_lambda: L1 regularization weight (used by downstream loss
            functions, not directly by the operator).
    """

    n_tfs: int = 50
    n_genes: int = 2000
    hidden_dim: int = 64
    num_heads: int = 4
    sparsity_temperature: float = 0.1
    sparsity_lambda: float = 0.01


class DifferentiableGRN(OperatorModule):
    """Differentiable gene regulatory network inference operator.

    Uses GATv2 graph attention on a TF-gene bipartite graph to infer
    regulatory strengths. Each TF is connected to every gene; the attention
    weight on each edge represents how strongly the TF regulates that gene.

    This is a novel differentiable alternative to GENIE3's random forest
    feature importance scoring. The key insight is that in GENIE3, each
    gene's expression is predicted from TF expression, and feature importance
    measures regulatory strength. Here, GATv2 attention performs an analogous
    role: TF nodes attend to gene nodes, and the learned attention weights
    capture regulatory relationships.

    Args:
        config: GRNInferenceConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = GRNInferenceConfig(n_tfs=5, n_genes=20, hidden_dim=16)
        >>> op = DifferentiableGRN(config, rngs=nnx.Rngs(0))
        >>> data = {"counts": counts, "tf_indices": jnp.arange(5)}
        >>> result, state, meta = op.apply(data, {}, None)
        >>> result["grn_matrix"].shape
        (5, 20)
    """

    def __init__(
        self,
        config: GRNInferenceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the GRN inference operator.

        Args:
            config: GRN inference configuration.
            rngs: Random number generators for parameter initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        self.n_tfs = config.n_tfs
        self.n_genes = config.n_genes
        self.hidden_dim = config.hidden_dim
        self.sparsity_temperature = config.sparsity_temperature

        # Project 1-d expression scalars to hidden_dim for each node
        self.node_proj = nnx.Linear(
            in_features=1,
            out_features=config.hidden_dim,
            rngs=rngs,
        )

        # GATv2 layer: attention on the bipartite graph
        # Edge features: [tf_expr, gene_expr, |tf_expr - gene_expr|] -> dim 3
        self.gat_layer = GATv2Layer(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim,
            num_heads=config.num_heads,
            edge_features=3,
            dropout_rate=0.0,
            rngs=rngs,
        )

        # Projection from GATv2 output to scalar regulatory score per edge
        self.score_proj = nnx.Linear(
            in_features=config.hidden_dim * 2,
            out_features=1,
            rngs=rngs,
        )

    def _build_bipartite_graph(
        self,
        n_tfs: int,
        n_genes: int,
    ) -> Int[Array, "2 n_edges"]:
        """Build dense bipartite edge index between TFs and genes.

        TF nodes are indexed ``[0, n_tfs)``, gene nodes are indexed
        ``[n_tfs, n_tfs + n_genes)``. Every TF is connected to every gene.

        Args:
            n_tfs: Number of transcription factors.
            n_genes: Number of genes.

        Returns:
            Edge index array of shape ``(2, n_tfs * n_genes)`` where row 0
            is source (TF) indices and row 1 is target (gene) indices.
        """
        # TF indices: 0..n_tfs-1, gene indices: n_tfs..n_tfs+n_genes-1
        tf_ids = jnp.arange(n_tfs)
        gene_ids = jnp.arange(n_tfs, n_tfs + n_genes)

        # Dense bipartite: every TF connected to every gene
        # sources: each TF repeated n_genes times
        sources = jnp.repeat(tf_ids, n_genes)
        # targets: gene_ids tiled n_tfs times
        targets = jnp.tile(gene_ids, n_tfs)

        return jnp.stack([sources, targets], axis=0)

    def _compute_edge_features(
        self,
        mean_counts: Float[Array, "n_genes"],
        tf_indices: Int[Array, "n_tfs"],
    ) -> Float[Array, "n_edges 3"]:
        """Compute per-edge expression features for the bipartite graph.

        For each TF-gene edge, the feature vector is
        ``[tf_mean_expr, gene_mean_expr, |tf_mean_expr - gene_mean_expr|]``.

        Args:
            mean_counts: Mean expression per gene across cells ``(n_genes,)``.
            tf_indices: Indices of TF genes in the expression matrix.

        Returns:
            Edge features of shape ``(n_tfs * n_genes, 3)``.
        """
        n_tfs = tf_indices.shape[0]
        n_genes = mean_counts.shape[0]

        tf_expr = mean_counts[tf_indices]  # (n_tfs,)

        # Expand to edge level: each TF expression repeated n_genes times
        tf_expr_edges = jnp.repeat(tf_expr, n_genes)  # (n_tfs * n_genes,)
        gene_expr_edges = jnp.tile(mean_counts, n_tfs)  # (n_tfs * n_genes,)

        abs_diff = jnp.abs(tf_expr_edges - gene_expr_edges)

        return jnp.stack([tf_expr_edges, gene_expr_edges, abs_diff], axis=-1)

    def _extract_grn_from_attention(
        self,
        node_features_updated: Float[Array, "n_nodes hidden_dim"],
        edge_index: Int[Array, "2 n_edges"],
        n_tfs: int,
        n_genes: int,
    ) -> Float[Array, "n_tfs n_genes"]:
        """Extract GRN matrix from updated node representations.

        Computes a regulatory score for each TF-gene pair by concatenating
        the updated TF and gene node features and projecting to a scalar.

        Args:
            node_features_updated: Updated node features from GATv2.
            edge_index: Bipartite edge index ``(2, n_edges)``.
            n_tfs: Number of TFs.
            n_genes: Number of genes.

        Returns:
            Raw GRN matrix of shape ``(n_tfs, n_genes)``.
        """
        sources = edge_index[0]  # TF node indices
        targets = edge_index[1]  # Gene node indices

        # Concatenate source (TF) and target (gene) features per edge
        src_features = node_features_updated[sources]  # (n_edges, hidden_dim)
        tgt_features = node_features_updated[targets]  # (n_edges, hidden_dim)
        edge_repr = jnp.concatenate([src_features, tgt_features], axis=-1)

        # Project to scalar score per edge
        scores = self.score_proj(edge_repr).squeeze(-1)  # (n_edges,)

        # Reshape to (n_tfs, n_genes)
        return scores.reshape(n_tfs, n_genes)

    def _apply_soft_sparsity(
        self,
        grn_matrix: Float[Array, "n_tfs n_genes"],
    ) -> Float[Array, "n_tfs n_genes"]:
        """Apply soft L1 sparsity via sigmoid gating.

        Implements ``grn * sigmoid(grn / temperature)`` which pushes small
        values toward zero while preserving strong regulatory signals.

        Args:
            grn_matrix: Raw GRN scores.

        Returns:
            Sparsified GRN matrix.
        """
        gate = jax.nn.sigmoid(grn_matrix / (self.sparsity_temperature + EPSILON))
        return grn_matrix * gate

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply differentiable GRN inference.

        Args:
            data: Dictionary containing:
                - ``"counts"``: Gene expression matrix ``(n_cells, n_genes)``
                - ``"tf_indices"``: Indices of TF genes ``(n_tfs,)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used (non-stochastic operator).
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains all original keys plus:

                    - ``"grn_matrix"``: Sparse regulatory matrix ``(n_tfs, n_genes)``
                    - ``"tf_activity"``: Per-cell TF activity ``(n_cells, n_tfs)``
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts = data["counts"]  # (n_cells, n_genes)
        tf_indices = data["tf_indices"]  # (n_tfs,)

        n_tfs = tf_indices.shape[0]
        n_genes = counts.shape[1]

        # Step 1: Build bipartite graph
        edge_index = self._build_bipartite_graph(n_tfs, n_genes)

        # Step 2: Compute mean expression per gene across cells
        mean_counts = jnp.mean(counts, axis=0)  # (n_genes,)

        # Step 3: Build node features -- one node per TF + one per gene
        # TF nodes get mean TF expression, gene nodes get mean gene expression
        tf_expr = mean_counts[tf_indices]  # (n_tfs,)
        all_expr = jnp.concatenate([tf_expr, mean_counts], axis=0)  # (n_tfs + n_genes,)

        # Project scalar expression to hidden_dim
        node_features = self.node_proj(all_expr[:, None])  # (n_tfs + n_genes, hidden_dim)

        # Step 4: Compute edge features
        edge_features = self._compute_edge_features(mean_counts, tf_indices)

        # Step 5: Apply GATv2 on bipartite graph
        node_features_updated = self.gat_layer(
            node_features,
            edge_index,
            edge_features,
            deterministic=True,
        )

        # Step 6: Extract GRN matrix from updated node features
        raw_grn = self._extract_grn_from_attention(
            node_features_updated, edge_index, n_tfs, n_genes
        )

        # Step 7: Apply soft L1 sparsity
        grn_matrix = self._apply_soft_sparsity(raw_grn)

        # Step 8: Compute TF activity per cell
        # Each TF's activity in a cell is the sum of all gene expressions
        # weighted by that TF's regulatory strengths: activity_tj = sum_g(expr_g * grn_tg)
        tf_activity = counts @ grn_matrix.T  # (n_cells, n_genes) @ (n_genes, n_tfs)

        transformed_data = {
            **data,
            "grn_matrix": grn_matrix,
            "tf_activity": tf_activity,
        }

        return transformed_data, state, metadata
