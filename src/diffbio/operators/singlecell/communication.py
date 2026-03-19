"""Ligand-receptor co-expression scoring for cell-cell communication.

This module provides a differentiable implementation of ligand-receptor
interaction scoring, enabling gradient-based optimization of cell-cell
communication analysis in single-cell data.

Key technique: Soft adjacency weighting via fuzzy k-NN combined with
product-of-expression scoring yields fully differentiable L-R scores
with analytical z-score significance testing.

Applications: CellChat/CellPhoneDB-style cell-cell communication
analysis with end-to-end differentiability.

Inherits from TemperatureOperator to get:

- _temperature property for temperature-controlled smoothing
- soft_max() for logsumexp-based smooth maximum
- soft_argmax() for soft position selection
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.constants import EPSILON
from diffbio.core.base_operators import TemperatureOperator
from diffbio.core.graph_utils import (
    compute_fuzzy_membership,
    compute_pairwise_distances,
    symmetrize_graph,
)


@dataclass
class LRScoringConfig(OperatorConfig):
    """Configuration for ligand-receptor co-expression scoring.

    Attributes:
        n_neighbors: Number of nearest neighbors for k-NN graph.
        temperature: Temperature for soft p-value sigmoid.
        learnable_temperature: Whether the temperature is a learnable parameter.
        metric: Distance metric for k-NN, either ``"euclidean"`` or ``"cosine"``.
    """

    n_neighbors: int = 15
    temperature: float = 1.0
    learnable_temperature: bool = False
    metric: str = "euclidean"


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
        distances = distances + jnp.eye(n_cells) * 1e10

        membership = compute_fuzzy_membership(distances, k=self.config.n_neighbors)
        return symmetrize_graph(membership)

    def _score_lr_pair(
        self,
        adjacency: Float[Array, "n_cells n_cells"],
        ligand_expression: Float[Array, "n_cells"],
        receptor_expression: Float[Array, "n_cells"],
    ) -> Float[Array, "n_cells"]:
        """Compute per-cell L-R interaction score for a single pair.

        ``score_i = sum_j(adjacency[i,j] * L[j] * R[i])``

        The score captures how much ligand signal cell *i* receives from its
        neighbors, weighted by its own receptor expression.

        Args:
            adjacency: Symmetric adjacency matrix.
            ligand_expression: Ligand gene expression per cell.
            receptor_expression: Receptor gene expression per cell.

        Returns:
            Per-cell interaction score of shape ``(n_cells,)``.
        """
        # Weighted ligand signal from neighbors: sum_j(A[i,j] * L[j])
        neighbor_ligand = adjacency @ ligand_expression  # (n_cells,)
        # Multiply by receiver's receptor expression
        return neighbor_ligand * receptor_expression

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
            expected = jnp.sum(mean_ligand * receptor_expr * row_sums)

            # Approximate std: use the variance of the product L[j]*R[i]
            # scaled by the adjacency magnitude
            var_ligand = jnp.var(ligand_expr) + EPSILON
            var_receptor = jnp.var(receptor_expr) + EPSILON
            # Variance approximation for the sum of weighted products
            sum_adj_sq = jnp.sum(adjacency**2)
            std_approx = jnp.sqrt(var_ligand * var_receptor * sum_adj_sq + EPSILON)

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
