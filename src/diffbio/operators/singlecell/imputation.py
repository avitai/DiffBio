"""MAGIC-style diffusion imputation operator for single-cell data.

This module provides a differentiable implementation of diffusion-based data
imputation inspired by MAGIC (Markov Affinity-based Graph Imputation of Cells).
A cell-cell affinity graph is constructed, symmetrized, row-normalized into a
Markov transition matrix, and then raised to power *t* via eigendecomposition
to diffuse information across neighboring cells.

Key technique: Eigendecomposition of the Markov matrix enables differentiable
matrix powering, allowing gradients to flow back through the diffusion process.

Applications: Denoising dropout events in scRNA-seq count matrices, recovering
gene-gene relationships masked by technical noise.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.core.graph_utils import (
    compute_fuzzy_membership,
    compute_pairwise_distances,
    symmetrize_graph,
)


@dataclass
class DiffusionImputerConfig(OperatorConfig):
    """Configuration for MAGIC-style diffusion imputation.

    Attributes:
        n_neighbors: Number of neighbors for local bandwidth estimation.
        diffusion_t: Number of diffusion time steps (matrix power).
        n_pca_components: Number of PCA components (reserved for future use).
        metric: Distance metric, either ``"euclidean"`` or ``"cosine"``.
    """

    n_neighbors: int = 15
    diffusion_t: int = 3
    n_pca_components: int = 100
    metric: str = "euclidean"


class DifferentiableDiffusionImputer(OperatorModule):
    """Differentiable MAGIC-style diffusion imputation.

    Constructs a cell-cell affinity graph from pairwise distances, computes
    a fuzzy membership matrix, symmetrizes it, row-normalizes to obtain a
    Markov transition matrix, and raises it to power *t* via eigendecomposition
    to perform diffusion-based imputation.

    Algorithm:
        1. Compute pairwise distances between cells
        2. Mask the diagonal with a large sentinel value
        3. Compute fuzzy membership using local bandwidth (k-th neighbor)
        4. Symmetrize the graph via fuzzy set union
        5. Row-normalize to obtain Markov transition matrix M
        6. Compute M^t via eigendecomposition
        7. Impute: ``imputed = M^t @ counts``

    Args:
        config: DiffusionImputerConfig with operator parameters.
        rngs: Flax NNX random number generators (not used, kept for API).
        name: Optional operator name.

    Example:
        >>> config = DiffusionImputerConfig(n_neighbors=15, diffusion_t=3)
        >>> imputer = DifferentiableDiffusionImputer(config, rngs=nnx.Rngs(0))
        >>> data = {"counts": jnp.ones((100, 2000))}
        >>> result, state, meta = imputer.apply(data, {}, None)
        >>> result["imputed_counts"].shape
        (100, 2000)
    """

    def __init__(
        self,
        config: DiffusionImputerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the diffusion imputer.

        Args:
            config: Imputation configuration.
            rngs: Random number generators (unused, present for API consistency).
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

    def _build_markov_matrix(
        self,
        counts: Float[Array, "n_cells n_genes"],
    ) -> Float[Array, "n_cells n_cells"]:
        """Build the row-stochastic Markov transition matrix from counts.

        Args:
            counts: Gene expression matrix of shape ``(n_cells, n_genes)``.

        Returns:
            Row-stochastic Markov transition matrix of shape ``(n_cells, n_cells)``.
        """
        n_cells = counts.shape[0]

        # Step 1: Pairwise distances
        distances = compute_pairwise_distances(counts, metric=self.config.metric)

        # Step 2: Mask diagonal
        distances = distances + jnp.eye(n_cells) * 1e10

        # Step 3: Fuzzy membership
        membership = compute_fuzzy_membership(distances, k=self.config.n_neighbors)

        # Step 4: Symmetrize
        symmetric = symmetrize_graph(membership)

        # Step 5: Row-normalize to get Markov transition matrix
        row_sums = jnp.sum(symmetric, axis=1, keepdims=True)
        markov = symmetric / (row_sums + 1e-10)

        return markov

    def _diffuse(
        self,
        markov: Float[Array, "n_cells n_cells"],
        counts: Float[Array, "n_cells n_genes"],
        t: int,
    ) -> tuple[Float[Array, "n_cells n_genes"], Float[Array, "n_cells n_cells"]]:
        """Apply diffusion by raising the Markov matrix to power t.

        Uses eigendecomposition for differentiable matrix powering:
        ``M^t = V @ diag(lambda^t) @ V^T``

        Args:
            markov: Row-stochastic Markov transition matrix.
            counts: Original gene expression counts.
            t: Diffusion time (exponent).

        Returns:
            Tuple of (imputed counts, diffusion operator M^t).
        """
        n_cells = markov.shape[0]

        if t == 0:
            identity = jnp.eye(n_cells)
            return counts, identity

        # Make the matrix symmetric for eigh (it should be nearly symmetric
        # after symmetrize_graph + row normalization, but enforce it)
        markov_sym = (markov + markov.T) / 2.0

        # Eigendecomposition of the symmetric matrix
        eigenvalues, eigenvectors = jnp.linalg.eigh(markov_sym)

        # Clamp eigenvalues to [0, 1] for numerical stability
        eigenvalues = jnp.clip(eigenvalues, 0.0, 1.0)

        # M^t = V @ diag(lambda^t) @ V^T
        eigenvalues_t = eigenvalues**t
        diffusion_op = eigenvectors @ jnp.diag(eigenvalues_t) @ eigenvectors.T

        # Ensure row-stochasticity after powering
        row_sums = jnp.sum(diffusion_op, axis=1, keepdims=True)
        diffusion_op = diffusion_op / (row_sums + 1e-10)

        # Impute
        imputed = diffusion_op @ counts

        return imputed, diffusion_op

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply diffusion imputation to single-cell count data.

        Args:
            data: Dictionary containing:
                - ``"counts"``: Gene expression matrix ``(n_cells, n_genes)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used (deterministic operator).
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - ``"counts"``: Original counts
                    - ``"imputed_counts"``: Diffusion-imputed counts
                    - ``"diffusion_operator"``: The M^t matrix
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts = data["counts"]

        # Build Markov transition matrix
        markov = self._build_markov_matrix(counts)

        # Diffuse
        imputed, diffusion_op = self._diffuse(markov, counts, self.config.diffusion_t)

        transformed_data = {
            **data,
            "imputed_counts": imputed,
            "diffusion_operator": diffusion_op,
        }

        return transformed_data, state, metadata
