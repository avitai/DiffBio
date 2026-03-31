"""Differentiable trajectory inference for single-cell analysis.

This module provides differentiable pseudotime computation and fate probability
estimation, enabling gradient-based optimization of trajectory inference in
single-cell RNA-seq data.

Key techniques:
- Diffusion maps via subspace iteration with QR orthogonalization for
  pseudotime ordering.  This replaces eigendecomposition (whose backward pass
  produces NaN when eigenvalues are near-degenerate — JAX issue #669) with
  repeated matmul + QR, which has well-conditioned gradients.
- Absorption probabilities via linear solve on the fundamental matrix for
  fate probability estimation.

Both operations are end-to-end differentiable, allowing backpropagation
into upstream embeddings.

Applications: Developmental trajectory ordering, lineage fate commitment
analysis, and cell-state transition characterization.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.constants import DISTANCE_MASK_SENTINEL

from diffbio.core.graph_utils import (
    compute_fuzzy_membership,
    compute_pairwise_distances,
    symmetrize_graph,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PseudotimeConfig",
    "DifferentiablePseudotime",
    "FateProbabilityConfig",
    "DifferentiableFateProbability",
]


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PseudotimeConfig(OperatorConfig):
    """Configuration for pseudotime computation.

    Attributes:
        n_neighbors: Number of neighbors for k-NN graph construction.
        n_diffusion_components: Number of diffusion map components to retain.
        root_cell_index: Index of the root cell (pseudotime origin).
        metric: Distance metric, ``"euclidean"`` or ``"cosine"``.
    """

    n_neighbors: int = 15
    n_diffusion_components: int = 10
    root_cell_index: int = 0
    metric: str = "euclidean"


@dataclass(frozen=True)
class FateProbabilityConfig(OperatorConfig):
    """Configuration for fate probability computation.

    Attributes:
        n_macrostates: Number of macrostates (terminal fates).
    """

    n_macrostates: int = 2


# ---------------------------------------------------------------------------
# DifferentiablePseudotime
# ---------------------------------------------------------------------------


class DifferentiablePseudotime(OperatorModule):
    """Differentiable pseudotime computation via diffusion maps.

    Constructs a k-NN affinity graph, builds a Markov transition matrix, and
    computes diffusion components through subspace iteration with QR
    orthogonalization.  Pseudotime is defined as the Euclidean distance in
    diffusion-component space from the designated root cell.

    Algorithm:
        1. Compute pairwise distances between cells.
        2. Compute fuzzy membership with local bandwidth (k-th neighbor).
        3. Symmetrize the graph via fuzzy set union.
        4. Row-normalize to obtain a Markov transition matrix.
        5. Extract the top ``n_diffusion_components`` eigenvectors of the
           symmetrized transition matrix via subspace iteration (repeated
           matmul + QR), excluding the trivial eigenvalue 1.
        6. Weight eigenvectors by their Rayleigh-quotient eigenvalues to
           form diffusion components.
        7. Pseudotime = L2 distance from root cell in diffusion-component
           space.

    Args:
        config: PseudotimeConfig with operator parameters.
        rngs: Flax NNX random number generators (unused, kept for API).
        name: Optional operator name.

    Example:
        >>> config = PseudotimeConfig(n_neighbors=15, n_diffusion_components=10)
        >>> op = DifferentiablePseudotime(config)
        >>> data = {"embeddings": jnp.ones((50, 20))}
        >>> result, state, meta = op.apply(data, {}, None)
        >>> result["pseudotime"].shape
        (50,)
    """

    def __init__(
        self,
        config: PseudotimeConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the pseudotime operator.

        Args:
            config: Pseudotime configuration.
            rngs: Random number generators (unused, present for API consistency).
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

    def _build_transition_matrix(
        self,
        embeddings: Float[Array, "n_cells n_features"],
    ) -> Float[Array, "n_cells n_cells"]:
        """Build a row-stochastic transition matrix from cell embeddings.

        Args:
            embeddings: Cell embedding matrix.

        Returns:
            Row-stochastic Markov transition matrix.
        """
        n_cells = embeddings.shape[0]

        # Pairwise distances
        distances = compute_pairwise_distances(embeddings, metric=self.config.metric)

        # Mask diagonal with large sentinel
        distances = distances + jnp.eye(n_cells) * DISTANCE_MASK_SENTINEL

        # Fuzzy membership with local bandwidth
        membership = compute_fuzzy_membership(distances, k=self.config.n_neighbors)

        # Symmetrize via fuzzy set union
        symmetric = symmetrize_graph(membership)

        # Row-normalize to Markov transition matrix
        row_sums = jnp.sum(symmetric, axis=1, keepdims=True)
        transition = symmetric / (row_sums + 1e-10)

        return transition

    def _compute_diffusion_embedding(
        self,
        transition: Float[Array, "n_cells n_cells"],
        n_components: int,
        root_index: int,
    ) -> tuple[
        Float[Array, "n_cells"],
        Float[Array, "n_cells n_comp"],
    ]:
        """Compute pseudotime and diffusion components from Markov powers.

        Instead of eigendecomposing the transition matrix (whose backward
        pass produces NaN when eigenvalues are near-degenerate — JAX
        issue #669), this method accumulates
        ``M_sum = sum_{t=1}^{T} M^t`` via repeated matrix multiplication.
        The rows of ``M_sum`` form a diffusion embedding: the DPT
        distance between cells *i* and *j* equals the L2 distance
        between rows *i* and *j* of ``M_sum`` (Haghverdi et al. 2016).

        Because the computation uses only matrix multiplication and
        addition, the backward pass is free of the degenerate-eigenvalue
        singularity and produces well-conditioned finite gradients.

        Args:
            transition: Row-stochastic transition matrix.
            n_components: Number of diffusion components to retain in
                the output embedding.
            root_index: Index of the root cell for pseudotime origin.

        Returns:
            Tuple of (pseudotime, diffusion_components) where pseudotime
            has shape ``(n_cells,)`` and diffusion_components has shape
            ``(n_cells, n_components)``.
        """
        # Symmetrize (should be nearly symmetric already)
        sym = (transition + transition.T) / 2.0

        n_cells = transition.shape[0]
        # Use Python min() so the result is a static int (required for JIT).
        n_comp = min(n_components, n_cells - 1)

        # Accumulate M_sum = sum_{t=1}^{T} sym^t.
        # This approximates (I - sym)^{-1} - I (the DPT kernel).
        # Enough terms for the geometric series to converge.
        n_powers = max(n_comp * 2, 10)
        m_power = sym
        m_sum = jnp.zeros_like(sym)
        for _ in range(n_powers):
            m_sum = m_sum + m_power
            m_power = m_power @ sym

        # Pseudotime = L2 distance from root cell in M_sum row space.
        root_row = m_sum[root_index]
        diff = m_sum - root_row[None, :]
        pseudotime = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10)
        pseudotime = pseudotime - pseudotime[root_index]

        # Extract n_comp diffusion components for the output embedding.
        # Center rows to remove the trivial (constant) component, then
        # take the first n_comp columns as coordinates.
        row_mean = jnp.mean(m_sum, axis=0, keepdims=True)
        diffusion_components = (m_sum - row_mean)[:, :n_comp]

        return pseudotime, diffusion_components

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply pseudotime computation to cell embeddings.

        Args:
            data: Dictionary containing:
                - ``"embeddings"``: Cell embeddings ``(n_cells, n_features)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used (deterministic operator).
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - ``"pseudotime"``: Pseudotime values ``(n_cells,)``
                    - ``"diffusion_components"``: Diffusion map coordinates
                      ``(n_cells, n_diffusion_components)``
                    - ``"transition_matrix"``: Markov transition matrix
                      ``(n_cells, n_cells)``
                    - All original data keys are preserved
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        embeddings = data["embeddings"]

        # Build transition matrix
        transition = self._build_transition_matrix(embeddings)

        # Pseudotime and diffusion components from Markov powers
        pseudotime, dc = self._compute_diffusion_embedding(
            transition,
            self.config.n_diffusion_components,
            self.config.root_cell_index,
        )

        transformed_data = {
            **data,
            "pseudotime": pseudotime,
            "diffusion_components": dc,
            "transition_matrix": transition,
        }

        return transformed_data, state, metadata


# ---------------------------------------------------------------------------
# DifferentiableFateProbability
# ---------------------------------------------------------------------------


class DifferentiableFateProbability(OperatorModule):
    """Differentiable fate probability estimation via absorption probabilities.

    Given a Markov transition matrix and a set of terminal (absorbing) state
    indices, partitions cells into transient and absorbing sets and computes
    the probability that each transient cell will eventually reach each
    absorbing state.

    Algorithm:
        1. Partition states into transient (T) and absorbing (A).
        2. Extract sub-matrices Q = transition[T, T] and R = transition[T, A].
        3. Solve ``(I - Q) @ B = R`` for B (absorption probabilities).
        4. Assign probability 1 to each absorbing state for itself.

    The linear solve ``jnp.linalg.solve`` is fully differentiable.

    Args:
        config: FateProbabilityConfig with operator parameters.
        rngs: Flax NNX random number generators (unused, kept for API).
        name: Optional operator name.

    Example:
        >>> config = FateProbabilityConfig(n_macrostates=2)
        >>> op = DifferentiableFateProbability(config)
        >>> data = {"transition_matrix": T, "terminal_states": jnp.array([18, 19])}
        >>> result, state, meta = op.apply(data, {}, None)
        >>> result["fate_probabilities"].shape
        (20, 2)
    """

    def __init__(
        self,
        config: FateProbabilityConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the fate probability operator.

        Args:
            config: Fate probability configuration.
            rngs: Random number generators (unused, present for API consistency).
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

    def _compute_absorption_probabilities(
        self,
        transition_matrix: Float[Array, "n n"],
        terminal_states: Int[Array, "n_terminal"],
    ) -> tuple[Float[Array, "n n_terminal"], Int[Array, "n"]]:
        """Compute absorption probabilities for all cells.

        Args:
            transition_matrix: Row-stochastic Markov transition matrix.
            terminal_states: Indices of terminal (absorbing) states.

        Returns:
            Tuple of (fate_probabilities, macrostates) where
            fate_probabilities has shape ``(n_cells, n_terminal)`` and
            macrostates has shape ``(n_cells,)`` (argmax assignment).
        """
        n_cells = transition_matrix.shape[0]
        n_terminal = terminal_states.shape[0]

        # Build boolean mask for transient states
        is_terminal = jnp.zeros(n_cells, dtype=jnp.bool_)
        is_terminal = is_terminal.at[terminal_states].set(True)
        is_transient = ~is_terminal

        # Index arrays for transient and absorbing states
        transient_indices = jnp.where(is_transient, size=n_cells - n_terminal)[0]

        # Extract sub-matrices
        # Q = transition[transient, transient]
        q_matrix = transition_matrix[jnp.ix_(transient_indices, transient_indices)]

        # R = transition[transient, absorbing]
        r_matrix = transition_matrix[jnp.ix_(transient_indices, terminal_states)]

        # Solve (I - Q) @ B = R  =>  B = (I - Q)^{-1} @ R
        n_transient = transient_indices.shape[0]
        identity = jnp.eye(n_transient)
        # Adding small regularization for numerical stability
        lhs = identity - q_matrix + jnp.eye(n_transient) * 1e-8
        absorption = jnp.linalg.solve(lhs, r_matrix)

        # Clamp to valid probability range
        absorption = jnp.clip(absorption, 0.0, 1.0)

        # Normalize rows to sum to 1
        row_sums = jnp.sum(absorption, axis=1, keepdims=True)
        absorption = absorption / (row_sums + 1e-10)

        # Build full fate probability matrix
        fate = jnp.zeros((n_cells, n_terminal))

        # Set transient-cell probabilities
        fate = fate.at[transient_indices].set(absorption)

        # Set absorbing-cell probabilities: 1 for self, 0 for others
        fate = fate.at[terminal_states, jnp.arange(n_terminal)].set(1.0)

        # Macrostate assignment = argmax of fate probabilities
        macrostates = jnp.argmax(fate, axis=1)

        return fate, macrostates

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply fate probability estimation.

        Args:
            data: Dictionary containing:
                - ``"transition_matrix"``: Markov transition matrix
                  ``(n_cells, n_cells)``
                - ``"terminal_states"``: Indices of terminal states
                  ``(n_terminal,)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used (deterministic operator).
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - ``"fate_probabilities"``: Absorption probabilities
                      ``(n_cells, n_terminal)``
                    - ``"macrostates"``: Argmax fate assignment ``(n_cells,)``
                    - All original data keys are preserved
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        transition_matrix = data["transition_matrix"]
        terminal_states = data["terminal_states"]

        fate, macrostates = self._compute_absorption_probabilities(
            transition_matrix, terminal_states
        )

        transformed_data = {
            **data,
            "fate_probabilities": fate,
            "macrostates": macrostates,
        }

        return transformed_data, state, metadata
