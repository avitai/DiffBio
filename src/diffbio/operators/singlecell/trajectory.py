"""Differentiable trajectory inference for single-cell analysis.

This module provides differentiable pseudotime computation and fate probability
estimation, enabling gradient-based optimization of trajectory inference in
single-cell RNA-seq data.

Key techniques:
- Diffusion maps via eigendecomposition of the Markov transition matrix for
  pseudotime ordering.
- Absorption probabilities via linear solve on the fundamental matrix for
  fate probability estimation.

Both operations are end-to-end differentiable through ``jnp.linalg.eigh``
and ``jnp.linalg.solve``, allowing backpropagation into upstream embeddings.

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
    computes diffusion components through eigendecomposition. Pseudotime is
    defined as the Euclidean distance in diffusion-component space from the
    designated root cell.

    Algorithm:
        1. Compute pairwise distances between cells.
        2. Compute fuzzy membership with local bandwidth (k-th neighbor).
        3. Symmetrize the graph via fuzzy set union.
        4. Row-normalize to obtain a Markov transition matrix.
        5. Eigendecompose the symmetrized transition matrix; take the top
           ``n_diffusion_components`` eigenvectors (excluding the trivial
           eigenvalue 1).
        6. Weight eigenvectors by their eigenvalues to form diffusion
           components.
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

    def _compute_diffusion_components(
        self,
        transition: Float[Array, "n_cells n_cells"],
        n_components: int,
    ) -> tuple[Float[Array, "n_cells n_comp"], Float[Array, "n_comp"]]:
        """Compute diffusion components via eigendecomposition.

        Args:
            transition: Row-stochastic transition matrix.
            n_components: Number of diffusion components to retain.

        Returns:
            Tuple of (diffusion_components, eigenvalues) where components
            have shape ``(n_cells, n_components)`` and eigenvalues have
            shape ``(n_components,)``.
        """
        # Symmetrize for eigh (should be nearly symmetric already)
        sym = (transition + transition.T) / 2.0

        # Eigendecompose -- eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = jnp.linalg.eigh(sym)

        # Take top n_components (largest eigenvalues, excluding trivial lambda=1)
        # The trivial eigenvector is the last one (constant vector).
        # Use Python min() so the result is a static int (required for JIT slicing).
        n_cells = transition.shape[0]
        n_comp = min(n_components, n_cells - 1)

        # eigh returns ascending order; biggest eigenvalues are at the end.
        # Skip the last (trivial lambda~=1) and take the next n_comp.
        selected_vals = eigenvalues[-(n_comp + 1) : -1]  # shape (n_comp,)
        selected_vecs = eigenvectors[:, -(n_comp + 1) : -1]  # (n_cells, n_comp)

        # Clamp eigenvalues to [0, 1] for stability
        selected_vals = jnp.clip(selected_vals, 0.0, 1.0)

        # Diffusion components = eigenvectors * eigenvalues
        diffusion_components = selected_vecs * selected_vals[None, :]

        return diffusion_components, selected_vals

    def _compute_pseudotime(
        self,
        diffusion_components: Float[Array, "n_cells n_comp"],
        root_index: int,
    ) -> Float[Array, "n_cells"]:
        """Compute pseudotime as L2 distance from root in diffusion space.

        Args:
            diffusion_components: Diffusion component matrix.
            root_index: Index of the root cell.

        Returns:
            Pseudotime values for all cells.
        """
        root_dc = diffusion_components[root_index]
        diff = diffusion_components - root_dc[None, :]
        pseudotime = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10)

        # Normalize so root is exactly 0
        pseudotime = pseudotime - pseudotime[root_index]

        return pseudotime

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

        # Diffusion components
        dc, _eigenvalues = self._compute_diffusion_components(
            transition, self.config.n_diffusion_components
        )

        # Pseudotime from root
        pseudotime = self._compute_pseudotime(dc, self.config.root_cell_index)

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
