"""Differentiable PHATE dimensionality reduction.

This module implements a differentiable version of PHATE (Potential of
Heat-diffusion for Affinity-based Trajectory Embedding) for dimensionality
reduction with end-to-end gradient flow.

PHATE embeds high-dimensional data by:

1. Building an alpha-decay affinity kernel from pairwise distances.
2. Symmetrizing and row-normalizing to a Markov transition matrix.
3. Powering the matrix to diffusion time *t* via eigendecomposition.
4. Computing potential distances (log or sqrt transform of diffused matrix).
5. Applying classical MDS to the potential distance matrix for embedding.

Reference:
    Moon et al., *Visualizing transitions and structure for biological data
    exploration*, Nature Biotechnology, 2019.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx

from diffbio.constants import DISTANCE_MASK_SENTINEL, EPSILON
from diffbio.core.graph_utils import compute_pairwise_distances, symmetrize_graph

# Regularization added to eigenvalues to prevent NaN gradients from
# repeated or near-zero eigenvalues in ``jnp.linalg.eigh``.
_EIGENVALUE_REGULARIZATION = 1e-6


@dataclass
class PHATEConfig(OperatorConfig):
    """Configuration for differentiable PHATE.

    Attributes:
        n_components: Number of dimensions in the embedding.
        n_neighbors: Number of nearest neighbors for local bandwidth.
        decay: Exponent for the alpha-decaying kernel. Higher values
            produce sharper kernel tails (PHATE default 40).
        diffusion_t: Power to which the diffusion operator is raised.
            Controls the level of diffusion smoothing.
        gamma: Informational distance constant. ``gamma=1`` gives the log
            potential, ``gamma=0`` gives the sqrt potential.
        input_features: Number of input features (used for projection network).
        hidden_dim: Hidden dimension for the projection network.
    """

    n_components: int = 2
    n_neighbors: int = 5
    decay: float = 40.0
    diffusion_t: int = 10
    gamma: float = 1.0
    input_features: int = 64
    hidden_dim: int = 32


class DifferentiablePHATE(OperatorModule):
    """Differentiable PHATE for dimensionality reduction.

    Implements the full PHATE pipeline in a differentiable manner using JAX:

    1. Pairwise distances via ``compute_pairwise_distances`` (DRY).
    2. Alpha-decay affinity kernel: ``K(i,j) = exp(-(d(i,j)/sigma_i)^decay)``
       where ``sigma_i`` is the distance to the k-th neighbor.
    3. Symmetrize via ``symmetrize_graph`` (DRY).
    4. Row-normalize to Markov matrix ``M``.
    5. Diffusion ``M^t`` via eigendecomposition.
    6. Potential distance: ``-log(M^t + eps)`` for ``gamma=1`` (log),
       or ``(M^t)^((1-gamma)/2) / ((1-gamma)/2)`` otherwise.
    7. Classical MDS on the potential distance matrix: center, eigendecompose,
       take top ``n_components`` eigenvectors.

    Example:
        ```python
        config = PHATEConfig(n_components=2, n_neighbors=5, diffusion_t=10)
        phate = DifferentiablePHATE(config, rngs=rngs)

        data = {"features": high_dim_data}  # (n_samples, n_features)
        result, state, metadata = phate.apply(data, {}, None)
        embedding = result["embedding"]  # (n_samples, n_components)
        ```
    """

    def __init__(
        self,
        config: PHATEConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize the differentiable PHATE operator.

        Args:
            config: Configuration for PHATE.
            rngs: Random number generators for parameter initialization.
        """
        super().__init__(config, rngs=rngs)
        self.config = config

    def _build_alpha_decay_affinity(
        self,
        distances: jax.Array,
        k: int,
        decay: float,
    ) -> jax.Array:
        """Build alpha-decaying kernel following PHATE.

        Computes a locality-adaptive Gaussian kernel:
        ``K(i,j) = exp(-alpha * (d(i,j) / sigma_i)^2)``
        where ``sigma_i`` is the distance to the k-th nearest neighbor and
        ``alpha = decay / 2``.

        This is a differentiable relaxation of the original PHATE kernel
        ``exp(-(d/sigma)^decay)`` which has vanishing gradients for high decay.
        The Gaussian formulation preserves the same adaptive-bandwidth locality
        structure while maintaining stable gradient flow.

        Args:
            distances: Pairwise distance matrix of shape ``(n, n)`` with
                diagonal masked to ``DISTANCE_MASK_SENTINEL``.
            k: Number of neighbors for local bandwidth estimation.
            decay: Controls kernel sharpness.  Mapped to Gaussian scale
                ``alpha = decay / 2``.  Higher values produce sharper falloff.

        Returns:
            Affinity matrix of shape ``(n, n)`` with zero diagonal.
        """
        n = distances.shape[0]
        k_eff = min(k, n - 1)

        # Local bandwidth: distance to the k-th nearest neighbor
        sorted_dists = jnp.sort(distances, axis=-1)
        sigma = jnp.maximum(sorted_dists[:, k_eff - 1], EPSILON)

        # Gaussian kernel with adaptive bandwidth:
        # K(i,j) = exp(-alpha * (d(i,j) / sigma_i)^2)
        # alpha = decay / 2 maps the PHATE decay parameter to
        # the Gaussian scale while preserving locality structure.
        alpha = decay / 2.0
        ratio = distances / sigma[:, None]
        affinity = jnp.exp(-alpha * ratio**2)

        # Zero diagonal
        affinity = affinity * (1.0 - jnp.eye(n))

        return affinity

    def _build_markov_matrix(self, affinity_sym: jax.Array) -> jax.Array:
        """Row-normalize a symmetric affinity matrix to a Markov matrix.

        Args:
            affinity_sym: Symmetric affinity matrix of shape ``(n, n)``.

        Returns:
            Row-stochastic Markov matrix of shape ``(n, n)``.
        """
        row_sums = jnp.sum(affinity_sym, axis=1, keepdims=True)
        return affinity_sym / jnp.maximum(row_sums, EPSILON)

    def _diffuse_eigendecomposition(
        self,
        markov: jax.Array,
        t: int,
    ) -> jax.Array:
        """Compute ``M^t`` via eigendecomposition.

        Symmetrizes the Markov matrix via the similarity transform
        ``S = D^{1/2} M D^{-1/2}`` (where ``D = diag(rowsums)``),
        eigendecomposes the symmetric ``S``, powers the eigenvalues,
        and reconstructs ``M^t = D^{-1/2} V diag(lambda^t) V^T D^{1/2}``.

        A small regularization is added to diagonal of ``S`` before
        eigendecomposition so that repeated eigenvalues are slightly split,
        preventing NaN gradients in the backward pass.

        Args:
            markov: Row-stochastic Markov matrix of shape ``(n, n)``.
            t: Diffusion time (power to raise the matrix to).

        Returns:
            Diffusion operator ``M^t`` of shape ``(n, n)``.
        """
        n = markov.shape[0]

        if t == 0:
            return jnp.eye(n)

        # Similarity transform to symmetric matrix with same spectrum
        degree = jnp.sum(markov, axis=1)
        degree = jnp.maximum(degree, EPSILON)
        d_sqrt = jnp.sqrt(degree)
        d_inv_sqrt = 1.0 / d_sqrt

        sym_matrix = d_sqrt[:, None] * markov * d_inv_sqrt[None, :]

        # Force exact symmetry
        sym_matrix = 0.5 * (sym_matrix + sym_matrix.T)

        # Add tiny diagonal regularization to split repeated eigenvalues
        reg = _EIGENVALUE_REGULARIZATION * jnp.eye(n)
        sym_matrix = sym_matrix + reg

        # Eigendecompose
        eigenvalues, eigenvectors = jnp.linalg.eigh(sym_matrix)

        # Clamp eigenvalues to non-negative, normalize by max
        eigenvalues = jnp.maximum(eigenvalues, 0.0)
        lambda_max = jnp.maximum(eigenvalues[-1], EPSILON)
        eigenvalues_normalized = eigenvalues / lambda_max

        # Power the normalized eigenvalues
        eigenvalues_t = eigenvalues_normalized**t

        # Reconstruct: M^t = D^{-1/2} V diag(lambda^t) V^T D^{1/2}
        v_left = d_inv_sqrt[:, None] * eigenvectors
        v_right = d_sqrt[:, None] * eigenvectors
        diffusion_op = v_left @ jnp.diag(eigenvalues_t) @ v_right.T

        # Ensure row-stochasticity after reconstruction
        row_sums = jnp.sum(diffusion_op, axis=1, keepdims=True)
        diffusion_op = diffusion_op / jnp.maximum(row_sums, EPSILON)

        # Clamp to non-negative (numerical noise can create tiny negatives)
        diffusion_op = jnp.maximum(diffusion_op, 0.0)

        return diffusion_op

    def _compute_potential_distances(
        self,
        diffusion_op: jax.Array,
        gamma: float,
    ) -> jax.Array:
        """Compute potential distances from the diffusion operator.

        Following PHATE:
        - ``gamma=1``: log potential ``D = -log(M^t + eps)``
        - ``gamma=0``: sqrt potential ``D = (M^t)^{1/2} / (1/2)``
        - ``gamma=-1``: identity (no transform)
        - general: ``D = (M^t)^c / c`` where ``c = (1-gamma)/2``

        The result is symmetrized for use in MDS.

        Args:
            diffusion_op: Diffusion operator ``M^t`` of shape ``(n, n)``.
            gamma: Informational distance constant.

        Returns:
            Symmetric potential distance matrix of shape ``(n, n)``.
        """
        if gamma == 1.0:
            # Log potential (standard PHATE)
            potential = -jnp.log(diffusion_op + EPSILON)
        elif gamma == -1.0:
            # Identity (no transform)
            potential = diffusion_op
        else:
            # General power transform, covers gamma=0 (sqrt) etc.
            c = (1.0 - gamma) / 2.0
            potential = jnp.power(diffusion_op + EPSILON, c) / jnp.maximum(c, EPSILON)

        # Symmetrize for MDS
        potential = 0.5 * (potential + potential.T)

        return potential

    def _classical_mds(
        self,
        distance_matrix: jax.Array,
        n_components: int,
    ) -> jax.Array:
        """Perform classical MDS on a distance matrix.

        Classical MDS (Torgerson scaling):

        1. Square the distances: ``D2 = D^2``
        2. Double-center: ``B = -0.5 * (D2 - row_mean - col_mean + grand_mean)``
        3. Eigendecompose B and take the top ``n_components`` eigenvectors
           scaled by ``sqrt(eigenvalue)``.

        A small diagonal regularization is added to the centered matrix
        before eigendecomposition to split repeated eigenvalues and ensure
        stable gradients.

        Args:
            distance_matrix: Symmetric distance matrix of shape ``(n, n)``.
            n_components: Number of embedding dimensions.

        Returns:
            Embedding of shape ``(n, n_components)``.
        """
        n = distance_matrix.shape[0]
        d_squared = distance_matrix**2

        # Double centering
        row_mean = jnp.mean(d_squared, axis=1, keepdims=True)
        col_mean = jnp.mean(d_squared, axis=0, keepdims=True)
        grand_mean = jnp.mean(d_squared)
        centered = -0.5 * (d_squared - row_mean - col_mean + grand_mean)

        # Force symmetry
        centered = 0.5 * (centered + centered.T)

        # Regularize to split degenerate eigenvalues for stable gradients
        centered = centered + _EIGENVALUE_REGULARIZATION * jnp.eye(n)

        # Eigendecompose: eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = jnp.linalg.eigh(centered)

        # Take the top n_components (largest eigenvalues = last ones)
        top_eigenvalues = eigenvalues[-n_components:]
        top_eigenvectors = eigenvectors[:, -n_components:]

        # Clamp to non-negative for sqrt
        top_eigenvalues = jnp.maximum(top_eigenvalues, EPSILON)

        # Embedding: eigenvectors scaled by sqrt(eigenvalues)
        embedding = top_eigenvectors * jnp.sqrt(top_eigenvalues)[None, :]

        # Reverse so the largest component comes first
        embedding = embedding[:, ::-1]

        return embedding

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Apply PHATE dimensionality reduction.

        Args:
            data: Dictionary containing:
                - ``"features"``: High-dimensional features ``(n_samples, n_features)``
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of ``(output_data, state, metadata)`` where output_data contains:

                - ``"features"``: Original high-dimensional features
                - ``"embedding"``: Low-dimensional PHATE embedding
                  ``(n_samples, n_components)``
                - ``"potential_distances"``: Symmetric potential distance matrix
                  ``(n_samples, n_samples)``
                - ``"diffusion_operator"``: Row-stochastic diffusion matrix
                  ``M^t`` ``(n_samples, n_samples)``
        """
        del random_params, stats  # Unused

        features = data["features"]
        n_samples = features.shape[0]

        # Step 1: Pairwise distances (DRY: reuse graph_utils)
        distances = compute_pairwise_distances(features, metric="euclidean")

        # Mask diagonal for neighbor computation
        distances_masked = distances + jnp.eye(n_samples) * DISTANCE_MASK_SENTINEL

        # Step 2: Alpha-decay affinity kernel
        affinity = self._build_alpha_decay_affinity(
            distances_masked, self.config.n_neighbors, self.config.decay
        )

        # Step 3: Symmetrize (DRY: reuse graph_utils)
        affinity_sym = symmetrize_graph(affinity)

        # Step 4: Row-normalize to Markov matrix
        markov = self._build_markov_matrix(affinity_sym)

        # Step 5: Diffusion via eigendecomposition
        diffusion_op = self._diffuse_eigendecomposition(markov, self.config.diffusion_t)

        # Step 6: Potential distances
        potential_distances = self._compute_potential_distances(
            diffusion_op, self.config.gamma
        )

        # Step 7: Classical MDS embedding
        embedding = self._classical_mds(potential_distances, self.config.n_components)

        output_data = {
            **data,
            "embedding": embedding,
            "potential_distances": potential_distances,
            "diffusion_operator": diffusion_op,
        }

        return output_data, state, metadata
