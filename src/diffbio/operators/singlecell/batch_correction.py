"""Differentiable Harmony-style batch correction operator.

This module provides a differentiable implementation of batch correction
using soft clustering with batch-aware centroid updates.

Key technique: Unrolled iterations enable gradient flow through
the entire batch correction process.

Applications: Multi-sample integration, batch effect removal.

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

from diffbio.core.base_operators import TemperatureOperator


@dataclass
class BatchCorrectionConfig(OperatorConfig):
    """Configuration for DifferentiableHarmony.

    Attributes:
        n_clusters: Number of clusters for soft assignment.
        n_features: Dimensionality of input embeddings.
        n_batches: Number of distinct batches.
        n_iterations: Number of correction iterations.
        theta: Diversity penalty parameter.
        sigma: Soft assignment bandwidth.
        temperature: Temperature for softmax operations.
    """

    n_clusters: int = 100
    n_features: int = 50
    n_batches: int = 2
    n_iterations: int = 10
    theta: float = 2.0
    sigma: float = 0.1
    temperature: float = 1.0


class DifferentiableHarmony(TemperatureOperator):
    """Differentiable Harmony-style batch correction.

    This operator implements iterative batch correction using soft
    clustering with batch-aware updates. The fixed number of iterations
    enables gradient flow through the entire correction process.

    Algorithm:
    1. Initialize cluster centroids from data
    2. Soft assignment of cells to clusters
    3. Compute batch-aware centroid corrections
    4. Update cell embeddings toward corrected centroids
    5. Repeat for n_iterations

    Inherits from TemperatureOperator to get:
    - _temperature property for temperature-controlled smoothing
    - soft_max() for logsumexp-based smooth maximum
    - soft_argmax() for soft position selection

    Args:
        config: BatchCorrectionConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = BatchCorrectionConfig(n_clusters=100, n_batches=3)
        >>> harmony = DifferentiableHarmony(config, rngs=nnx.Rngs(42))
        >>> data = {"embeddings": X, "batch_labels": batch}
        >>> result, state, meta = harmony.apply(data, {}, None)
    """

    def __init__(
        self,
        config: BatchCorrectionConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the batch correction operator.

        Args:
            config: Batch correction configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.n_clusters = config.n_clusters
        self.n_features = config.n_features
        self.n_batches = config.n_batches
        self.n_iterations = config.n_iterations
        self.theta = config.theta
        self.sigma = config.sigma
        # Temperature is now managed by TemperatureOperator via self._temperature

        # Initialize cluster centroids
        key = rngs.params()
        init_centroids = jax.random.normal(key, (config.n_clusters, config.n_features)) * 0.1
        self.cluster_centroids = nnx.Param(init_centroids)

    def compute_soft_assignments(
        self,
        embeddings: Float[Array, "n_cells n_features"],
        centroids: Float[Array, "n_clusters n_features"],
    ) -> Float[Array, "n_cells n_clusters"]:
        """Compute soft cluster assignments.

        Args:
            embeddings: Cell embeddings.
            centroids: Cluster centroids.

        Returns:
            Soft assignment probabilities.
        """
        # Compute squared distances
        # ||x - c||² = ||x||² + ||c||² - 2 * x · c
        emb_sq = jnp.sum(embeddings**2, axis=-1, keepdims=True)
        cent_sq = jnp.sum(centroids**2, axis=-1)
        dot_product = jnp.einsum("nf,kf->nk", embeddings, centroids)
        distances_sq = emb_sq + cent_sq - 2 * dot_product

        # Soft assignments
        # Use inherited _temperature property from TemperatureOperator
        assignments = jax.nn.softmax(-distances_sq / (self.sigma * self._temperature), axis=-1)

        return assignments

    def compute_batch_proportions(
        self,
        batch_labels: Int[Array, "n_cells"],
        assignments: Float[Array, "n_cells n_clusters"],
    ) -> Float[Array, "n_clusters n_batches"]:
        """Compute batch proportions within each cluster.

        Args:
            batch_labels: Batch assignments for each cell.
            assignments: Soft cluster assignments.

        Returns:
            Proportion of each batch in each cluster.
        """
        # Create one-hot batch encoding
        batch_onehot = jax.nn.one_hot(batch_labels, self.n_batches)  # (n_cells, n_batches)

        # Weighted count of each batch in each cluster
        # (n_cells, n_clusters).T @ (n_cells, n_batches) -> (n_clusters, n_batches)
        batch_counts = jnp.einsum("nk,nb->kb", assignments, batch_onehot)

        # Normalize to get proportions
        total_per_cluster = jnp.sum(batch_counts, axis=-1, keepdims=True) + 1e-10
        batch_proportions = batch_counts / total_per_cluster

        return batch_proportions

    def correction_step(
        self,
        embeddings: Float[Array, "n_cells n_features"],
        batch_labels: Int[Array, "n_cells"],
        centroids: Float[Array, "n_clusters n_features"],
    ) -> tuple[Float[Array, "n_cells n_features"], Float[Array, "n_cells n_clusters"]]:
        """Perform one correction iteration.

        Args:
            embeddings: Current cell embeddings.
            batch_labels: Batch assignments.
            centroids: Current cluster centroids.

        Returns:
            Corrected embeddings and soft assignments.
        """
        # Compute soft assignments
        assignments = self.compute_soft_assignments(embeddings, centroids)

        # Compute batch proportions
        batch_props = self.compute_batch_proportions(batch_labels, assignments)

        # Global batch proportions (target)
        batch_onehot = jax.nn.one_hot(batch_labels, self.n_batches)
        global_batch_props = jnp.mean(batch_onehot, axis=0)  # (n_batches,)

        # Compute diversity correction factor
        # Higher diversity penalty for clusters with skewed batch composition
        # Note: diversity_penalty could be used for additional regularization
        _diversity = 1.0 - jnp.sum(batch_props**2, axis=-1)  # (n_clusters,)

        # Compute correction direction for each cell
        # Move cells toward cluster centroids weighted by assignment and batch correction
        weighted_centroids = jnp.einsum(
            "nk,kf->nf", assignments, centroids
        )  # (n_cells, n_features)

        # Correction: small step toward weighted centroid
        correction = (weighted_centroids - embeddings) * 0.1

        # Apply batch-specific scaling
        # Cells from over-represented batches get larger corrections
        batch_idx = batch_labels  # (n_cells,)
        # Get batch proportion at each cell's most likely cluster
        top_cluster = jnp.argmax(assignments, axis=-1)  # (n_cells,)
        cell_batch_prop = batch_props[top_cluster, batch_idx]  # (n_cells,)
        cell_global_prop = global_batch_props[batch_idx]  # (n_cells,)

        # Scale correction by how overrepresented the batch is
        correction_scale = jnp.clip(cell_batch_prop / (cell_global_prop + 1e-10), 0.5, 2.0)
        correction = correction * correction_scale[:, None]

        # Apply correction
        corrected = embeddings + correction

        return corrected, assignments

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply batch correction to cell embeddings.

        Args:
            data: Dictionary containing:
                - "embeddings": Cell embeddings (n_cells, n_features)
                - "batch_labels": Batch assignments (n_cells,)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:
                    - "embeddings": Original embeddings
                    - "batch_labels": Original batch labels
                    - "corrected_embeddings": Batch-corrected embeddings
                    - "cluster_assignments": Final soft cluster assignments
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        embeddings = data["embeddings"]
        batch_labels = data["batch_labels"]
        centroids = self.cluster_centroids[...]

        # Run correction iterations
        corrected = embeddings
        assignments = None

        def iteration_step(carry, _):
            corrected_emb, centroids = carry
            new_corrected, new_assignments = self.correction_step(
                corrected_emb, batch_labels, centroids
            )
            return (new_corrected, centroids), new_assignments

        (corrected, _), assignments = jax.lax.scan(
            iteration_step, (corrected, centroids), None, length=self.n_iterations
        )

        # Get final assignments
        final_assignments = self.compute_soft_assignments(corrected, centroids)

        # Build output data
        transformed_data = {
            "embeddings": embeddings,
            "batch_labels": batch_labels,
            "corrected_embeddings": corrected,
            "cluster_assignments": final_assignments,
        }

        return transformed_data, state, metadata
