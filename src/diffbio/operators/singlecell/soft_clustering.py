"""Soft K-Means clustering operator for single-cell analysis.

This module provides a differentiable implementation of soft k-means
clustering, enabling gradient-based learning of cluster centroids.

Key technique: Replace hard cluster assignment with softmax-based
soft assignments for fully differentiable clustering.

Applications: Cell type clustering, Leiden-like community detection.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree


@dataclass
class SoftClusteringConfig(OperatorConfig):
    """Configuration for SoftKMeansClustering.

    Attributes:
        n_clusters: Number of clusters.
        n_features: Dimensionality of input embeddings.
        temperature: Temperature for softmax (lower = sharper).
        learnable_centroids: Whether centroids are learnable parameters.
        stochastic: Whether the operator uses randomness.
        stream_name: RNG stream name.
    """

    n_clusters: int = 10
    n_features: int = 50
    temperature: float = 1.0
    learnable_centroids: bool = True
    stochastic: bool = False
    stream_name: str | None = None


class SoftKMeansClustering(OperatorModule):
    """Differentiable soft k-means clustering.

    This operator implements soft k-means with learnable cluster centroids.
    Instead of hard cluster assignments, cells are softly assigned to clusters
    using softmax over negative squared distances.

    Algorithm:
    1. Compute squared distances from cells to centroids
    2. Apply softmax for soft assignments: P(k|x) = softmax(-||x - c_k||² / T)
    3. Optionally update centroids based on weighted means

    Args:
        config: SoftClusteringConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = SoftClusteringConfig(n_clusters=10, n_features=50)
        >>> clusterer = SoftKMeansClustering(config, rngs=nnx.Rngs(42))
        >>> data = {"embeddings": cell_embeddings}
        >>> result, state, meta = clusterer.apply(data, {}, None)
    """

    def __init__(
        self,
        config: SoftClusteringConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the soft k-means clustering operator.

        Args:
            config: Clustering configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.n_clusters = config.n_clusters
        self.n_features = config.n_features
        self.temperature = config.temperature

        # Initialize cluster centroids
        key = rngs.params()
        init_centroids = jax.random.normal(
            key, (config.n_clusters, config.n_features)
        ) * 0.1
        self.centroids = nnx.Param(init_centroids)

    def compute_distances(
        self,
        embeddings: Float[Array, "n_cells n_features"],
    ) -> Float[Array, "n_cells n_clusters"]:
        """Compute squared distances from cells to centroids.

        Args:
            embeddings: Cell embedding vectors.

        Returns:
            Squared Euclidean distances to each centroid.
        """
        centroids = self.centroids[...]  # (n_clusters, n_features)

        # Efficient distance computation using expansion
        # ||x - c||² = ||x||² + ||c||² - 2 * x · c
        emb_sq = jnp.sum(embeddings ** 2, axis=-1, keepdims=True)  # (n_cells, 1)
        cent_sq = jnp.sum(centroids ** 2, axis=-1)  # (n_clusters,)
        dot_product = jnp.einsum("nf,kf->nk", embeddings, centroids)  # (n_cells, n_clusters)

        distances_sq = emb_sq + cent_sq - 2 * dot_product

        return distances_sq

    def compute_assignments(
        self,
        embeddings: Float[Array, "n_cells n_features"],
    ) -> Float[Array, "n_cells n_clusters"]:
        """Compute soft cluster assignments.

        Args:
            embeddings: Cell embedding vectors.

        Returns:
            Soft assignment probabilities for each cluster.
        """
        distances_sq = self.compute_distances(embeddings)

        # Soft assignments via softmax over negative distances
        assignments = jax.nn.softmax(-distances_sq / self.temperature, axis=-1)

        return assignments

    def get_hard_labels(
        self,
        assignments: Float[Array, "n_cells n_clusters"],
    ) -> Int[Array, "n_cells"]:
        """Get hard cluster labels from soft assignments.

        Args:
            assignments: Soft cluster assignments.

        Returns:
            Hard cluster labels (argmax).
        """
        return jnp.argmax(assignments, axis=-1)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply soft k-means clustering to cell embeddings.

        Args:
            data: Dictionary containing:
                - "embeddings": Cell embeddings (n_cells, n_features)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:
                    - "embeddings": Original embeddings
                    - "cluster_assignments": Soft assignment probabilities
                    - "cluster_labels": Hard cluster labels
                    - "centroids": Cluster centroid positions
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        embeddings = data["embeddings"]

        # Compute soft assignments
        assignments = self.compute_assignments(embeddings)

        # Get hard labels
        labels = self.get_hard_labels(assignments)

        # Build output data
        transformed_data = {
            "embeddings": embeddings,
            "cluster_assignments": assignments,
            "cluster_labels": labels,
            "centroids": self.centroids[...],
        }

        return transformed_data, state, metadata
