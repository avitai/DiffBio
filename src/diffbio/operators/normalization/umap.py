"""Differentiable UMAP dimensionality reduction.

This module implements a differentiable version of UMAP (Uniform Manifold
Approximation and Projection) for dimensionality reduction with end-to-end
gradient flow.
"""

import logging
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from artifex.generative_models.core.base import MLP
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule

from diffbio.constants import DISTANCE_MASK_SENTINEL
from diffbio.core.graph_utils import (
    compute_fuzzy_membership,
    compute_pairwise_distances,
    symmetrize_graph,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UMAPConfig(OperatorConfig):
    """Configuration for differentiable UMAP.

    Attributes:
        n_components: Number of dimensions in the embedding.
        n_neighbors: Number of neighbors for local structure preservation.
        min_dist: Minimum distance between points in the embedding.
        spread: Effective scale of embedded points.
        metric: Distance metric ('euclidean' or 'cosine').
        learning_rate: Learning rate for optimization.
        negative_sample_rate: Number of negative samples per positive.
        input_features: Number of input features (required for initialization).
        hidden_dim: Hidden dimension for projection network.
        stream_name: Name of the data stream to process.
    """

    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    spread: float = 1.0
    metric: str = "euclidean"
    learning_rate: float = 1.0
    negative_sample_rate: int = 5
    input_features: int = 64
    hidden_dim: int = 32


class ParametricUMAPHead(nnx.Module):
    """Parametric UMAP embedding head with learnable similarity curve."""

    def __init__(self, config: UMAPConfig, *, rngs: nnx.Rngs):
        """Initialize the parametric embedding head."""
        super().__init__()
        self.curve_params = nnx.Param(jnp.array([1.929, 0.7915], dtype=jnp.float32))
        self.projection_backbone = MLP(
            hidden_dims=[config.hidden_dim, config.n_components],
            in_features=config.input_features,
            activation="relu",
            output_activation=None,
            use_batch_norm=False,
            rngs=rngs,
        )

    def project(self, features: jax.Array) -> jax.Array:
        """Project high-dimensional features to low-dimensional embedding."""
        embedding = self.projection_backbone(features)
        if isinstance(embedding, tuple):
            raise TypeError("ParametricUMAPHead projection backbone must return a single tensor.")
        return embedding

    def curve_coefficients(self) -> tuple[jax.Array, jax.Array]:
        """Return positive low-dimensional similarity curve coefficients."""
        a = jnp.abs(self.curve_params[0]) + 1e-6
        b = jnp.abs(self.curve_params[1]) + 1e-6
        return a, b


class DifferentiableUMAP(OperatorModule):
    """Differentiable UMAP for dimensionality reduction.

    This operator implements a simplified differentiable version of UMAP that
    learns a low-dimensional embedding while preserving local structure.

    The UMAP loss function is:
        L = sum_edges [p_ij * log(q_ij) + (1 - p_ij) * log(1 - q_ij)]

    where:
        - p_ij is the high-dimensional similarity (fuzzy set membership)
        - q_ij is the low-dimensional similarity

    This implementation uses a parametric approach with learnable curve
    parameters (a, b) for the low-dimensional similarity function.

    Example:
        ```python
        config = UMAPConfig(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
        )
        umap = DifferentiableUMAP(config, rngs=rngs)

        data = {"features": high_dim_data}  # (n_samples, n_features)
        result, state, metadata = umap.apply(data, {}, None)
        embedding = result["embedding"]  # (n_samples, n_components)
        ```
    """

    def __init__(self, config: UMAPConfig, *, rngs: nnx.Rngs | None = None):
        """Initialize the differentiable UMAP.

        Args:
            config: Configuration for UMAP.
            rngs: Random number generators for initialization.
        """
        super().__init__(config, rngs=rngs)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.embedding_head = ParametricUMAPHead(config, rngs=rngs)

    def _project(self, features: jax.Array) -> jax.Array:
        """Project high-dimensional features to low-dimensional embedding.

        Args:
            features: Input features of shape (n_samples, n_features).

        Returns:
            Embedding of shape (n_samples, n_components).
        """
        return self.embedding_head.project(features)

    def _compute_high_dim_similarities(self, features: jax.Array) -> jax.Array:
        """Compute high-dimensional fuzzy set membership (p_ij).

        Uses k-nearest neighbors and Gaussian kernel with local bandwidth.
        Delegates to reusable graph utilities in ``diffbio.core.graph_utils``.

        Args:
            features: Input features of shape (n_samples, n_features).

        Returns:
            Similarity matrix of shape (n_samples, n_samples).
        """
        n_samples = features.shape[0]
        n_neighbors = min(self.config.n_neighbors, n_samples - 1)

        distances = compute_pairwise_distances(features, metric=self.config.metric)
        distances = distances + jnp.eye(n_samples) * DISTANCE_MASK_SENTINEL

        p_ij = compute_fuzzy_membership(distances, k=n_neighbors)
        return symmetrize_graph(p_ij)

    def _compute_low_dim_similarities(self, embedding: jax.Array) -> jax.Array:
        """Compute low-dimensional similarities (q_ij).

        Uses the UMAP student-t like kernel:
            q(d) = 1 / (1 + a * d^(2b))

        Args:
            embedding: Low-dimensional embedding of shape (n_samples, n_components).

        Returns:
            Similarity matrix of shape (n_samples, n_samples).
        """
        n_samples = embedding.shape[0]

        # Compute pairwise distances in embedding space
        diff = embedding[:, None, :] - embedding[None, :, :]
        distances_sq = jnp.sum(diff**2, axis=-1)

        # UMAP similarity kernel
        a, b = self.embedding_head.curve_coefficients()

        q_ij = 1.0 / (1.0 + a * jnp.power(distances_sq + 1e-8, b))

        # Set diagonal to 0
        q_ij = q_ij * (1 - jnp.eye(n_samples))

        return q_ij

    def _compute_umap_loss(self, p_ij: jax.Array, q_ij: jax.Array) -> jax.Array:
        """Compute UMAP cross-entropy loss.

        Args:
            p_ij: High-dimensional similarities.
            q_ij: Low-dimensional similarities.

        Returns:
            Scalar loss value.
        """
        # Cross-entropy: -sum(p * log(q) + (1-p) * log(1-q))
        eps = 1e-8
        q_ij = jnp.clip(q_ij, eps, 1 - eps)

        # Attractive term
        attractive = -jnp.sum(p_ij * jnp.log(q_ij))

        # Repulsive term
        repulsive = -jnp.sum((1 - p_ij) * jnp.log(1 - q_ij))

        return attractive + repulsive

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict | None,
        random_params: dict | None = None,
        stats: dict | None = None,
    ) -> tuple[dict, dict, dict | None]:
        """Apply UMAP dimensionality reduction.

        Args:
            data: Dictionary containing:
                - 'features': High-dimensional features of shape (n_samples, n_features)
            state: Operator state dictionary.
            metadata: Optional metadata dictionary.
            random_params: Optional random parameters (unused).
            stats: Optional statistics dictionary (unused).

        Returns:
            Tuple of (output_data, state, metadata) where output_data contains:

                - 'features': Original high-dimensional features
                - 'embedding': Low-dimensional embedding
                - 'high_dim_similarities': Fuzzy set memberships (p_ij)
                - 'low_dim_similarities': Embedding similarities (q_ij)
        """
        del random_params, stats  # Unused

        features = data["features"]

        # Project to low-dimensional space
        embedding = self._project(features)

        # Compute similarities
        p_ij = self._compute_high_dim_similarities(features)
        q_ij = self._compute_low_dim_similarities(embedding)

        output_data = {
            **data,
            "embedding": embedding,
            "high_dim_similarities": p_ij,
            "low_dim_similarities": q_ij,
        }

        return output_data, state, metadata
