"""Differentiable UMAP dimensionality reduction.

This module implements a differentiable version of UMAP (Uniform Manifold
Approximation and Projection) for dimensionality reduction with end-to-end
gradient flow.
"""

from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule


@dataclass
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
        self.config = config

        if rngs is None:
            rngs = nnx.Rngs(0)

        # Initialize UMAP curve parameters a and b
        # These control the shape of the low-dimensional similarity function
        # q(d) = 1 / (1 + a * d^(2b))
        # Default values from original UMAP: a ≈ 1.929, b ≈ 0.7915
        self.a_param = nnx.Param(jnp.array(1.929))
        self.b_param = nnx.Param(jnp.array(0.7915))

        # Initialize projection network (parametric UMAP) using nnx.Linear
        self.projection_layer1 = nnx.Linear(
            in_features=config.input_features,
            out_features=config.hidden_dim,
            rngs=rngs,
        )
        self.projection_layer2 = nnx.Linear(
            in_features=config.hidden_dim,
            out_features=config.n_components,
            rngs=rngs,
        )

    def _project(self, features: jax.Array) -> jax.Array:
        """Project high-dimensional features to low-dimensional embedding.

        Args:
            features: Input features of shape (n_samples, n_features).

        Returns:
            Embedding of shape (n_samples, n_components).
        """
        # Simple 2-layer MLP with ReLU activation
        h = self.projection_layer1(features)
        h = nnx.relu(h)
        embedding = self.projection_layer2(h)

        return embedding

    def _compute_high_dim_similarities(self, features: jax.Array) -> jax.Array:
        """Compute high-dimensional fuzzy set membership (p_ij).

        Uses k-nearest neighbors and Gaussian kernel with local bandwidth.

        Args:
            features: Input features of shape (n_samples, n_features).

        Returns:
            Similarity matrix of shape (n_samples, n_samples).
        """
        n_samples = features.shape[0]
        n_neighbors = min(self.config.n_neighbors, n_samples - 1)

        # Compute pairwise distances
        if self.config.metric == "cosine":
            # Normalize features for cosine distance
            features_norm = features / (jnp.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
            # Cosine similarity -> distance
            sim = jnp.dot(features_norm, features_norm.T)
            distances = jnp.sqrt(2 * (1 - sim + 1e-8))
        else:
            # Euclidean distance
            diff = features[:, None, :] - features[None, :, :]
            distances = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)

        # Set diagonal to large value to exclude self-connections
        distances = distances + jnp.eye(n_samples) * 1e10

        # Find local bandwidth (distance to k-th neighbor) using soft top-k
        # Sort distances and take k-th smallest
        sorted_dists = jnp.sort(distances, axis=-1)
        sigma = sorted_dists[:, n_neighbors - 1 : n_neighbors].squeeze(-1)
        sigma = jnp.maximum(sigma, 1e-8)

        # Compute fuzzy membership using Gaussian kernel
        # p_ij = exp(-max(0, d_ij - rho_i) / sigma_i)
        # Simplified: use sigma as bandwidth
        p_ij = jnp.exp(-distances / sigma[:, None])

        # Set diagonal to 0
        p_ij = p_ij * (1 - jnp.eye(n_samples))

        # Symmetrize: p_sym = p_ij + p_ji - p_ij * p_ji
        p_sym = p_ij + p_ij.T - p_ij * p_ij.T

        return p_sym

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
        a = jnp.abs(self.a_param[...]) + 1e-6
        b = jnp.abs(self.b_param[...]) + 1e-6

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
