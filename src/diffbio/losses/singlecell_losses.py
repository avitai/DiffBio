"""Single-cell specific loss functions for differentiable bioinformatics.

This module provides differentiable loss functions for single-cell analysis
pipelines, including batch correction, clustering, and RNA velocity.

Includes:
- BatchMixingLoss: Maximizes batch mixing in latent space
- ClusteringCompactnessLoss: Encourages tight, well-separated clusters
- VelocityConsistencyLoss: Enforces consistency between velocity and trajectory
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int


class BatchMixingLoss(nnx.Module):
    """Loss function to maximize batch mixing in latent space.

    Computes how well batches are mixed in the embedding space by measuring
    the entropy of batch labels among k-nearest neighbors for each cell.
    Higher entropy indicates better mixing.

    The loss encourages the model to learn representations where cells from
    different batches are interleaved, reducing batch effects.

    Args:
        n_neighbors: Number of nearest neighbors to consider.
        n_batches: Number of batches (required for JIT compatibility).
        temperature: Temperature for softmax in distance computation.
        rngs: Flax NNX random number generators.

    Example:
        ```python
        loss_fn = BatchMixingLoss(n_neighbors=15, n_batches=3, rngs=nnx.Rngs(42))
        loss = loss_fn(embeddings, batch_labels)
        ```
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        n_batches: int = 3,
        temperature: float = 1.0,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the batch mixing loss.

        Args:
            n_neighbors: Number of nearest neighbors to consider.
            n_batches: Number of batches (static for JIT compatibility).
            temperature: Temperature for soft neighbor selection.
            rngs: Random number generators (not used, for API consistency).
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.n_batches = n_batches
        self.temperature = temperature
        # Precompute max entropy for normalization (static constant)
        self._max_entropy = jnp.log(jnp.array(n_batches, dtype=jnp.float32))

    def __call__(
        self,
        embeddings: Float[Array, "n_cells latent_dim"],
        batch_labels: Int[Array, "n_cells"],
    ) -> Float[Array, ""]:
        """Compute batch mixing loss.

        Args:
            embeddings: Cell embeddings in latent space.
            batch_labels: Integer batch label for each cell.

        Returns:
            Negative mean entropy of batch distribution in neighborhoods (scalar).
            Lower loss means better mixing.
        """
        n_cells = embeddings.shape[0]
        n_batches = self.n_batches

        # Compute pairwise distances
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
        sq_norms = jnp.sum(embeddings**2, axis=-1)
        distances = sq_norms[:, None] + sq_norms[None, :] - 2 * embeddings @ embeddings.T

        # Set self-distance to inf to exclude self from neighbors
        distances = distances + jnp.eye(n_cells) * 1e10

        # Soft neighbor weights using softmax over negative distances
        neighbor_weights = jax.nn.softmax(-distances / self.temperature, axis=-1)

        # Keep only top k neighbors (soft selection)
        # Sort to get top-k, then create soft mask
        sorted_indices = jnp.argsort(distances, axis=-1)
        k_mask = jnp.zeros((n_cells, n_cells))
        k_mask = k_mask.at[
            jnp.arange(n_cells)[:, None],
            sorted_indices[:, : self.n_neighbors],
        ].set(1.0)

        # Apply mask
        neighbor_weights = neighbor_weights * k_mask
        neighbor_weights = neighbor_weights / (neighbor_weights.sum(axis=-1, keepdims=True) + 1e-8)

        # One-hot encode batch labels
        batch_onehot = jax.nn.one_hot(batch_labels, n_batches)

        # Compute batch distribution in neighborhoods
        # For each cell, weighted average of neighbor batch labels
        batch_dist = neighbor_weights @ batch_onehot  # (n_cells, n_batches)

        # Compute entropy of batch distribution
        # H = -sum(p * log(p))
        eps = 1e-8
        entropy = -jnp.sum(batch_dist * jnp.log(batch_dist + eps), axis=-1)

        # Normalized entropy (0 to 1, higher is better)
        # Use precomputed max_entropy for JIT compatibility
        normalized_entropy = entropy / (self._max_entropy + eps)

        # Return negative mean entropy (lower loss = better mixing)
        return -jnp.mean(normalized_entropy)


class ClusteringCompactnessLoss(nnx.Module):
    """Loss function to encourage compact and well-separated clusters.

    Combines two components:
    1. Compactness: Minimize within-cluster variance
    2. Separation: Maximize between-cluster distances

    Works with soft cluster assignments for end-to-end differentiability.

    Args:
        separation_weight: Weight for the separation term.
        min_separation: Minimum desired distance between cluster centers.
        rngs: Flax NNX random number generators.

    Example:
        ```python
        loss_fn = ClusteringCompactnessLoss(rngs=nnx.Rngs(42))
        loss = loss_fn(embeddings, soft_assignments)
        ```
    """

    def __init__(
        self,
        separation_weight: float = 1.0,
        min_separation: float = 1.0,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the clustering compactness loss.

        Args:
            separation_weight: Weight for separation term.
            min_separation: Minimum desired distance between centroids.
            rngs: Random number generators (not used, for API consistency).
        """
        super().__init__()
        self.separation_weight = separation_weight
        self.min_separation = min_separation

    def __call__(
        self,
        embeddings: Float[Array, "n_cells latent_dim"],
        assignments: Float[Array, "n_cells n_clusters"],
        centroids: Float[Array, "n_clusters latent_dim"] | None = None,
    ) -> Float[Array, ""]:
        """Compute clustering compactness loss.

        Args:
            embeddings: Cell embeddings in latent space.
            assignments: Soft cluster assignments (should sum to 1 per cell).
            centroids: Optional cluster centroids. If provided, uses these directly
                for gradient flow. If None, computes soft centroids from assignments.

        Returns:
            Combined compactness and separation loss (scalar).
        """
        n_clusters = assignments.shape[1]

        # Use provided centroids or compute soft centroids from assignments
        if centroids is None:
            # Compute soft cluster centroids
            # centroid_k = sum_i(assignment_ik * embedding_i) / sum_i(assignment_ik)
            assignment_sums = assignments.sum(axis=0, keepdims=True).T  # (n_clusters, 1)
            centroids = (assignments.T @ embeddings) / (
                assignment_sums + 1e-8
            )  # (n_clusters, latent_dim)

        # Compactness: weighted sum of squared distances to centroids
        # For each cell, compute distance to each centroid
        # Then weight by assignment
        distances_to_centroids = jnp.sum(
            (embeddings[:, None, :] - centroids[None, :, :]) ** 2, axis=-1
        )  # (n_cells, n_clusters)

        # Weighted compactness
        compactness = jnp.sum(assignments * distances_to_centroids) / embeddings.shape[0]

        # Separation: pairwise distances between centroids
        # We want centroids to be at least min_separation apart
        centroid_dists = jnp.sqrt(
            jnp.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=-1) + 1e-8
        )  # (n_clusters, n_clusters)

        # Hinge loss: penalize if distance < min_separation
        # Exclude diagonal (self-distance)
        mask = 1.0 - jnp.eye(n_clusters)
        separation_violations = jax.nn.relu(self.min_separation - centroid_dists) * mask

        # Mean separation loss (excluding diagonal)
        n_pairs = n_clusters * (n_clusters - 1)
        separation_loss = jnp.sum(separation_violations) / (n_pairs + 1e-8)

        # Combined loss
        return compactness + self.separation_weight * separation_loss


class VelocityConsistencyLoss(nnx.Module):
    """Loss function to enforce consistency between velocity and trajectory.

    Ensures that the predicted RNA velocity is consistent with actual
    expression changes over time. Combines directional (cosine) and
    magnitude consistency.

    Args:
        dt: Time step for velocity extrapolation.
        cosine_weight: Weight for directional consistency.
        magnitude_weight: Weight for magnitude consistency.
        rngs: Flax NNX random number generators.

    Example:
        ```python
        loss_fn = VelocityConsistencyLoss(rngs=nnx.Rngs(42))
        loss = loss_fn(expression, velocity, future_expression)
        ```
    """

    def __init__(
        self,
        dt: float = 0.1,
        cosine_weight: float = 1.0,
        magnitude_weight: float = 1.0,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the velocity consistency loss.

        Args:
            dt: Time step for velocity extrapolation.
            cosine_weight: Weight for cosine similarity loss.
            magnitude_weight: Weight for magnitude loss.
            rngs: Random number generators (not used, for API consistency).
        """
        super().__init__()
        self.dt = dt
        self.cosine_weight = cosine_weight
        self.magnitude_weight = magnitude_weight

    def __call__(
        self,
        expression: Float[Array, "n_cells n_genes"],
        velocity: Float[Array, "n_cells n_genes"],
        future_expression: Float[Array, "n_cells n_genes"],
    ) -> Float[Array, ""]:
        """Compute velocity consistency loss.

        Args:
            expression: Current gene expression.
            velocity: Predicted RNA velocity (rate of change).
            future_expression: Future gene expression (ground truth or estimated).

        Returns:
            Combined directional and magnitude consistency loss (scalar).
        """
        # Predicted change based on velocity
        predicted_delta = velocity * self.dt

        # Actual change
        actual_delta = future_expression - expression

        # Cosine similarity loss (directional consistency)
        # cosine_sim = (a . b) / (||a|| * ||b||)
        eps = 1e-8
        pred_norm = jnp.sqrt(jnp.sum(predicted_delta**2, axis=-1) + eps)
        actual_norm = jnp.sqrt(jnp.sum(actual_delta**2, axis=-1) + eps)
        dot_product = jnp.sum(predicted_delta * actual_delta, axis=-1)

        cosine_sim = dot_product / (pred_norm * actual_norm)

        # Cosine loss: 1 - cosine_sim (ranges from 0 to 2)
        cosine_loss = jnp.mean(1 - cosine_sim)

        # Magnitude loss: MSE between predicted and actual delta magnitudes
        magnitude_loss = jnp.mean((pred_norm - actual_norm) ** 2)

        # Combined loss
        return self.cosine_weight * cosine_loss + self.magnitude_weight * magnitude_loss
