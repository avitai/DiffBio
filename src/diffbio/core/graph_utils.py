"""Graph utility functions for k-NN graph construction and similarity computation.

Provides reusable, differentiable functions for pairwise distance computation,
k-nearest-neighbor graph construction, fuzzy set membership, and graph
symmetrization. These primitives underpin UMAP, trajectory inference,
imputation, and other graph-based operators.
"""

import jax
import jax.numpy as jnp

__all__ = [
    "compute_pairwise_distances",
    "compute_knn_graph",
    "compute_fuzzy_membership",
    "symmetrize_graph",
]


def compute_pairwise_distances(
    features: jax.Array,
    metric: str = "euclidean",
) -> jax.Array:
    """Compute pairwise distance matrix between all samples.

    Args:
        features: Input feature matrix of shape ``(n_samples, n_features)``.
        metric: Distance metric, either ``"euclidean"`` or ``"cosine"``.

    Returns:
        Distance matrix of shape ``(n_samples, n_samples)`` where entry
        ``(i, j)`` is the distance from sample *i* to sample *j*.
    """
    if metric == "cosine":
        norms = jnp.linalg.norm(features, axis=-1, keepdims=True)
        features_norm = features / (norms + 1e-8)
        similarity = jnp.dot(features_norm, features_norm.T)
        # Clamp to avoid sqrt of negative due to floating-point drift
        distances = jnp.sqrt(jnp.maximum(2.0 * (1.0 - similarity), 0.0) + 1e-8)
    else:
        diff = features[:, None, :] - features[None, :, :]
        distances = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)

    return distances


def compute_knn_graph(
    distances: jax.Array,
    k: int,
) -> tuple[jax.Array, jax.Array]:
    """Build a k-nearest-neighbor graph from a dense distance matrix.

    For each node the *k* closest neighbours (by distance) are selected.
    Self-connections are assumed to already be masked out by setting the
    diagonal to a large value before calling this function.

    Args:
        distances: Dense distance matrix of shape ``(n, n)``.  The diagonal
            should contain large sentinel values (e.g. ``1e10``) so that
            self-loops are never selected.
        k: Number of nearest neighbours per node.  Clipped to ``n - 1``
            when larger than the number of samples minus one.

    Returns:
        A tuple ``(edge_indices, edge_weights)`` where:

        - ``edge_indices`` has shape ``(n * k_eff, 2)`` with each row
          ``[source, target]``.
        - ``edge_weights`` has shape ``(n * k_eff,)`` containing the
          corresponding distances.

        ``k_eff = min(k, n - 1)``.
    """
    n = distances.shape[0]
    k_eff = min(k, n - 1)

    # Argsort each row; first k_eff entries are the nearest neighbours
    sorted_indices = jnp.argsort(distances, axis=-1)
    knn_indices = sorted_indices[:, :k_eff]  # (n, k_eff)

    # Source indices: each node repeated k_eff times
    sources = jnp.repeat(jnp.arange(n), k_eff)  # (n * k_eff,)
    targets = knn_indices.reshape(-1)  # (n * k_eff,)

    edge_indices = jnp.stack([sources, targets], axis=-1)  # (n * k_eff, 2)
    edge_weights = distances[sources, targets]  # (n * k_eff,)

    return edge_indices, edge_weights


def compute_fuzzy_membership(
    distances: jax.Array,
    k: int,
) -> jax.Array:
    """Compute fuzzy set membership using a Gaussian kernel with local bandwidth.

    The bandwidth (sigma) for each sample is set to the distance to its *k*-th
    nearest neighbour, making the kernel adapt to local density.  The diagonal
    of the output is forced to zero (no self-similarity).

    Args:
        distances: Dense distance matrix of shape ``(n, n)``.  The diagonal
            should contain large sentinel values so that self-distances are
            excluded from the bandwidth computation.
        k: Number of neighbours used to determine the local bandwidth.
            Clipped to ``n - 1`` when larger.

    Returns:
        Fuzzy membership matrix of shape ``(n, n)`` with values in ``[0, 1]``.
    """
    n = distances.shape[0]
    k_eff = min(k, n - 1)

    # Local bandwidth: distance to the k-th nearest neighbour
    sorted_dists = jnp.sort(distances, axis=-1)
    sigma = sorted_dists[:, k_eff - 1 : k_eff].squeeze(-1)
    sigma = jnp.maximum(sigma, 1e-8)

    # Gaussian kernel with local bandwidth
    p_ij = jnp.exp(-distances / sigma[:, None])

    # Zero out self-similarity
    p_ij = p_ij * (1.0 - jnp.eye(n))

    return p_ij


def symmetrize_graph(adjacency: jax.Array) -> jax.Array:
    """Symmetrize a directed adjacency matrix via fuzzy set union.

    Applies the probabilistic (fuzzy) union:
    ``p_sym = p + p^T - p * p^T``

    This ensures the output is symmetric and, when inputs are in ``[0, 1]``,
    the outputs remain in ``[0, 1]``.

    Args:
        adjacency: Directed adjacency / membership matrix of shape ``(n, n)``.

    Returns:
        Symmetric adjacency matrix of shape ``(n, n)``.
    """
    return adjacency + adjacency.T - adjacency * adjacency.T
