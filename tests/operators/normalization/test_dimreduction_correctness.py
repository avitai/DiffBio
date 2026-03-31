#!/usr/bin/env python3
"""Dimensionality Reduction Benchmark for DiffBio.

Evaluates DiffBio's differentiable dimensionality reduction operators:

- DifferentiableUMAP (parametric UMAP with learnable projection network)
- DifferentiablePHATE (diffusion-based embedding via eigendecomposition)

Metrics:
- Output shape correctness
- Value finiteness and diversity
- Gradient flow through both operators
- Cluster separation in the low-dimensional embedding

Usage:
    python benchmarks/normalization/dimreduction_benchmark.py
    python benchmarks/normalization/dimreduction_benchmark.py --quick
"""

from __future__ import annotations

import jax.numpy as jnp


# -------------------------------------------------------------------
# Cluster separation metric
# -------------------------------------------------------------------


def compute_cluster_separation(
    embedding: jnp.ndarray,
    labels: jnp.ndarray,
    n_types: int,
) -> float:
    """Compute ratio of between-cluster to within-cluster variance.

    A higher ratio indicates better cluster separation in the
    low-dimensional embedding.

    Args:
        embedding: Low-dimensional embedding of shape
            ``(n_cells, n_components)``.
        labels: Integer cell-type labels of shape ``(n_cells,)``.
        n_types: Number of distinct cell types.

    Returns:
        Ratio of between-cluster variance to within-cluster variance.
        Returns 0.0 when within-cluster variance is near zero.
    """
    global_mean = jnp.mean(embedding, axis=0)

    between_var = 0.0
    within_var = 0.0

    for t in range(n_types):
        mask = labels == t
        count = float(jnp.sum(mask))
        if count < 2:
            continue
        cluster_points = embedding[mask]
        cluster_mean = jnp.mean(cluster_points, axis=0)

        # Between-cluster: weighted squared distance of cluster mean
        # from global mean
        between_var += count * float(jnp.sum((cluster_mean - global_mean) ** 2))

        # Within-cluster: sum of squared distances to cluster mean
        within_var += float(jnp.sum((cluster_points - cluster_mean[None, :]) ** 2))

    if within_var < 1e-12:
        return 0.0
    return float(between_var / within_var)
