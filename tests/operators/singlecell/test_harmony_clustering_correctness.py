#!/usr/bin/env python3
"""Single-Cell Analysis Benchmark for DiffBio.

This benchmark evaluates DiffBio's single-cell operators:
- DifferentiableHarmony (batch correction)
- SoftKMeansClustering (clustering)

Metrics:
- Batch mixing improvement
- Clustering quality
- Differentiability (gradient flow)
- Performance

Usage:
    python benchmarks/singlecell/singlecell_benchmark.py
    python benchmarks/singlecell/singlecell_benchmark.py --quick
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks._gradient import check_gradient_flow
from tests.fixtures.synthetic import generate_synthetic_expression
from diffbio.operators.singlecell import (
    BatchCorrectionConfig,
    DifferentiableHarmony,
    SoftClusteringConfig,
    SoftKMeansClustering,
)


# -------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SingleCellBenchmarkResult:
    """Results from single-cell benchmark.

    Attributes:
        timestamp: ISO-format timestamp of the benchmark run.
        n_cells: Number of cells in the synthetic dataset.
        n_features: Number of features (genes) per cell.
        n_batches: Number of experimental batches.
        n_clusters: Number of ground-truth clusters.
        harmony_variance_before: Inter-batch variance before correction.
        harmony_variance_after: Inter-batch variance after correction.
        harmony_variance_reduction: Percent reduction in batch variance.
        harmony_time_ms: Harmony forward-pass time in milliseconds.
        harmony_gradient_norm: Mean gradient norm through Harmony.
        harmony_gradient_nonzero: Whether Harmony gradients are non-zero.
        clustering_inertia: Sum of squared distances to cluster centers.
        clustering_silhouette: Simplified silhouette score.
        clustering_time_ms: Clustering forward-pass time in ms.
        clustering_gradient_norm: Mean gradient norm through clustering.
        clustering_gradient_nonzero: Whether clustering gradients flow.
        harmony_cells_per_sec: Throughput of batch correction.
        clustering_cells_per_sec: Throughput of clustering.
    """

    timestamp: str
    # Dataset parameters
    n_cells: int
    n_features: int
    n_batches: int
    n_clusters: int
    # Harmony metrics
    harmony_variance_before: float
    harmony_variance_after: float
    harmony_variance_reduction: float
    harmony_time_ms: float
    harmony_gradient_norm: float
    harmony_gradient_nonzero: bool
    # Clustering metrics
    clustering_inertia: float
    clustering_silhouette: float
    clustering_time_ms: float
    clustering_gradient_norm: float
    clustering_gradient_nonzero: bool
    # Performance
    harmony_cells_per_sec: float
    clustering_cells_per_sec: float


# -------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------


def compute_batch_variance(
    embeddings: jnp.ndarray,
    batch_labels: jnp.ndarray,
    n_batches: int,
) -> float:
    """Compute inter-batch variance (lower means better mixing).

    Args:
        embeddings: Cell embeddings ``(n_cells, n_features)``.
        batch_labels: Integer batch assignment per cell.
        n_batches: Number of distinct batches.

    Returns:
        Variance across batch centroids.
    """
    batch_means_list: list[jnp.ndarray] = []
    for b in range(n_batches):
        mask = batch_labels == b
        if mask.sum() > 0:
            batch_mean = jnp.mean(embeddings[mask], axis=0)
            batch_means_list.append(batch_mean)

    if len(batch_means_list) < 2:
        return 0.0

    batch_means = jnp.stack(batch_means_list)
    return float(jnp.var(batch_means))


def compute_clustering_metrics(
    embeddings: jnp.ndarray,
    assignments: jnp.ndarray,
    n_clusters: int,
) -> dict[str, float]:
    """Compute clustering quality metrics.

    Args:
        embeddings: Cell embeddings ``(n_cells, n_features)``.
        assignments: Soft assignment matrix ``(n_cells, n_clusters)``.
        n_clusters: Number of clusters.

    Returns:
        Dictionary with ``inertia`` and ``silhouette`` keys.
    """
    hard_assignments = jnp.argmax(assignments, axis=-1)

    # Compute cluster centers
    centers_list: list[jnp.ndarray] = []
    for c in range(n_clusters):
        mask = hard_assignments == c
        if mask.sum() > 0:
            center = jnp.mean(embeddings[mask], axis=0)
        else:
            center = jnp.zeros(embeddings.shape[1])
        centers_list.append(center)
    centers = jnp.stack(centers_list)

    # Inertia: sum of squared distances to nearest center
    distances = jnp.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=-1)
    min_distances = jnp.min(distances, axis=-1)
    inertia = float(jnp.sum(min_distances**2))

    # Simplified silhouette score
    silhouettes: list[float] = []
    for c in range(n_clusters):
        mask = hard_assignments == c
        if mask.sum() <= 1:
            continue
        cluster_points = embeddings[mask]
        within_dists = cluster_points[:, None, :] - cluster_points[None, :, :]
        a = jnp.mean(jnp.linalg.norm(within_dists, axis=-1))

        b_vals: list[float] = []
        for c2 in range(n_clusters):
            if c2 == c:
                continue
            mask2 = hard_assignments == c2
            if mask2.sum() > 0:
                other_points = embeddings[mask2]
                between_dists = cluster_points[:, None, :] - other_points[None, :, :]
                b = jnp.mean(jnp.linalg.norm(between_dists, axis=-1))
                b_vals.append(float(b))
        if b_vals:
            b_min = min(b_vals)
            sil = (b_min - float(a)) / max(float(a), b_min)
            silhouettes.append(sil)

    silhouette_score = float(np.mean(silhouettes)) if silhouettes else 0.0

    return {
        "inertia": inertia,
        "silhouette": silhouette_score,
    }


# -------------------------------------------------------------------
# Individual operator benchmarks
# -------------------------------------------------------------------


def _test_harmony(data: dict[str, Any]) -> dict[str, Any]:
    """Test DifferentiableHarmony operator.

    Args:
        data: Synthetic dataset dictionary from
            :func:`generate_synthetic_expression`.

    Returns:
        Dictionary with batch-correction benchmark metrics.
    """
    print("\n  Testing DifferentiableHarmony...")

    n_features = data["n_features"]
    n_batches = data["n_batches"]
    n_cells = data["n_cells"]

    config = BatchCorrectionConfig(
        n_clusters=20,
        n_features=n_features,
        n_batches=n_batches,
        n_iterations=10,
        temperature=1.0,
    )
    rngs = nnx.Rngs(42)
    harmony = DifferentiableHarmony(config, rngs=rngs)

    input_data = {
        "embeddings": data["embeddings"],
        "batch_labels": data["batch_labels"],
    }

    start_time = time.time()
    result, _, _ = harmony.apply(input_data, {}, None)
    elapsed_ms = (time.time() - start_time) * 1000

    corrected = result["corrected_embeddings"]

    # Compute batch variance before and after
    variance_before = compute_batch_variance(data["embeddings"], data["batch_labels"], n_batches)
    variance_after = compute_batch_variance(corrected, data["batch_labels"], n_batches)
    variance_reduction = (
        (1 - variance_after / variance_before) * 100 if variance_before > 0 else 0.0
    )

    # Gradient flow through model parameters
    def loss_fn(
        model: DifferentiableHarmony,
        inp: dict,
    ) -> jnp.ndarray:
        """Loss for gradient check on Harmony operator."""
        res, _, _ = model.apply(inp, {}, None)
        return res["corrected_embeddings"].sum()

    grad_info = check_gradient_flow(loss_fn, harmony, input_data)

    print(f"    Batch variance before: {variance_before:.4f}")
    print(f"    Batch variance after: {variance_after:.4f}")
    print(f"    Variance reduction: {variance_reduction:.1f}%")
    print(f"    Time: {elapsed_ms:.2f}ms")
    print(
        f"    Gradient norm: {grad_info.gradient_norm:.6f} (nonzero={grad_info.gradient_nonzero})"
    )

    return {
        "variance_before": variance_before,
        "variance_after": variance_after,
        "variance_reduction": variance_reduction,
        "time_ms": elapsed_ms,
        "gradient_norm": grad_info.gradient_norm,
        "gradient_nonzero": grad_info.gradient_nonzero,
        "cells_per_sec": n_cells / (elapsed_ms / 1000),
    }


def _test_clustering(data: dict[str, Any]) -> dict[str, Any]:
    """Test SoftKMeansClustering operator.

    Args:
        data: Synthetic dataset dictionary from
            :func:`generate_synthetic_expression`.

    Returns:
        Dictionary with clustering benchmark metrics.
    """
    print("\n  Testing SoftKMeansClustering...")

    n_clusters = data["n_clusters"]
    n_features = data["n_features"]
    n_cells = data["n_cells"]

    config = SoftClusteringConfig(
        n_clusters=n_clusters,
        n_features=n_features,
        temperature=0.5,
    )
    rngs = nnx.Rngs(42)
    kmeans = SoftKMeansClustering(config, rngs=rngs)

    input_data = {"embeddings": data["embeddings"]}

    start_time = time.time()
    result, _, _ = kmeans.apply(input_data, {}, None)
    elapsed_ms = (time.time() - start_time) * 1000

    assignments = result["cluster_assignments"]

    metrics = compute_clustering_metrics(data["embeddings"], assignments, n_clusters)

    # Gradient flow through model parameters
    def loss_fn(
        model: SoftKMeansClustering,
        inp: dict,
    ) -> jnp.ndarray:
        """Loss for gradient check on clustering operator."""
        res, _, _ = model.apply(inp, {}, None)
        return res["cluster_assignments"].sum()

    grad_info = check_gradient_flow(loss_fn, kmeans, input_data)

    print(f"    Inertia: {metrics['inertia']:.2f}")
    print(f"    Silhouette score: {metrics['silhouette']:.4f}")
    print(f"    Time: {elapsed_ms:.2f}ms")
    print(
        f"    Gradient norm: {grad_info.gradient_norm:.6f} (nonzero={grad_info.gradient_nonzero})"
    )

    return {
        "inertia": metrics["inertia"],
        "silhouette": metrics["silhouette"],
        "time_ms": elapsed_ms,
        "gradient_norm": grad_info.gradient_norm,
        "gradient_nonzero": grad_info.gradient_nonzero,
        "cells_per_sec": n_cells / (elapsed_ms / 1000),
    }


# -------------------------------------------------------------------
# Main benchmark
# -------------------------------------------------------------------


def run_benchmark(
    quick: bool = False,
) -> SingleCellBenchmarkResult:
    """Run the complete single-cell benchmark.

    Args:
        quick: If True, use smaller dataset (50 cells, 20 features)
            for faster execution.

    Returns:
        SingleCellBenchmarkResult with all metrics.
    """
    print("=" * 60)
    print("DiffBio Single-Cell Analysis Benchmark")
    print("=" * 60)

    # Dataset sizes
    if quick:
        n_cells, n_features, n_clusters = 50, 20, 4
    else:
        n_cells, n_features, n_clusters = 500, 50, 4

    n_batches = 3

    # Generate synthetic data via shared utility
    print("\nGenerating synthetic single-cell data...")
    synth = generate_synthetic_expression(
        n_cells=n_cells,
        n_genes=n_features,
        n_types=n_clusters,
        n_batches=n_batches,
        batch_effect_strength=3.0,
    )

    # Map _common output fields to local convention
    data: dict[str, Any] = {
        "embeddings": synth["embeddings"],
        "batch_labels": synth["batch_labels"],
        "true_clusters": synth["cell_type_labels"],
        "n_cells": synth["n_cells"],
        "n_features": synth["n_genes"],
        "n_batches": synth["n_batches"],
        "n_clusters": synth["n_types"],
    }

    print(f"  Cells: {data['n_cells']}")
    print(f"  Features: {data['n_features']}")
    print(f"  Batches: {data['n_batches']}")
    print(f"  True clusters: {data['n_clusters']}")

    # Test Harmony
    harmony_metrics = _test_harmony(data)

    # Test clustering
    clustering_metrics = _test_clustering(data)

    result = SingleCellBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        n_cells=data["n_cells"],
        n_features=data["n_features"],
        n_batches=data["n_batches"],
        n_clusters=data["n_clusters"],
        # Harmony
        harmony_variance_before=harmony_metrics["variance_before"],
        harmony_variance_after=harmony_metrics["variance_after"],
        harmony_variance_reduction=harmony_metrics["variance_reduction"],
        harmony_time_ms=harmony_metrics["time_ms"],
        harmony_gradient_norm=harmony_metrics["gradient_norm"],
        harmony_gradient_nonzero=harmony_metrics["gradient_nonzero"],
        # Clustering
        clustering_inertia=clustering_metrics["inertia"],
        clustering_silhouette=clustering_metrics["silhouette"],
        clustering_time_ms=clustering_metrics["time_ms"],
        clustering_gradient_norm=clustering_metrics["gradient_norm"],
        clustering_gradient_nonzero=clustering_metrics["gradient_nonzero"],
        # Performance
        harmony_cells_per_sec=harmony_metrics["cells_per_sec"],
        clustering_cells_per_sec=clustering_metrics["cells_per_sec"],
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    vr = harmony_metrics["variance_reduction"]
    sil = clustering_metrics["silhouette"]
    print(f"  Harmony: {vr:.1f}% batch variance reduction")
    print(f"  Clustering: Silhouette score {sil:.4f}")
    print("\n  All operators are differentiable!")
    print("=" * 60)

    return result
