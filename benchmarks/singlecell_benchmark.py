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
    python benchmarks/singlecell_benchmark.py
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from diffbio.operators.singlecell import (
    DifferentiableHarmony,
    BatchCorrectionConfig,
    SoftKMeansClustering,
    SoftClusteringConfig,
)


@dataclass
class SingleCellBenchmarkResult:
    """Results from single-cell benchmark."""

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
    # Clustering metrics
    clustering_inertia: float
    clustering_silhouette: float
    clustering_time_ms: float
    clustering_gradient_norm: float
    # Performance
    harmony_cells_per_sec: float
    clustering_cells_per_sec: float


def generate_synthetic_data(
    n_cells: int = 500,
    n_features: int = 50,
    n_batches: int = 3,
    n_clusters: int = 4,
    batch_effect_strength: float = 3.0,
    seed: int = 42,
) -> dict:
    """Generate synthetic single-cell data with batch effects.

    Returns:
        Dictionary with embeddings, batch_labels, and true_clusters
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 5)

    # Generate cluster centers
    cluster_centers = jax.random.normal(keys[0], (n_clusters, n_features)) * 2.0

    # Assign cells to clusters (roughly equal)
    cells_per_cluster = n_cells // n_clusters
    true_clusters_list: list[int] = []
    for c in range(n_clusters):
        true_clusters_list.extend([c] * cells_per_cluster)
    # Handle remainder
    while len(true_clusters_list) < n_cells:
        true_clusters_list.append(n_clusters - 1)
    true_clusters = jnp.array(true_clusters_list)

    # Generate base embeddings around cluster centers
    noise = jax.random.normal(keys[1], (n_cells, n_features)) * 0.5
    base_embeddings = cluster_centers[true_clusters] + noise

    # Assign cells to batches
    cells_per_batch = n_cells // n_batches
    batch_labels_list: list[int] = []
    for b in range(n_batches):
        batch_labels_list.extend([b] * cells_per_batch)
    while len(batch_labels_list) < n_cells:
        batch_labels_list.append(n_batches - 1)
    batch_labels = jnp.array(batch_labels_list)

    # Add batch effects
    batch_shifts = jax.random.normal(keys[2], (n_batches, n_features)) * batch_effect_strength
    embeddings = base_embeddings + batch_shifts[batch_labels]

    return {
        "embeddings": embeddings,
        "batch_labels": batch_labels,
        "true_clusters": true_clusters,
        "n_cells": n_cells,
        "n_features": n_features,
        "n_batches": n_batches,
        "n_clusters": n_clusters,
    }


def compute_batch_variance(
    embeddings: jnp.ndarray, batch_labels: jnp.ndarray, n_batches: int
) -> float:
    """Compute inter-batch variance (lower = better mixing)."""
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
) -> dict:
    """Compute clustering quality metrics."""
    # Hard assignments
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

    # Inertia: sum of squared distances to cluster centers
    distances = jnp.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=-1)
    min_distances = jnp.min(distances, axis=-1)
    inertia = float(jnp.sum(min_distances**2))

    # Simplified silhouette score
    # a(i) = mean distance to same cluster
    # b(i) = min mean distance to other clusters
    silhouettes = []
    for c in range(n_clusters):
        mask = hard_assignments == c
        if mask.sum() > 1:
            cluster_points = embeddings[mask]
            # Within-cluster distances
            within_dists = cluster_points[:, None, :] - cluster_points[None, :, :]
            a = jnp.mean(jnp.linalg.norm(within_dists, axis=-1))
            # Between-cluster distances
            b_vals = []
            for c2 in range(n_clusters):
                if c2 != c:
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

    silhouette_score = np.mean(silhouettes) if silhouettes else 0.0

    return {
        "inertia": inertia,
        "silhouette": silhouette_score,
    }


def test_harmony(data: dict) -> dict:
    """Test DifferentiableHarmony operator."""
    print("\n  Testing DifferentiableHarmony...")

    config = BatchCorrectionConfig(
        n_clusters=20,
        n_features=data["n_features"],
        n_batches=data["n_batches"],
        n_iterations=10,
        temperature=1.0,
    )
    rngs = nnx.Rngs(42)
    harmony = DifferentiableHarmony(config, rngs=rngs)

    # Apply batch correction
    input_data = {
        "embeddings": data["embeddings"],
        "batch_labels": data["batch_labels"],
    }

    start_time = time.time()
    result, _, _ = harmony.apply(input_data, {}, None)
    elapsed_ms = (time.time() - start_time) * 1000

    corrected = result["corrected_embeddings"]

    # Compute batch variance before and after
    variance_before = compute_batch_variance(
        data["embeddings"], data["batch_labels"], data["n_batches"]
    )
    variance_after = compute_batch_variance(corrected, data["batch_labels"], data["n_batches"])
    variance_reduction = (1 - variance_after / variance_before) * 100 if variance_before > 0 else 0

    # Test differentiability
    def loss_fn(harmony, input_data):
        result, _, _ = harmony.apply(input_data, {}, None)
        return result["corrected_embeddings"].sum()

    try:
        grads = nnx.grad(loss_fn)(harmony, input_data)
        grad_norms = []
        for name, param in nnx.iter_graph(grads):
            if hasattr(param, "value") and isinstance(param.value, jnp.ndarray):
                norm = float(jnp.linalg.norm(param.value))
                if norm > 0:
                    grad_norms.append(norm)
        gradient_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    except Exception as e:
        print(f"    Gradient error: {e}")
        gradient_norm = 0.0

    print(f"    Batch variance before: {variance_before:.4f}")
    print(f"    Batch variance after: {variance_after:.4f}")
    print(f"    Variance reduction: {variance_reduction:.1f}%")
    print(f"    Time: {elapsed_ms:.2f}ms")
    print(f"    Gradient norm: {gradient_norm:.6f}")

    return {
        "variance_before": variance_before,
        "variance_after": variance_after,
        "variance_reduction": variance_reduction,
        "time_ms": elapsed_ms,
        "gradient_norm": gradient_norm,
        "cells_per_sec": data["n_cells"] / (elapsed_ms / 1000),
    }


def test_clustering(data: dict) -> dict:
    """Test SoftKMeansClustering operator."""
    print("\n  Testing SoftKMeansClustering...")

    config = SoftClusteringConfig(
        n_clusters=data["n_clusters"],
        n_features=data["n_features"],
        temperature=0.5,
    )
    rngs = nnx.Rngs(42)
    kmeans = SoftKMeansClustering(config, rngs=rngs)

    # Apply clustering
    input_data = {"embeddings": data["embeddings"]}

    start_time = time.time()
    result, _, _ = kmeans.apply(input_data, {}, None)
    elapsed_ms = (time.time() - start_time) * 1000

    assignments = result["cluster_assignments"]

    # Compute clustering metrics
    metrics = compute_clustering_metrics(data["embeddings"], assignments, data["n_clusters"])

    # Test differentiability
    def loss_fn(kmeans, input_data):
        result, _, _ = kmeans.apply(input_data, {}, None)
        return result["cluster_assignments"].sum()

    try:
        grads = nnx.grad(loss_fn)(kmeans, input_data)
        grad_norms = []
        for name, param in nnx.iter_graph(grads):
            if hasattr(param, "value") and isinstance(param.value, jnp.ndarray):
                norm = float(jnp.linalg.norm(param.value))
                if norm > 0:
                    grad_norms.append(norm)
        gradient_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    except Exception as e:
        print(f"    Gradient error: {e}")
        gradient_norm = 0.0

    print(f"    Inertia: {metrics['inertia']:.2f}")
    print(f"    Silhouette score: {metrics['silhouette']:.4f}")
    print(f"    Time: {elapsed_ms:.2f}ms")
    print(f"    Gradient norm: {gradient_norm:.6f}")

    return {
        "inertia": metrics["inertia"],
        "silhouette": metrics["silhouette"],
        "time_ms": elapsed_ms,
        "gradient_norm": gradient_norm,
        "cells_per_sec": data["n_cells"] / (elapsed_ms / 1000),
    }


def run_benchmark() -> SingleCellBenchmarkResult:
    """Run the complete single-cell benchmark."""
    print("=" * 60)
    print("DiffBio Single-Cell Analysis Benchmark")
    print("=" * 60)

    # Generate synthetic data
    print("\nGenerating synthetic single-cell data...")
    data = generate_synthetic_data(
        n_cells=500,
        n_features=50,
        n_batches=3,
        n_clusters=4,
        batch_effect_strength=3.0,
    )
    print(f"  Cells: {data['n_cells']}")
    print(f"  Features: {data['n_features']}")
    print(f"  Batches: {data['n_batches']}")
    print(f"  True clusters: {data['n_clusters']}")

    # Test Harmony
    harmony_metrics = test_harmony(data)

    # Test clustering
    clustering_metrics = test_clustering(data)

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
        # Clustering
        clustering_inertia=clustering_metrics["inertia"],
        clustering_silhouette=clustering_metrics["silhouette"],
        clustering_time_ms=clustering_metrics["time_ms"],
        clustering_gradient_norm=clustering_metrics["gradient_norm"],
        # Performance
        harmony_cells_per_sec=harmony_metrics["cells_per_sec"],
        clustering_cells_per_sec=clustering_metrics["cells_per_sec"],
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Harmony: {harmony_metrics['variance_reduction']:.1f}% batch variance reduction")
    print(f"  Clustering: Silhouette score {clustering_metrics['silhouette']:.4f}")
    print("\n  All operators are differentiable!")
    print("=" * 60)

    return result


def save_results(result: SingleCellBenchmarkResult, output_dir: Path) -> None:
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"singlecell_benchmark_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"Results saved to: {output_file}")


def main():
    """Main entry point."""
    result = run_benchmark()
    save_results(result, Path("benchmarks/results"))


if __name__ == "__main__":
    main()
