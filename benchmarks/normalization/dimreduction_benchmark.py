#!/usr/bin/env python3
"""Dimensionality Reduction Benchmark for DiffBio.

Evaluates DiffBio's differentiable dimensionality reduction operators:

- DifferentiableUMAP (parametric UMAP with learnable projection network)
- DifferentiablePHATE (diffusion-based embedding via eigendecomposition)

Metrics:
- Output shape correctness
- Value finiteness and diversity
- Gradient flow through both operators
- Throughput (cells/second)
- Cluster separation in the low-dimensional embedding

Usage:
    python benchmarks/normalization/dimreduction_benchmark.py
    python benchmarks/normalization/dimreduction_benchmark.py --quick
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from flax import nnx

from benchmarks._common import (
    check_gradient_flow,
    generate_synthetic_expression,
    measure_throughput,
    save_benchmark_result,
)
from diffbio.operators.normalization.phate import (
    DifferentiablePHATE,
    PHATEConfig,
)
from diffbio.operators.normalization.umap import (
    DifferentiableUMAP,
    UMAPConfig,
)


# -------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class DimReductionBenchmarkResult:
    """Results from dimensionality reduction benchmark."""

    timestamp: str
    n_cells: int
    n_genes: int
    n_types: int

    # UMAP metrics
    umap_shape_correct: bool
    umap_values_finite: bool
    umap_values_diverse: bool
    umap_gradient_norm: float
    umap_gradient_nonzero: bool
    umap_cells_per_second: float
    umap_cluster_separation: float

    # PHATE metrics
    phate_shape_correct: bool
    phate_values_finite: bool
    phate_values_diverse: bool
    phate_gradient_norm: float
    phate_gradient_nonzero: bool
    phate_cells_per_second: float
    phate_cluster_separation: float

    # Operator configuration summaries
    umap_config: dict[str, Any] = field(default_factory=dict)
    phate_config: dict[str, Any] = field(default_factory=dict)


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
        between_var += count * float(
            jnp.sum((cluster_mean - global_mean) ** 2)
        )

        # Within-cluster: sum of squared distances to cluster mean
        within_var += float(
            jnp.sum((cluster_points - cluster_mean[None, :]) ** 2)
        )

    if within_var < 1e-12:
        return 0.0
    return float(between_var / within_var)


# -------------------------------------------------------------------
# Individual operator benchmarks
# -------------------------------------------------------------------


def _benchmark_umap(
    features: jnp.ndarray,
    cell_type_labels: jnp.ndarray,
    n_types: int,
    n_iterations: int,
) -> dict[str, Any]:
    """Benchmark DifferentiableUMAP.

    Args:
        features: Input feature matrix ``(n_cells, n_features)``.
        cell_type_labels: Integer labels ``(n_cells,)``.
        n_types: Number of cell types.
        n_iterations: Throughput measurement iterations.

    Returns:
        Dictionary of UMAP benchmark metrics.
    """
    n_cells, n_features = features.shape

    config = UMAPConfig(
        n_components=2,
        n_neighbors=min(15, n_cells - 1),
        min_dist=0.1,
        input_features=n_features,
        hidden_dim=min(32, n_features),
    )
    rngs = nnx.Rngs(0)
    umap = DifferentiableUMAP(config, rngs=rngs)

    data: dict[str, Any] = {"features": features}
    result, _, _ = umap.apply(data, {}, None)
    embedding = result["embedding"]

    # Shape check
    expected_shape = (n_cells, config.n_components)
    shape_correct = embedding.shape == expected_shape

    # Finiteness and diversity
    values_finite = bool(jnp.all(jnp.isfinite(embedding)))
    values_diverse = bool(jnp.std(embedding) > 1e-6)

    # Gradient flow
    def umap_loss(model: DifferentiableUMAP, d: dict) -> jnp.ndarray:
        """Scalar loss for gradient checking."""
        out, _, _ = model.apply(d, {}, None)
        return jnp.sum(out["embedding"])

    grad_info = check_gradient_flow(umap_loss, umap, data)

    # Throughput
    throughput_info = measure_throughput(
        fn=lambda d: umap.apply(d, {}, None),
        args=(data,),
        n_iterations=n_iterations,
        warmup=3,
    )
    cells_per_second = n_cells * throughput_info["items_per_sec"]

    # Cluster separation
    separation = compute_cluster_separation(
        embedding, cell_type_labels, n_types
    )

    print("  UMAP results:")
    print(f"    Shape correct:      {shape_correct}")
    print(f"    Values finite:      {values_finite}")
    print(f"    Values diverse:     {values_diverse}")
    print(f"    Gradient norm:      {grad_info.gradient_norm:.6f}")
    print(f"    Gradient nonzero:   {grad_info.gradient_nonzero}")
    print(f"    Cells/sec:          {cells_per_second:.1f}")
    print(f"    Cluster separation: {separation:.4f}")

    return {
        "shape_correct": shape_correct,
        "values_finite": values_finite,
        "values_diverse": values_diverse,
        "gradient_norm": grad_info.gradient_norm,
        "gradient_nonzero": grad_info.gradient_nonzero,
        "cells_per_second": cells_per_second,
        "cluster_separation": separation,
        "config": {
            "n_components": config.n_components,
            "n_neighbors": config.n_neighbors,
            "min_dist": config.min_dist,
            "input_features": config.input_features,
            "hidden_dim": config.hidden_dim,
        },
    }


def _benchmark_phate(
    features: jnp.ndarray,
    cell_type_labels: jnp.ndarray,
    n_types: int,
    n_iterations: int,
) -> dict[str, Any]:
    """Benchmark DifferentiablePHATE.

    Args:
        features: Input feature matrix ``(n_cells, n_features)``.
        cell_type_labels: Integer labels ``(n_cells,)``.
        n_types: Number of cell types.
        n_iterations: Throughput measurement iterations.

    Returns:
        Dictionary of PHATE benchmark metrics.
    """
    n_cells, n_features = features.shape

    config = PHATEConfig(
        n_components=2,
        n_neighbors=min(5, n_cells - 1),
        decay=40.0,
        diffusion_t=10,
        gamma=1.0,
        input_features=n_features,
        hidden_dim=min(32, n_features),
    )
    rngs = nnx.Rngs(0)
    phate = DifferentiablePHATE(config, rngs=rngs)

    data: dict[str, Any] = {"features": features}
    result, _, _ = phate.apply(data, {}, None)
    embedding = result["embedding"]

    # Shape check
    expected_shape = (n_cells, config.n_components)
    shape_correct = embedding.shape == expected_shape

    # Finiteness and diversity
    values_finite = bool(jnp.all(jnp.isfinite(embedding)))
    values_diverse = bool(jnp.std(embedding) > 1e-6)

    # Gradient flow
    def phate_loss(
        model: DifferentiablePHATE, d: dict
    ) -> jnp.ndarray:
        """Scalar loss for gradient checking."""
        out, _, _ = model.apply(d, {}, None)
        return jnp.sum(out["embedding"])

    grad_info = check_gradient_flow(phate_loss, phate, data)

    # Throughput
    throughput_info = measure_throughput(
        fn=lambda d: phate.apply(d, {}, None),
        args=(data,),
        n_iterations=n_iterations,
        warmup=3,
    )
    cells_per_second = n_cells * throughput_info["items_per_sec"]

    # Cluster separation
    separation = compute_cluster_separation(
        embedding, cell_type_labels, n_types
    )

    print("  PHATE results:")
    print(f"    Shape correct:      {shape_correct}")
    print(f"    Values finite:      {values_finite}")
    print(f"    Values diverse:     {values_diverse}")
    print(f"    Gradient norm:      {grad_info.gradient_norm:.6f}")
    print(f"    Gradient nonzero:   {grad_info.gradient_nonzero}")
    print(f"    Cells/sec:          {cells_per_second:.1f}")
    print(f"    Cluster separation: {separation:.4f}")

    return {
        "shape_correct": shape_correct,
        "values_finite": values_finite,
        "values_diverse": values_diverse,
        "gradient_norm": grad_info.gradient_norm,
        "gradient_nonzero": grad_info.gradient_nonzero,
        "cells_per_second": cells_per_second,
        "cluster_separation": separation,
        "config": {
            "n_components": config.n_components,
            "n_neighbors": config.n_neighbors,
            "decay": config.decay,
            "diffusion_t": config.diffusion_t,
            "gamma": config.gamma,
            "input_features": config.input_features,
            "hidden_dim": config.hidden_dim,
        },
    }


# -------------------------------------------------------------------
# Main benchmark entry point
# -------------------------------------------------------------------


def run_benchmark(
    quick: bool = False,
) -> DimReductionBenchmarkResult:
    """Run the dimensionality reduction benchmark.

    Args:
        quick: If True, use smaller data for faster execution.

    Returns:
        Frozen dataclass with all benchmark metrics.
    """
    print("=" * 60)
    print("DiffBio Dimensionality Reduction Benchmark")
    print("=" * 60)

    # Data dimensions
    if quick:
        n_cells, n_genes, n_types = 50, 20, 4
        n_iterations = 10
    else:
        n_cells, n_genes, n_types = 200, 50, 4
        n_iterations = 50

    print(f"\n  Mode:    {'quick' if quick else 'full'}")
    print(f"  Cells:   {n_cells}")
    print(f"  Genes:   {n_genes}")
    print(f"  Types:   {n_types}")

    # Generate synthetic expression data
    print("\nGenerating synthetic expression data...")
    synth = generate_synthetic_expression(
        n_cells=n_cells,
        n_genes=n_genes,
        n_types=n_types,
    )
    features = synth["embeddings"]
    cell_type_labels = synth["cell_type_labels"]

    # Benchmark UMAP
    print("\nBenchmarking DifferentiableUMAP...")
    umap_metrics = _benchmark_umap(
        features, cell_type_labels, n_types, n_iterations
    )

    # Benchmark PHATE
    print("\nBenchmarking DifferentiablePHATE...")
    phate_metrics = _benchmark_phate(
        features, cell_type_labels, n_types, n_iterations
    )

    result = DimReductionBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        n_cells=n_cells,
        n_genes=n_genes,
        n_types=n_types,
        # UMAP
        umap_shape_correct=umap_metrics["shape_correct"],
        umap_values_finite=umap_metrics["values_finite"],
        umap_values_diverse=umap_metrics["values_diverse"],
        umap_gradient_norm=umap_metrics["gradient_norm"],
        umap_gradient_nonzero=umap_metrics["gradient_nonzero"],
        umap_cells_per_second=umap_metrics["cells_per_second"],
        umap_cluster_separation=umap_metrics["cluster_separation"],
        umap_config=umap_metrics["config"],
        # PHATE
        phate_shape_correct=phate_metrics["shape_correct"],
        phate_values_finite=phate_metrics["values_finite"],
        phate_values_diverse=phate_metrics["values_diverse"],
        phate_gradient_norm=phate_metrics["gradient_norm"],
        phate_gradient_nonzero=phate_metrics["gradient_nonzero"],
        phate_cells_per_second=phate_metrics["cells_per_second"],
        phate_cluster_separation=phate_metrics["cluster_separation"],
        phate_config=phate_metrics["config"],
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        f"  UMAP  - grad ok: {result.umap_gradient_nonzero}, "
        f"separation: {result.umap_cluster_separation:.4f}, "
        f"{result.umap_cells_per_second:.0f} cells/s"
    )
    print(
        f"  PHATE - grad ok: {result.phate_gradient_nonzero}, "
        f"separation: {result.phate_cluster_separation:.4f}, "
        f"{result.phate_cells_per_second:.0f} cells/s"
    )
    print("=" * 60)

    return result


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the dimensionality reduction benchmark."""
    quick = "--quick" in sys.argv
    result = run_benchmark(quick=quick)
    output_path = save_benchmark_result(
        result=asdict(result),
        domain="normalization",
        benchmark_name="dimreduction",
        output_dir=Path("benchmarks/results"),
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
