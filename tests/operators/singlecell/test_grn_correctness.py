#!/usr/bin/env python3
"""GRN Inference and Cell Communication Benchmark for DiffBio.

This benchmark evaluates DiffBio's gene regulatory network inference and
cell-cell communication operators:

- DifferentiableGRN: GATv2-based GRN inference from scRNA-seq data
- SINDyGRNOperator: SINDy-based GRN inference from time-series expression
- DifferentiableCellCommunication: GNN-based spatial cell communication

Metrics:
- Output shape and finiteness verification
- GRN sparsity analysis
- Gradient flow through all operators
- Throughput measurement

Usage:
    python benchmarks/singlecell/grn_benchmark.py
    python benchmarks/singlecell/grn_benchmark.py --quick
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._gradient import check_gradient_flow
from diffbio.operators.singlecell.communication import (
    CellCommunicationConfig,
    DifferentiableCellCommunication,
)
from diffbio.operators.singlecell.grn_inference import (
    DifferentiableGRN,
    GRNInferenceConfig,
)
from diffbio.operators.singlecell.sindy_grn import (
    SINDyGRNConfig,
    SINDyGRNOperator,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class GRNBenchmarkResult:
    """Results from GRN inference and cell communication benchmark."""

    timestamp: str

    # GRN operator metrics
    grn_matrix_shape: tuple[int, ...]
    grn_matrix_finite: bool
    grn_tf_activity_shape: tuple[int, ...]
    grn_gradient_norm: float
    grn_gradient_nonzero: bool
    grn_throughput_items_per_sec: float

    # SINDy operator metrics
    sindy_coefficients_shape: tuple[int, ...]
    sindy_coefficients_finite: bool
    sindy_sparsity_ratio: float
    sindy_gradient_norm: float
    sindy_gradient_nonzero: bool
    sindy_throughput_items_per_sec: float

    # Cell communication operator metrics
    comm_scores_shape: tuple[int, ...]
    comm_scores_finite: bool
    comm_signaling_shape: tuple[int, ...]
    comm_niche_shape: tuple[int, ...]
    comm_gradient_norm: float
    comm_gradient_nonzero: bool
    comm_throughput_items_per_sec: float

    # Dataset parameters
    dataset_params: dict[str, int] = field(default_factory=dict)


# -------------------------------------------------------------------
# Synthetic data generation
# -------------------------------------------------------------------


def generate_grn_data(
    n_cells: int,
    n_genes: int,
    n_tfs: int,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic scRNA-seq data for GRN inference.

    Creates a count matrix with non-negative expression values and
    TF indices pointing to the first ``n_tfs`` genes.

    Args:
        n_cells: Number of cells.
        n_genes: Number of genes.
        n_tfs: Number of transcription factors.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with ``counts`` and ``tf_indices`` arrays.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)

    # Log-normal-ish expression to mimic real scRNA-seq
    log_expr = jax.random.normal(k1, (n_cells, n_genes)) * 1.5 + 2.0
    counts = jnp.exp(log_expr)

    # Add sparsity (dropout zeros)
    dropout_mask = jax.random.bernoulli(k2, 0.7, (n_cells, n_genes))
    counts = counts * dropout_mask

    tf_indices = jnp.arange(n_tfs, dtype=jnp.int32)

    return {"counts": counts, "tf_indices": tf_indices}


def generate_sindy_data(
    n_timepoints: int,
    n_genes: int,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate smooth time-series expression data for SINDy GRN inference.

    Produces a temporally smooth expression matrix suitable for
    numerical differentiation by SINDy. Uses sinusoidal basis functions
    with random frequencies and amplitudes.

    Args:
        n_timepoints: Number of time points.
        n_genes: Number of genes.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with ``counts`` array of shape ``(n_timepoints, n_genes)``.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    # Time axis
    t = jnp.linspace(0.0, 2.0 * jnp.pi, n_timepoints)[:, None]

    # Random frequencies and amplitudes per gene
    freqs = jax.random.uniform(k1, (1, n_genes), minval=0.5, maxval=3.0)
    amplitudes = jax.random.uniform(k2, (1, n_genes), minval=0.5, maxval=2.0)
    phases = jax.random.uniform(k3, (1, n_genes), minval=0.0, maxval=2.0 * jnp.pi)

    # Smooth sinusoidal expression (always positive)
    counts = amplitudes * jnp.sin(freqs * t + phases) + amplitudes + 0.1

    return {"counts": counts}


def generate_communication_data(
    n_cells: int,
    n_genes: int,
    k_neighbors: int,
    n_lr_pairs: int,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic spatial expression data for cell communication.

    Creates expression counts, a k-NN spatial graph, and ligand-receptor
    pair indices.

    Args:
        n_cells: Number of cells.
        n_genes: Number of genes.
        k_neighbors: Number of spatial neighbors per cell.
        n_lr_pairs: Number of ligand-receptor pairs.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with ``counts``, ``spatial_graph``, and ``lr_pairs``.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    # Non-negative expression
    counts = jnp.abs(jax.random.normal(k1, (n_cells, n_genes))) + 0.1

    # Build k-NN spatial graph from random 2D positions
    positions = jax.random.normal(k2, (n_cells, 2))
    # Pairwise distances
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = jnp.sum(diffs ** 2, axis=-1)
    # Mask self-distances
    dists = dists + jnp.eye(n_cells) * 1e9

    # For each cell, pick k nearest neighbors
    # Use argsort and take first k per row
    sorted_indices = jnp.argsort(dists, axis=1)[:, :k_neighbors]

    # Build edge list: (2, n_cells * k_neighbors)
    sources = jnp.repeat(jnp.arange(n_cells), k_neighbors)
    targets = sorted_indices.reshape(-1)
    spatial_graph = jnp.stack([sources, targets], axis=0).astype(jnp.int32)

    # Random LR pair gene indices (must be valid gene indices)
    lr_gene_indices = jax.random.randint(
        k3, (n_lr_pairs, 2), 0, n_genes
    ).astype(jnp.int32)

    return {
        "counts": counts,
        "spatial_graph": spatial_graph,
        "lr_pairs": lr_gene_indices,
    }


# -------------------------------------------------------------------
# Individual operator benchmarks
# -------------------------------------------------------------------


def benchmark_grn(
    data: dict[str, jnp.ndarray],
    n_tfs: int,
    n_genes: int,
    quick: bool,
) -> dict[str, float | bool | tuple[int, ...]]:
    """Benchmark DifferentiableGRN operator.

    Args:
        data: Input data dictionary with ``counts`` and ``tf_indices``.
        n_tfs: Number of transcription factors.
        n_genes: Number of genes.
        quick: If True, reduce throughput iterations.

    Returns:
        Dictionary of benchmark metrics.
    """
    print("\n  Benchmarking DifferentiableGRN...")

    config = GRNInferenceConfig(
        n_tfs=n_tfs,
        n_genes=n_genes,
        hidden_dim=32,
        num_heads=4,
        sparsity_temperature=0.1,
        sparsity_lambda=0.01,
    )
    operator = DifferentiableGRN(config, rngs=nnx.Rngs(0))

    # Forward pass
    result, _, _ = operator.apply(data, {}, None)
    grn_matrix = result["grn_matrix"]
    tf_activity = result["tf_activity"]

    matrix_shape = tuple(int(d) for d in grn_matrix.shape)
    activity_shape = tuple(int(d) for d in tf_activity.shape)
    is_finite = bool(jnp.all(jnp.isfinite(grn_matrix)))

    print(f"    GRN matrix shape: {matrix_shape}")
    print(f"    TF activity shape: {activity_shape}")
    print(f"    All values finite: {is_finite}")

    # Gradient flow
    def loss_fn(op: DifferentiableGRN, input_data: dict) -> jnp.ndarray:
        """Scalar loss from GRN matrix sum."""
        out, _, _ = op.apply(input_data, {}, None)
        return out["grn_matrix"].sum()

    grad_result = check_gradient_flow(loss_fn, operator, data)
    print(f"    Gradient norm: {grad_result.gradient_norm:.6f}")
    print(f"    Gradient nonzero: {grad_result.gradient_nonzero}")

    # Throughput
    n_iters = 10 if quick else 50
    throughput = measure_throughput(
        lambda d: operator.apply(d, {}, None),
        (data,),
        n_iterations=n_iters,
        warmup=2,
    )
    print(f"    Throughput: {throughput['items_per_sec']:.1f} calls/sec")

    return {
        "grn_matrix_shape": matrix_shape,
        "grn_matrix_finite": is_finite,
        "grn_tf_activity_shape": activity_shape,
        "grn_gradient_norm": grad_result.gradient_norm,
        "grn_gradient_nonzero": grad_result.gradient_nonzero,
        "grn_throughput_items_per_sec": throughput["items_per_sec"],
    }


def benchmark_sindy(
    data: dict[str, jnp.ndarray],
    n_genes: int,
    quick: bool,
) -> dict[str, float | bool | tuple[int, ...]]:
    """Benchmark SINDyGRNOperator.

    Args:
        data: Input data dictionary with ``counts``.
        n_genes: Number of genes.
        quick: If True, reduce throughput iterations.

    Returns:
        Dictionary of benchmark metrics.
    """
    print("\n  Benchmarking SINDyGRNOperator...")

    config = SINDyGRNConfig(
        n_genes=n_genes,
        polynomial_degree=1,
        sparsity_threshold=0.1,
        n_iterations=10,
        ridge_alpha=0.01,
    )
    operator = SINDyGRNOperator(config, rngs=nnx.Rngs(0))

    # Forward pass
    result, _, _ = operator.apply(data, {}, None)
    coefficients = result["grn_coefficients"]

    coeff_shape = tuple(int(d) for d in coefficients.shape)
    is_finite = bool(jnp.all(jnp.isfinite(coefficients)))

    # Sparsity: fraction of near-zero coefficients
    near_zero = jnp.sum(jnp.abs(coefficients) < 1e-3)
    total_elements = coefficients.size
    sparsity_ratio = float(near_zero / total_elements)

    print(f"    Coefficients shape: {coeff_shape}")
    print(f"    All values finite: {is_finite}")
    print(f"    Sparsity ratio: {sparsity_ratio:.2%}")

    # Gradient flow -- SINDy has no nnx.Param leaves, so differentiate
    # w.r.t. input data instead
    def loss_fn_data(input_data: dict) -> jnp.ndarray:
        """Scalar loss from coefficients sum."""
        out, _, _ = operator.apply(input_data, {}, None)
        return out["grn_coefficients"].sum()

    try:
        grad_fn = jax.grad(loss_fn_data)
        grads = grad_fn(data)
        grad_norm = float(
            jnp.linalg.norm(grads["counts"])
        )
        grad_nonzero = grad_norm > 1e-8
    except Exception as exc:
        logger.warning("SINDy gradient check failed: %s", exc)
        grad_norm = 0.0
        grad_nonzero = False

    print(f"    Gradient norm (w.r.t. input): {grad_norm:.6f}")
    print(f"    Gradient nonzero: {grad_nonzero}")

    # Throughput
    n_iters = 10 if quick else 50
    throughput = measure_throughput(
        lambda d: operator.apply(d, {}, None),
        (data,),
        n_iterations=n_iters,
        warmup=2,
    )
    print(f"    Throughput: {throughput['items_per_sec']:.1f} calls/sec")

    return {
        "sindy_coefficients_shape": coeff_shape,
        "sindy_coefficients_finite": is_finite,
        "sindy_sparsity_ratio": sparsity_ratio,
        "sindy_gradient_norm": grad_norm,
        "sindy_gradient_nonzero": grad_nonzero,
        "sindy_throughput_items_per_sec": throughput["items_per_sec"],
    }


def benchmark_communication(
    data: dict[str, jnp.ndarray],
    n_genes: int,
    n_lr_pairs: int,
    quick: bool,
) -> dict[str, float | bool | tuple[int, ...]]:
    """Benchmark DifferentiableCellCommunication operator.

    Args:
        data: Input data with ``counts``, ``spatial_graph``, ``lr_pairs``.
        n_genes: Number of genes.
        n_lr_pairs: Number of ligand-receptor pairs.
        quick: If True, reduce throughput iterations.

    Returns:
        Dictionary of benchmark metrics.
    """
    print("\n  Benchmarking DifferentiableCellCommunication...")

    config = CellCommunicationConfig(
        n_genes=n_genes,
        n_lr_pairs=n_lr_pairs,
        hidden_dim=32,
        num_heads=4,
        edge_features_dim=8,
        num_gnn_layers=2,
        n_pathways=10,
        dropout_rate=0.0,
    )
    operator = DifferentiableCellCommunication(config, rngs=nnx.Rngs(0))

    # Forward pass
    result, _, _ = operator.apply(data, {}, None)
    comm_scores = result["communication_scores"]
    signaling = result["signaling_activity"]
    niche = result["niche_embeddings"]

    scores_shape = tuple(int(d) for d in comm_scores.shape)
    signaling_shape = tuple(int(d) for d in signaling.shape)
    niche_shape = tuple(int(d) for d in niche.shape)
    is_finite = bool(
        jnp.all(jnp.isfinite(comm_scores))
        & jnp.all(jnp.isfinite(signaling))
        & jnp.all(jnp.isfinite(niche))
    )

    print(f"    Communication scores shape: {scores_shape}")
    print(f"    Signaling activity shape: {signaling_shape}")
    print(f"    Niche embeddings shape: {niche_shape}")
    print(f"    All values finite: {is_finite}")

    # Gradient flow
    def loss_fn(
        op: DifferentiableCellCommunication,
        input_data: dict,
    ) -> jnp.ndarray:
        """Scalar loss from communication scores sum."""
        out, _, _ = op.apply(input_data, {}, None)
        return out["communication_scores"].sum()

    grad_result = check_gradient_flow(loss_fn, operator, data)
    print(f"    Gradient norm: {grad_result.gradient_norm:.6f}")
    print(f"    Gradient nonzero: {grad_result.gradient_nonzero}")

    # Throughput
    n_iters = 10 if quick else 50
    throughput = measure_throughput(
        lambda d: operator.apply(d, {}, None),
        (data,),
        n_iterations=n_iters,
        warmup=2,
    )
    print(f"    Throughput: {throughput['items_per_sec']:.1f} calls/sec")

    return {
        "comm_scores_shape": scores_shape,
        "comm_scores_finite": is_finite,
        "comm_signaling_shape": signaling_shape,
        "comm_niche_shape": niche_shape,
        "comm_gradient_norm": grad_result.gradient_norm,
        "comm_gradient_nonzero": grad_result.gradient_nonzero,
        "comm_throughput_items_per_sec": throughput["items_per_sec"],
    }


# -------------------------------------------------------------------
# Main benchmark runner
# -------------------------------------------------------------------


def run_benchmark(quick: bool = False) -> GRNBenchmarkResult:
    """Run the complete GRN and cell communication benchmark.

    Args:
        quick: If True, use smaller datasets and fewer iterations.

    Returns:
        Aggregated benchmark result.
    """
    print("=" * 60)
    print("DiffBio GRN & Cell Communication Benchmark")
    print("=" * 60)

    # Dataset parameters
    grn_n_cells = 30 if quick else 100
    grn_n_genes = 50
    grn_n_tfs = 10

    sindy_n_timepoints = 20 if quick else 50
    sindy_n_genes = 20

    comm_n_cells = 30 if quick else 80
    comm_n_genes = 30
    comm_k_neighbors = 5
    comm_n_lr_pairs = 5

    dataset_params = {
        "grn_n_cells": grn_n_cells,
        "grn_n_genes": grn_n_genes,
        "grn_n_tfs": grn_n_tfs,
        "sindy_n_timepoints": sindy_n_timepoints,
        "sindy_n_genes": sindy_n_genes,
        "comm_n_cells": comm_n_cells,
        "comm_n_genes": comm_n_genes,
        "comm_k_neighbors": comm_k_neighbors,
        "comm_n_lr_pairs": comm_n_lr_pairs,
    }

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    grn_data = generate_grn_data(
        n_cells=grn_n_cells,
        n_genes=grn_n_genes,
        n_tfs=grn_n_tfs,
        seed=42,
    )
    print(
        f"  GRN: {grn_n_cells} cells, "
        f"{grn_n_genes} genes, {grn_n_tfs} TFs"
    )

    sindy_data = generate_sindy_data(
        n_timepoints=sindy_n_timepoints,
        n_genes=sindy_n_genes,
        seed=43,
    )
    print(
        f"  SINDy: {sindy_n_timepoints} timepoints, "
        f"{sindy_n_genes} genes"
    )

    comm_data = generate_communication_data(
        n_cells=comm_n_cells,
        n_genes=comm_n_genes,
        k_neighbors=comm_k_neighbors,
        n_lr_pairs=comm_n_lr_pairs,
        seed=44,
    )
    n_edges = comm_data["spatial_graph"].shape[1]
    print(
        f"  Communication: {comm_n_cells} cells, "
        f"{comm_n_genes} genes, {n_edges} edges, "
        f"{comm_n_lr_pairs} LR pairs"
    )

    # Run benchmarks
    grn_metrics = benchmark_grn(
        grn_data, grn_n_tfs, grn_n_genes, quick
    )
    sindy_metrics = benchmark_sindy(
        sindy_data, sindy_n_genes, quick
    )
    comm_metrics = benchmark_communication(
        comm_data, comm_n_genes, comm_n_lr_pairs, quick
    )

    # Assemble result
    result = GRNBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        grn_matrix_shape=grn_metrics["grn_matrix_shape"],
        grn_matrix_finite=grn_metrics["grn_matrix_finite"],
        grn_tf_activity_shape=grn_metrics["grn_tf_activity_shape"],
        grn_gradient_norm=grn_metrics["grn_gradient_norm"],
        grn_gradient_nonzero=grn_metrics["grn_gradient_nonzero"],
        grn_throughput_items_per_sec=grn_metrics[
            "grn_throughput_items_per_sec"
        ],
        sindy_coefficients_shape=sindy_metrics[
            "sindy_coefficients_shape"
        ],
        sindy_coefficients_finite=sindy_metrics[
            "sindy_coefficients_finite"
        ],
        sindy_sparsity_ratio=sindy_metrics["sindy_sparsity_ratio"],
        sindy_gradient_norm=sindy_metrics["sindy_gradient_norm"],
        sindy_gradient_nonzero=sindy_metrics["sindy_gradient_nonzero"],
        sindy_throughput_items_per_sec=sindy_metrics[
            "sindy_throughput_items_per_sec"
        ],
        comm_scores_shape=comm_metrics["comm_scores_shape"],
        comm_scores_finite=comm_metrics["comm_scores_finite"],
        comm_signaling_shape=comm_metrics["comm_signaling_shape"],
        comm_niche_shape=comm_metrics["comm_niche_shape"],
        comm_gradient_norm=comm_metrics["comm_gradient_norm"],
        comm_gradient_nonzero=comm_metrics["comm_gradient_nonzero"],
        comm_throughput_items_per_sec=comm_metrics[
            "comm_throughput_items_per_sec"
        ],
        dataset_params=dataset_params,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        f"  GRN: shape={grn_metrics['grn_matrix_shape']}, "
        f"grad={grn_metrics['grn_gradient_nonzero']}"
    )
    print(
        f"  SINDy: shape={sindy_metrics['sindy_coefficients_shape']}, "
        f"sparsity={sindy_metrics['sindy_sparsity_ratio']:.2%}, "
        f"grad={sindy_metrics['sindy_gradient_nonzero']}"
    )
    print(
        f"  Comm: shape={comm_metrics['comm_scores_shape']}, "
        f"grad={comm_metrics['comm_gradient_nonzero']}"
    )
    print("=" * 60)

    return result


def main() -> None:
    """Main entry point for the GRN benchmark."""
    quick = "--quick" in sys.argv
    result = run_benchmark(quick=quick)
    output_path = save_benchmark_result(
        asdict(result),
        domain="singlecell",
        benchmark_name="grn_benchmark",
        output_dir=Path("benchmarks/results"),
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
