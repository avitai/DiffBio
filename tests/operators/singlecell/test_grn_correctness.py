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

import jax
import jax.numpy as jnp

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
    dists = jnp.sum(diffs**2, axis=-1)
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
    lr_gene_indices = jax.random.randint(k3, (n_lr_pairs, 2), 0, n_genes).astype(jnp.int32)

    return {
        "counts": counts,
        "spatial_graph": spatial_graph,
        "lr_pairs": lr_gene_indices,
    }
