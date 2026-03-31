#!/usr/bin/env python3
"""Multi-omics integration benchmark for DiffBio.

Evaluates DiffBio's multi-omics operators for correctness and
differentiability:

- DifferentiableMultiOmicsVAE  (Product-of-Experts fusion)
- SpatialDeconvolution         (cell-type deconvolution)
- HiCContactAnalysis           (chromatin contact analysis)
- DifferentiableSpatialGeneDetector  (spatially variable genes)

Usage:
    python benchmarks/multiomics/multiomics_benchmark.py
    python benchmarks/multiomics/multiomics_benchmark.py --quick
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ------------------------------------------------------------------
# Synthetic data generators
# ------------------------------------------------------------------


def _generate_vae_data(
    n_cells: int,
    n_rna_genes: int,
    n_atac_peaks: int,
    seed: int = 0,
) -> dict[str, jax.Array]:
    """Generate synthetic multi-omics count matrices.

    Args:
        n_cells: Number of cells.
        n_rna_genes: Number of RNA genes (first modality).
        n_atac_peaks: Number of ATAC peaks (second modality).
        seed: Random seed.

    Returns:
        Dictionary with ``rna_counts`` and ``atac_counts`` arrays.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    rna = jnp.abs(jax.random.normal(k1, (n_cells, n_rna_genes))) * 5.0
    atac = jnp.abs(jax.random.normal(k2, (n_cells, n_atac_peaks))) * 3.0
    return {"rna_counts": rna, "atac_counts": atac}


def _generate_deconv_data(
    n_spots: int,
    n_genes: int,
    n_cell_types: int,
    seed: int = 1,
) -> dict[str, jax.Array]:
    """Generate synthetic spatial deconvolution inputs.

    Creates spot expression, reference profiles, and 2-D coordinates.

    Args:
        n_spots: Number of spatial spots.
        n_genes: Number of genes.
        n_cell_types: Number of reference cell types.
        seed: Random seed.

    Returns:
        Dictionary with ``spot_expression``, ``reference_profiles``,
        and ``coordinates`` arrays.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    spot_expr = jnp.abs(jax.random.normal(k1, (n_spots, n_genes))) * 4.0
    ref_profiles = jnp.abs(jax.random.normal(k2, (n_cell_types, n_genes))) * 3.0
    coords = jax.random.uniform(k3, (n_spots, 2)) * 10.0
    return {
        "spot_expression": spot_expr,
        "reference_profiles": ref_profiles,
        "coordinates": coords,
    }


def _generate_hic_data(
    n_bins: int,
    n_tad_blocks: int = 5,
    bin_features_dim: int = 16,
    seed: int = 2,
) -> dict[str, jax.Array]:
    """Generate synthetic Hi-C contact matrix with block-diagonal TADs.

    Builds a symmetric contact matrix where blocks along the diagonal
    have elevated counts to simulate topologically associating domains.

    Args:
        n_bins: Number of genomic bins.
        n_tad_blocks: Approximate number of TAD blocks.
        bin_features_dim: Dimension of per-bin feature vectors.
        seed: Random seed.

    Returns:
        Dictionary with ``contact_matrix`` and ``bin_features`` arrays.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    # Background contacts decay with distance
    positions = jnp.arange(n_bins, dtype=jnp.float32)
    dist = jnp.abs(positions[:, None] - positions[None, :])
    background = jnp.exp(-dist / (n_bins / 4.0))

    # Block-diagonal TAD structure
    block_size = max(n_bins // n_tad_blocks, 1)
    block_ids = positions // block_size
    same_block = (block_ids[:, None] == block_ids[None, :]).astype(jnp.float32)

    noise = jnp.abs(jax.random.normal(k1, (n_bins, n_bins))) * 0.1
    contact_matrix = background + same_block * 2.0 + noise
    # Symmetrize
    contact_matrix = (contact_matrix + contact_matrix.T) / 2.0

    bin_features = jax.random.normal(k2, (n_bins, bin_features_dim))
    return {
        "contact_matrix": contact_matrix,
        "bin_features": bin_features,
    }


def _generate_spatial_gene_data(
    n_spots: int,
    n_genes: int,
    n_spatial_genes: int = 5,
    seed: int = 3,
) -> dict[str, jax.Array]:
    """Generate synthetic spatial gene expression data.

    A subset of genes are given spatially correlated expression
    patterns (sinusoidal over coordinates) while the rest are
    random noise.

    Args:
        n_spots: Number of spatial spots.
        n_genes: Number of genes.
        n_spatial_genes: Number of genes with spatial patterns.
        seed: Random seed.

    Returns:
        Dictionary with ``spatial_coords``, ``expression``, and
        ``total_counts`` arrays.
    """
    key = jax.random.key(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    coords = jax.random.uniform(k1, (n_spots, 2)) * 10.0

    # Non-spatial (noise) genes
    expression = jnp.abs(jax.random.normal(k2, (n_spots, n_genes))) * 2.0

    # Overwrite first n_spatial_genes with a spatial pattern
    freqs = jax.random.uniform(k3, (n_spatial_genes,), minval=0.5, maxval=2.0)
    phases = jax.random.uniform(k4, (n_spatial_genes,)) * 2.0 * jnp.pi
    # Sinusoidal pattern driven by x-coordinate
    spatial_signal = jnp.sin(coords[:, 0:1] * freqs[None, :] + phases[None, :])
    spatial_expr = jnp.abs(spatial_signal) * 5.0 + 1.0
    expression = expression.at[:, :n_spatial_genes].set(spatial_expr)

    total_counts = jnp.sum(expression, axis=-1)
    return {
        "spatial_coords": coords,
        "expression": expression,
        "total_counts": total_counts,
    }
