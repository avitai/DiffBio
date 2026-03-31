#!/usr/bin/env python3
"""Spatial deconvolution benchmark: SpatialDeconvolution on seqFISH.

Evaluates DiffBio's SpatialDeconvolution operator on the seqFISH
mouse cortex dataset (Lohoff et al., 19,416 cells, 351 genes,
22 cell types) using a split-and-aggregate evaluation protocol:

1. Load real seqFISH h5ad (spatially resolved single-cell data)
2. Split cells: 80% as single-cell reference, 20% as spatial
3. Build reference signatures: mean expression per cell type
   from the reference split
4. Create pseudo-bulk spots: aggregate nearby spatial cells into
   spots with known ground-truth cell type proportions
5. Run SpatialDeconvolution to predict proportions per spot
6. Compare predicted vs actual proportions (Pearson, RMSE)

Results are compared against published baselines: RCTD (~0.85),
Cell2location (~0.88), CARD (~0.86).

Usage:
    python benchmarks/multiomics/bench_spatial_deconv.py
    python benchmarks/multiomics/bench_spatial_deconv.py --quick
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks._base import (
    DiffBioBenchmark,
    DiffBioBenchmarkConfig,
)
from benchmarks._baselines.deconvolution import (
    DECONVOLUTION_BASELINES,
)
from diffbio.operators.multiomics import (
    SpatialDeconvolution,
    SpatialDeconvolutionConfig,
)
from diffbio.sources.seqfish import SeqFISHConfig, SeqFISHSource

logger = logging.getLogger(__name__)

_CONFIG = DiffBioBenchmarkConfig(
    name="multiomics/spatial_deconvolution",
    domain="multiomics",
    quick_subsample=2000,
)

_REFERENCE_FRACTION = 0.8
_CELLS_PER_SPOT_MIN = 5
_CELLS_PER_SPOT_MAX = 10
_SPATIAL_RADIUS_PERCENTILE = 5


def _split_reference_spatial(
    n_cells: int,
    reference_fraction: float,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split cell indices into reference and spatial sets.

    Args:
        n_cells: Total number of cells.
        reference_fraction: Fraction of cells for reference.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (reference_indices, spatial_indices).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_cells)
    n_ref = int(n_cells * reference_fraction)
    ref_idx = np.sort(indices[:n_ref])
    spatial_idx = np.sort(indices[n_ref:])
    return ref_idx, spatial_idx


def _build_reference_signatures(
    counts: np.ndarray,
    cell_type_labels: np.ndarray,
    n_cell_types: int,
) -> jnp.ndarray:
    """Build mean expression per cell type from reference cells.

    Args:
        counts: Expression matrix (n_ref_cells, n_genes).
        cell_type_labels: Integer cell type labels (n_ref_cells,).
        n_cell_types: Total number of cell types.

    Returns:
        Reference profiles (n_cell_types, n_genes).
    """
    n_genes = counts.shape[1]
    signatures = np.zeros((n_cell_types, n_genes), dtype=np.float32)
    for ct in range(n_cell_types):
        mask = cell_type_labels == ct
        if mask.sum() > 0:
            signatures[ct] = counts[mask].mean(axis=0)
    return jnp.array(signatures)


def _aggregate_spots_by_proximity(
    counts: np.ndarray,
    spatial_coords: np.ndarray,
    cell_type_labels: np.ndarray,
    n_cell_types: int,
    *,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Aggregate nearby spatial cells into pseudo-bulk spots.

    Uses spatial proximity: for each seed cell, finds the nearest
    neighbours within a radius and aggregates their expression.
    Records the true cell type proportions as ground truth.

    Args:
        counts: Expression matrix (n_spatial, n_genes).
        spatial_coords: Spatial coordinates (n_spatial, 2).
        cell_type_labels: Integer labels (n_spatial,).
        n_cell_types: Total number of cell types.
        seed: Random seed for spot seeding.

    Returns:
        Dictionary with keys:
            - spot_expression: (n_spots, n_genes)
            - true_proportions: (n_spots, n_cell_types)
            - coordinates: (n_spots, 2)
    """
    from scipy.spatial import KDTree  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    n_spatial = counts.shape[0]

    # Build spatial KDTree for neighbour lookups
    tree = KDTree(spatial_coords)

    # Determine radius: use percentile of pairwise distances
    sample_size = min(1000, n_spatial)
    sample_idx = rng.choice(n_spatial, size=sample_size, replace=False)
    sample_dists = tree.query(spatial_coords[sample_idx], k=_CELLS_PER_SPOT_MAX)[0]
    radius = float(np.percentile(sample_dists[:, -1], _SPATIAL_RADIUS_PERCENTILE))
    # Ensure radius captures at least a few cells
    radius = max(radius, float(np.median(sample_dists[:, 1])) * 2)

    # Seed spots: randomly pick cells as spot centres, skip used
    used = np.zeros(n_spatial, dtype=bool)
    spot_expressions = []
    spot_proportions = []
    spot_coordinates = []

    centre_order = rng.permutation(n_spatial)
    for centre_idx in centre_order:
        if used[centre_idx]:
            continue

        # Find neighbours within radius
        neighbour_idx = tree.query_ball_point(spatial_coords[centre_idx], r=radius)
        # Filter already used
        neighbour_idx = [j for j in neighbour_idx if not used[j]]

        if len(neighbour_idx) < _CELLS_PER_SPOT_MIN:
            continue

        # Cap at max cells per spot
        if len(neighbour_idx) > _CELLS_PER_SPOT_MAX:
            neighbour_idx = list(
                rng.choice(
                    neighbour_idx,
                    size=_CELLS_PER_SPOT_MAX,
                    replace=False,
                )
            )

        neighbour_arr = np.array(neighbour_idx)
        used[neighbour_arr] = True

        # Aggregate expression (mean)
        spot_expr = counts[neighbour_arr].mean(axis=0)
        spot_expressions.append(spot_expr)

        # Ground truth proportions
        proportions = np.zeros(n_cell_types, dtype=np.float32)
        for label in cell_type_labels[neighbour_arr]:
            proportions[label] += 1.0
        proportions /= len(neighbour_arr)
        spot_proportions.append(proportions)

        # Spot coordinate = centroid of constituent cells
        centroid = spatial_coords[neighbour_arr].mean(axis=0)
        spot_coordinates.append(centroid)

    return {
        "spot_expression": jnp.array(np.stack(spot_expressions, axis=0)),
        "true_proportions": jnp.array(np.stack(spot_proportions, axis=0)),
        "coordinates": jnp.array(np.stack(spot_coordinates, axis=0)),
    }


def _compute_deconvolution_metrics(
    predicted: jnp.ndarray,
    ground_truth: jnp.ndarray,
) -> dict[str, float]:
    """Compute deconvolution quality metrics.

    Args:
        predicted: Predicted proportions (n_spots, n_cell_types).
        ground_truth: True proportions (n_spots, n_cell_types).

    Returns:
        Dictionary with pearson_correlation, rmse, and
        proportion_sum_to_one metrics.
    """
    pred_np = np.asarray(predicted)
    true_np = np.asarray(ground_truth)

    # Per-spot Pearson correlation, averaged across spots
    correlations = []
    for i in range(pred_np.shape[0]):
        pred_row = pred_np[i]
        true_row = true_np[i]
        if np.std(pred_row) > 1e-8 and np.std(true_row) > 1e-8:
            corr = float(np.corrcoef(pred_row, true_row)[0, 1])
            correlations.append(corr)
        else:
            correlations.append(0.0)
    pearson_correlation = float(np.mean(correlations))

    # RMSE across all spots and cell types
    rmse = float(np.sqrt(np.mean((pred_np - true_np) ** 2)))

    # Proportion sum-to-one: mean absolute deviation from 1.0
    row_sums = pred_np.sum(axis=-1)
    proportion_sum_to_one = 1.0 - float(np.mean(np.abs(row_sums - 1.0)))

    return {
        "pearson_correlation": pearson_correlation,
        "rmse": rmse,
        "proportion_sum_to_one": proportion_sum_to_one,
    }


class SpatialDeconvBenchmark(DiffBioBenchmark):
    """Evaluate SpatialDeconvolution on real seqFISH cortex data."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "/media/mahdi/ssd23/Data/spatial",
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        """Load seqFISH, split, aggregate spots, deconvolve."""
        subsample = self.config.quick_subsample if self.quick else None

        # 1. Load seqFISH dataset
        logger.info("Loading seqFISH cortex dataset...")
        source = SeqFISHSource(SeqFISHConfig(data_dir=self.data_dir, subsample=subsample))
        data = source.load()
        counts_jnp = data["counts"]
        cell_type_labels = data["cell_type_labels"]
        spatial_coords_jnp = data["spatial_coords"]
        n_cell_types = data["n_types"]
        n_genes = data["n_genes"]
        n_cells = data["n_cells"]

        logger.info(
            "  %d cells, %d genes, %d types",
            n_cells,
            n_genes,
            n_cell_types,
        )

        # 2. Split: 80% reference, 20% spatial
        ref_idx, spatial_idx = _split_reference_spatial(n_cells, _REFERENCE_FRACTION)
        logger.info(
            "  Split: %d reference, %d spatial",
            len(ref_idx),
            len(spatial_idx),
        )

        counts_np = np.asarray(counts_jnp)
        labels_np = np.asarray(cell_type_labels)
        coords_np = np.asarray(spatial_coords_jnp)

        ref_counts = counts_np[ref_idx]
        ref_labels = labels_np[ref_idx]
        spatial_counts = counts_np[spatial_idx]
        spatial_labels = labels_np[spatial_idx]
        spatial_coords = coords_np[spatial_idx]

        # 3. Build reference signatures from reference cells
        logger.info("Building reference signatures...")
        reference_profiles = _build_reference_signatures(ref_counts, ref_labels, n_cell_types)

        # 4. Aggregate spatial cells into pseudo-bulk spots
        logger.info("Aggregating spatial cells into spots...")
        spots = _aggregate_spots_by_proximity(
            spatial_counts,
            spatial_coords,
            spatial_labels,
            n_cell_types,
        )
        n_spots = spots["spot_expression"].shape[0]
        logger.info("  Created %d pseudo-bulk spots", n_spots)

        # 5. Create and run operator
        op_config = SpatialDeconvolutionConfig(
            n_genes=n_genes,
            n_cell_types=n_cell_types,
            hidden_dim=128,
            num_layers=2,
            spatial_hidden=32,
            temperature=1.0,
        )
        rngs = nnx.Rngs(42)
        operator = SpatialDeconvolution(op_config, rngs=rngs)

        input_data = {
            "spot_expression": spots["spot_expression"],
            "reference_profiles": reference_profiles,
            "coordinates": spots["coordinates"],
        }
        result, _, _ = operator.apply(input_data, {}, None)
        predicted_proportions = result["cell_proportions"]

        # 6. Compute metrics
        logger.info("Computing deconvolution metrics...")
        quality = _compute_deconvolution_metrics(predicted_proportions, spots["true_proportions"])
        for key, value in sorted(quality.items()):
            logger.info("  %s: %.4f", key, value)

        # Loss function for gradient check
        def loss_fn(model: SpatialDeconvolution, d: dict[str, Any]) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["cell_proportions"])

        return {
            "metrics": quality,
            "operator": operator,
            "input_data": input_data,
            "loss_fn": loss_fn,
            "n_items": n_spots,
            "iterate_fn": lambda: operator.apply(input_data, {}, None),
            "baselines": DECONVOLUTION_BASELINES,
            "dataset_info": {
                "name": "seqfish_cortex",
                "n_cells": n_cells,
                "n_genes": n_genes,
                "n_types": n_cell_types,
                "n_spots": n_spots,
                "n_reference": len(ref_idx),
                "n_spatial": len(spatial_idx),
            },
            "operator_config": {
                "n_genes": n_genes,
                "n_cell_types": n_cell_types,
                "hidden_dim": op_config.hidden_dim,
                "temperature": op_config.temperature,
            },
            "operator_name": "SpatialDeconvolution",
            "dataset_name": "seqfish_cortex",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(
        SpatialDeconvBenchmark,
        _CONFIG,
        data_dir="/media/mahdi/ssd23/Data/spatial",
    )


if __name__ == "__main__":
    main()
