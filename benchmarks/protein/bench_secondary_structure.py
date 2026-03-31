#!/usr/bin/env python3
"""Protein secondary structure benchmark: DifferentiableSecondaryStructure.

Evaluates DiffBio's differentiable DSSP approximation on synthetic
protein backbone coordinates with known secondary structure labels.

The benchmark generates ideal backbone geometries for three secondary
structure types (alpha helix, beta strand, random coil) and measures
how accurately the operator recovers the correct Q3 assignments.

Results are compared against published baselines: DSSP exact (~0.99),
STRIDE (~0.97), KAKSI (~0.96), PyDSSP (~0.95).

Usage:
    python benchmarks/protein/bench_secondary_structure.py
    python benchmarks/protein/bench_secondary_structure.py --quick
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.protein import SS_BASELINES
from diffbio.operators.protein.secondary_structure import (
    DifferentiableSecondaryStructure,
    SecondaryStructureConfig,
    SS_HELIX,
    SS_LOOP,
    SS_STRAND,
)

logger = logging.getLogger(__name__)

# Backbone geometry constants (Angstroms and radians)
_HELIX_RISE_PER_RESIDUE = 1.5
_HELIX_RESIDUES_PER_TURN = 3.6
_HELIX_RADIUS = 2.3
_STRAND_RISE_PER_RESIDUE = 3.3
_STRAND_LATERAL_OFFSET = 0.8

# Bond geometry for backbone atoms relative to CA
_BOND_N_CA = 1.47  # N-CA bond length
_BOND_CA_C = 1.52  # CA-C bond length
_BOND_C_O = 1.23  # C=O bond length

_CONFIG = DiffBioBenchmarkConfig(
    name="protein/secondary_structure",
    domain="protein",
    n_iterations_quick=5,
    n_iterations_full=20,
)

# Section lengths
_N_HELIX = 20
_N_STRAND = 15
_N_COIL = 15
_N_TOTAL = _N_HELIX + _N_STRAND + _N_COIL


def _generate_helix_backbone(
    n_residues: int,
) -> jnp.ndarray:
    """Generate ideal alpha-helix backbone coordinates.

    Uses helical parameters: 1.5A rise per residue, 3.6 residues
    per turn, radius ~2.3A. Atoms are placed in the order
    [N, CA, C, O] for each residue.

    Args:
        n_residues: Number of residues to generate.

    Returns:
        Backbone coordinates of shape ``(n_residues, 4, 3)``.
    """
    coords = np.zeros((n_residues, 4, 3), dtype=np.float32)
    angular_step = 2.0 * np.pi / _HELIX_RESIDUES_PER_TURN

    for i in range(n_residues):
        theta = i * angular_step
        z = i * _HELIX_RISE_PER_RESIDUE

        # CA position on the helix
        ca_x = _HELIX_RADIUS * np.cos(theta)
        ca_y = _HELIX_RADIUS * np.sin(theta)
        ca_z = z
        ca = np.array([ca_x, ca_y, ca_z])

        # N: displaced along helix axis (toward previous residue)
        n_theta = (i - 0.3) * angular_step
        n_x = (_HELIX_RADIUS - 0.3) * np.cos(n_theta)
        n_y = (_HELIX_RADIUS - 0.3) * np.sin(n_theta)
        n_z = ca_z - _BOND_N_CA * 0.6
        n = np.array([n_x, n_y, n_z])

        # C: displaced along helix axis (toward next residue)
        c_theta = (i + 0.3) * angular_step
        c_x = (_HELIX_RADIUS + 0.2) * np.cos(c_theta)
        c_y = (_HELIX_RADIUS + 0.2) * np.sin(c_theta)
        c_z = ca_z + _BOND_CA_C * 0.5
        c = np.array([c_x, c_y, c_z])

        # O: perpendicular to C-CA bond, pointing outward
        ca_c_vec = c - ca
        ca_c_norm = ca_c_vec / (np.linalg.norm(ca_c_vec) + 1e-8)
        # Perpendicular in the xy plane
        perp = np.array([-ca_c_norm[1], ca_c_norm[0], 0.0])
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        o = c + _BOND_C_O * (0.5 * ca_c_norm + 0.866 * perp)

        coords[i, 0] = n
        coords[i, 1] = ca
        coords[i, 2] = c
        coords[i, 3] = o

    return jnp.array(coords)


def _generate_strand_backbone(
    n_residues: int,
) -> jnp.ndarray:
    """Generate ideal beta-strand backbone coordinates.

    Uses extended geometry: 3.3A rise per residue with alternating
    lateral displacement to create the characteristic zig-zag.

    Args:
        n_residues: Number of residues to generate.

    Returns:
        Backbone coordinates of shape ``(n_residues, 4, 3)``.
    """
    coords = np.zeros((n_residues, 4, 3), dtype=np.float32)

    for i in range(n_residues):
        z = i * _STRAND_RISE_PER_RESIDUE
        # Alternating y displacement for zig-zag
        y_offset = _STRAND_LATERAL_OFFSET * ((-1) ** i)

        # CA on the strand axis
        ca = np.array([0.0, y_offset, z])

        # N: slightly before CA along z
        n = np.array([0.0, y_offset * 0.8, z - _BOND_N_CA * 0.8])

        # C: slightly after CA along z
        c = np.array([0.0, y_offset * 0.6, z + _BOND_CA_C * 0.8])

        # O: perpendicular to the strand plane
        o = c + np.array([_BOND_C_O, 0.0, 0.0])

        coords[i, 0] = n
        coords[i, 1] = ca
        coords[i, 2] = c
        coords[i, 3] = o

    return jnp.array(coords)


def _generate_coil_backbone(
    n_residues: int,
    *,
    seed: int = 123,
) -> jnp.ndarray:
    """Generate random coil backbone coordinates.

    Uses a random walk along the z-axis with random lateral
    displacements. The geometry is deliberately irregular to
    avoid helix or strand patterns.

    Args:
        n_residues: Number of residues to generate.
        seed: Random seed for reproducibility.

    Returns:
        Backbone coordinates of shape ``(n_residues, 4, 3)``.
    """
    rng = np.random.default_rng(seed)
    coords = np.zeros((n_residues, 4, 3), dtype=np.float32)

    for i in range(n_residues):
        z = i * 3.0 + rng.uniform(-0.5, 0.5)
        x = rng.uniform(-3.0, 3.0)
        y = rng.uniform(-3.0, 3.0)

        ca = np.array([x, y, z])
        n = ca + np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), -1.2])
        c = ca + np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), 1.0])
        o = c + np.array([rng.uniform(-0.6, 0.6), _BOND_C_O, rng.uniform(-0.3, 0.3)])

        coords[i, 0] = n
        coords[i, 1] = ca
        coords[i, 2] = c
        coords[i, 3] = o

    return jnp.array(coords)


def generate_ideal_backbone() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate combined backbone with known SS labels.

    Creates a protein backbone with three segments:
    - 20 residues of alpha helix (label = 1 = SS_HELIX)
    - 15 residues of beta strand (label = 2 = SS_STRAND)
    - 15 residues of random coil (label = 0 = SS_LOOP)

    The strand segment is offset in x to avoid interactions
    with the helix segment.

    Returns:
        Tuple of:
        - coordinates: ``(1, 50, 4, 3)`` backbone atom positions
        - labels: ``(50,)`` integer SS labels (0=coil, 1=helix, 2=strand)
    """
    helix = _generate_helix_backbone(_N_HELIX)
    strand = _generate_strand_backbone(_N_STRAND)
    coil = _generate_coil_backbone(_N_COIL)

    # Offset strand and coil in x to avoid cross-segment H-bonds
    strand_offset = jnp.array([30.0, 0.0, 0.0])
    coil_offset = jnp.array([60.0, 0.0, 0.0])
    strand = strand + strand_offset[None, None, :]
    coil = coil + coil_offset[None, None, :]

    # Concatenate segments
    all_coords = jnp.concatenate([helix, strand, coil], axis=0)
    all_coords = all_coords[None, :, :, :]  # Add batch dim -> (1, 50, 4, 3)

    # Build labels
    labels = np.concatenate(
        [
            np.full(_N_HELIX, SS_HELIX, dtype=np.int32),
            np.full(_N_STRAND, SS_STRAND, dtype=np.int32),
            np.full(_N_COIL, SS_LOOP, dtype=np.int32),
        ]
    )

    return all_coords, jnp.array(labels)


def compute_q3_metrics(
    predicted: jnp.ndarray,
    true_labels: jnp.ndarray,
) -> dict[str, float]:
    """Compute Q3 secondary structure accuracy metrics.

    Q3 is the three-state accuracy: fraction of residues with
    correctly predicted secondary structure (helix, strand, coil).

    Args:
        predicted: Predicted SS indices ``(length,)``,
            values in {0=coil, 1=helix, 2=strand}.
        true_labels: Ground truth SS indices ``(length,)``.

    Returns:
        Dict with q3_overall, q3_helix, q3_strand, q3_coil.
    """
    pred_np = np.asarray(predicted)
    true_np = np.asarray(true_labels)

    total = len(pred_np)
    correct = int(np.sum(pred_np == true_np))
    q3_overall = correct / total if total > 0 else 0.0

    per_class: dict[str, float] = {}
    class_names = {SS_HELIX: "q3_helix", SS_STRAND: "q3_strand", SS_LOOP: "q3_coil"}
    for class_idx, metric_name in class_names.items():
        mask = true_np == class_idx
        n_class = int(np.sum(mask))
        if n_class > 0:
            n_correct = int(np.sum(pred_np[mask] == class_idx))
            per_class[metric_name] = n_correct / n_class
        else:
            per_class[metric_name] = 0.0

    return {"q3_overall": q3_overall, **per_class}


class SecondaryStructureBenchmark(DiffBioBenchmark):
    """Evaluate DifferentiableSecondaryStructure on ideal backbones."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
    ) -> None:
        super().__init__(config, quick=quick)

    def _run_core(self) -> dict[str, Any]:
        """Generate ideal structures, predict SS, compute Q3."""
        # 1. Generate ideal backbone coordinates with known labels
        logger.info("Generating ideal backbone coordinates...")
        coords, true_labels = generate_ideal_backbone()
        logger.info(
            "  Shape: %s (%d helix, %d strand, %d coil)",
            coords.shape,
            _N_HELIX,
            _N_STRAND,
            _N_COIL,
        )

        # 2. Create operator
        op_config = SecondaryStructureConfig(
            margin=1.0,
            cutoff=-0.5,
            temperature=1.0,
        )
        rngs = nnx.Rngs(42)
        operator = DifferentiableSecondaryStructure(op_config, rngs=rngs)

        # 3. Run prediction
        logger.info("Running DifferentiableSecondaryStructure...")
        input_data: dict[str, Any] = {"coordinates": coords}
        result, _, _ = operator.apply(input_data, {}, None)

        ss_indices = result["ss_indices"][0]  # Remove batch dim
        result["ss_onehot"][0]

        # 4. Compute Q3 metrics
        quality = compute_q3_metrics(ss_indices, true_labels)
        for key, value in sorted(quality.items()):
            logger.info("  %s: %.4f", key, value)

        # Log per-class prediction distribution
        pred_np = np.asarray(ss_indices)
        n_pred_helix = int(np.sum(pred_np == SS_HELIX))
        n_pred_strand = int(np.sum(pred_np == SS_STRAND))
        n_pred_coil = int(np.sum(pred_np == SS_LOOP))
        logger.info(
            "  Predicted: %d helix, %d strand, %d coil",
            n_pred_helix,
            n_pred_strand,
            n_pred_coil,
        )

        # 5. Build gradient check function
        def loss_fn(
            model: DifferentiableSecondaryStructure,
            d: dict[str, Any],
        ) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["ss_onehot"])

        return {
            "metrics": quality,
            "operator": operator,
            "input_data": input_data,
            "loss_fn": loss_fn,
            "n_items": _N_TOTAL,
            "iterate_fn": lambda: operator.apply(input_data, {}, None),
            "baselines": SS_BASELINES,
            "dataset_info": {
                "name": "ideal_structures",
                "n_residues": _N_TOTAL,
                "n_helix": _N_HELIX,
                "n_strand": _N_STRAND,
                "n_coil": _N_COIL,
            },
            "operator_config": {
                "margin": op_config.margin,
                "cutoff": op_config.cutoff,
                "temperature": op_config.temperature,
                "min_helix_length": op_config.min_helix_length,
            },
            "operator_name": "DifferentiableSecondaryStructure",
            "dataset_name": "ideal_structures",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(
        SecondaryStructureBenchmark,
        _CONFIG,
    )


if __name__ == "__main__":
    main()
