#!/usr/bin/env python3
"""Protein Secondary Structure Benchmark for DiffBio.

This benchmark evaluates DiffBio's DifferentiableSecondaryStructure
operator for correctness, differentiability, and performance on
synthetic backbone coordinates with known secondary structure.

Benchmarks:
- Secondary structure assignment correctness (Q3 accuracy)
- Output shape and value validation
- Hydrogen bond map validity
- Differentiability verification (gradient flow)
- Performance measurement (residues/second)

Usage:
    python benchmarks/protein/protein_structure_benchmark.py
"""

from __future__ import annotations

import jax.numpy as jnp

from diffbio.operators.protein.secondary_structure import (
    SS_HELIX,
    SS_LOOP,
    SS_STRAND,
)

# -- Synthetic data constants ------------------------------------------------

# Helix: ~1.5 A rise per residue, 3.6 residues per turn in xy-plane
HELIX_RISE_PER_RESIDUE = 1.5
HELIX_RESIDUES_PER_TURN = 3.6

# Strand: ~3.3 A rise per residue, fully extended
STRAND_RISE_PER_RESIDUE = 3.3

# Standard backbone bond geometry (Angstrom)
BOND_N_CA = 1.47
BOND_CA_C = 1.52
BOND_C_O = 1.24


# ---------------------------------------------------------------------------
# Synthetic backbone generation
# ---------------------------------------------------------------------------


def _generate_helix_backbone(
    n_residues: int,
    offset: jnp.ndarray,
) -> jnp.ndarray:
    """Generate approximate alpha-helix backbone coordinates.

    Uses a helical rise of ~1.5 A per residue and 3.6 residues per
    turn in the xy-plane. For each residue, N/CA/C/O atoms are placed
    with roughly correct bond geometry.

    Args:
        n_residues: Number of residues to generate.
        offset: (3,) translation vector for the helix start.

    Returns:
        Array of shape (n_residues, 4, 3) with atom order N, CA, C, O.
    """
    coords = []
    for i in range(n_residues):
        angle = 2.0 * jnp.pi * i / HELIX_RESIDUES_PER_TURN
        z = HELIX_RISE_PER_RESIDUE * i
        # Helix radius ~2.3 A
        radius = 2.3
        x = radius * jnp.cos(angle)
        y = radius * jnp.sin(angle)
        ca = jnp.array([x, y, z]) + offset

        # Place N slightly before CA along the helix
        angle_n = 2.0 * jnp.pi * (i - 0.3) / HELIX_RESIDUES_PER_TURN
        z_n = HELIX_RISE_PER_RESIDUE * (i - 0.3)
        n = (
            jnp.array(
                [
                    radius * jnp.cos(angle_n),
                    radius * jnp.sin(angle_n),
                    z_n,
                ]
            )
            + offset
        )

        # Place C slightly after CA
        angle_c = 2.0 * jnp.pi * (i + 0.3) / HELIX_RESIDUES_PER_TURN
        z_c = HELIX_RISE_PER_RESIDUE * (i + 0.3)
        c = (
            jnp.array(
                [
                    radius * jnp.cos(angle_c),
                    radius * jnp.sin(angle_c),
                    z_c,
                ]
            )
            + offset
        )

        # O is displaced from C perpendicular to the CA-C bond
        ca_c_vec = c - ca
        ca_c_norm = ca_c_vec / (jnp.linalg.norm(ca_c_vec) + 1e-8)
        # Perpendicular direction in the xy-plane
        perp = jnp.array([-ca_c_norm[1], ca_c_norm[0], 0.0])
        o = c + perp * BOND_C_O

        coords.append(jnp.stack([n, ca, c, o]))

    return jnp.stack(coords)


def _generate_strand_backbone(
    n_residues: int,
    offset: jnp.ndarray,
) -> jnp.ndarray:
    """Generate approximate beta-strand backbone coordinates.

    Uses an extended conformation with ~3.3 A rise per residue
    along the z-axis. Alternating slight zig-zag in x to mimic
    the pleated sheet geometry.

    Args:
        n_residues: Number of residues to generate.
        offset: (3,) translation vector for the strand start.

    Returns:
        Array of shape (n_residues, 4, 3) with atom order N, CA, C, O.
    """
    coords = []
    for i in range(n_residues):
        z = STRAND_RISE_PER_RESIDUE * i
        # Alternating zig-zag in x
        x = 0.5 * ((-1.0) ** i)
        y = 0.0
        ca = jnp.array([x, y, z]) + offset

        # N before CA along z
        n = ca + jnp.array([0.0, 0.0, -BOND_N_CA])

        # C after CA along z
        c = ca + jnp.array([0.0, 0.0, BOND_CA_C])

        # O perpendicular to backbone in y-direction
        o = c + jnp.array([0.0, BOND_C_O, 0.0])

        coords.append(jnp.stack([n, ca, c, o]))

    return jnp.stack(coords)


def _generate_coil_backbone(
    n_residues: int,
    offset: jnp.ndarray,
    seed: int = 99,
) -> jnp.ndarray:
    """Generate approximate random coil backbone coordinates.

    Places residues along the z-axis with small random perturbations
    to simulate an irregular backbone.

    Args:
        n_residues: Number of residues to generate.
        offset: (3,) translation vector for the coil start.
        seed: Random seed for perturbations.

    Returns:
        Array of shape (n_residues, 4, 3) with atom order N, CA, C, O.
    """
    import jax  # noqa: PLC0415

    key = jax.random.key(seed)
    coords = []
    for i in range(n_residues):
        key, k1, k2, k3 = jax.random.split(key, 4)
        z = 2.5 * i  # Slightly shorter rise than strand
        perturbation = jax.random.normal(k1, (3,)) * 0.8
        ca = jnp.array([0.0, 0.0, z]) + offset + perturbation

        n_perturb = jax.random.normal(k2, (3,)) * 0.2
        n = ca + jnp.array([0.0, 0.0, -BOND_N_CA]) + n_perturb

        c_perturb = jax.random.normal(k3, (3,)) * 0.2
        c = ca + jnp.array([0.0, 0.0, BOND_CA_C]) + c_perturb

        o = c + jnp.array([0.0, BOND_C_O, 0.0])

        coords.append(jnp.stack([n, ca, c, o]))

    return jnp.stack(coords)


def generate_synthetic_protein(
    *,
    quick: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a synthetic protein with known secondary structure.

    Concatenates helix, strand, and coil segments into a single
    protein backbone. Returns the coordinates and a per-residue
    ground truth label array.

    Args:
        quick: If True, use shorter segments for faster execution.

    Returns:
        Tuple of (coordinates, labels) where coordinates has shape
        (1, total_residues, 4, 3) and labels has shape
        (total_residues,) with values SS_LOOP, SS_HELIX, SS_STRAND.
    """
    n_helix = 10 if quick else 20
    n_strand = 8 if quick else 15
    n_coil = 8 if quick else 15

    # Offset each segment so they don't overlap spatially
    helix_coords = _generate_helix_backbone(
        n_helix,
        offset=jnp.array([0.0, 0.0, 0.0]),
    )
    strand_coords = _generate_strand_backbone(
        n_strand,
        offset=jnp.array([20.0, 0.0, 0.0]),
    )
    coil_coords = _generate_coil_backbone(
        n_coil,
        offset=jnp.array([40.0, 0.0, 0.0]),
    )

    # Concatenate along residue axis
    all_coords = jnp.concatenate(
        [helix_coords, strand_coords, coil_coords],
        axis=0,
    )
    # Add batch dimension
    batched_coords = all_coords[None, :, :, :]

    # Ground truth labels
    labels = jnp.concatenate(
        [
            jnp.full((n_helix,), SS_HELIX, dtype=jnp.int32),
            jnp.full((n_strand,), SS_STRAND, dtype=jnp.int32),
            jnp.full((n_coil,), SS_LOOP, dtype=jnp.int32),
        ]
    )

    return batched_coords, labels


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _compute_q3_accuracy(
    ss_indices: jnp.ndarray,
    labels: jnp.ndarray,
    n_helix: int,
    n_strand: int,
    n_coil: int,
) -> dict[str, float]:
    """Compute per-class and overall Q3 accuracy.

    Args:
        ss_indices: Predicted SS indices of shape (total_residues,).
        labels: Ground truth labels of shape (total_residues,).
        n_helix: Number of helix residues.
        n_strand: Number of strand residues.
        n_coil: Number of coil residues.

    Returns:
        Dictionary with q3_helix, q3_strand, q3_coil, q3_overall.
    """
    helix_pred = ss_indices[:n_helix]
    helix_true = labels[:n_helix]
    q3_helix = float(jnp.mean(helix_pred == helix_true))

    strand_start = n_helix
    strand_end = n_helix + n_strand
    strand_pred = ss_indices[strand_start:strand_end]
    strand_true = labels[strand_start:strand_end]
    q3_strand = float(jnp.mean(strand_pred == strand_true))

    coil_pred = ss_indices[strand_end:]
    coil_true = labels[strand_end:]
    q3_coil = float(jnp.mean(coil_pred == coil_true))

    total = n_helix + n_strand + n_coil
    correct = (
        int(jnp.sum(helix_pred == helix_true))
        + int(jnp.sum(strand_pred == strand_true))
        + int(jnp.sum(coil_pred == coil_true))
    )
    q3_overall = correct / total if total > 0 else 0.0

    return {
        "q3_helix": q3_helix,
        "q3_strand": q3_strand,
        "q3_coil": q3_coil,
        "q3_overall": q3_overall,
    }


def _validate_ss_onehot(
    ss_onehot: jnp.ndarray,
    batch: int,
    length: int,
) -> dict[str, bool]:
    """Validate ss_onehot shape and value constraints.

    Args:
        ss_onehot: Predicted soft SS assignments.
        batch: Expected batch dimension.
        length: Expected sequence length.

    Returns:
        Dictionary with shape and value validity flags.
    """
    correct_shape = ss_onehot.shape == (batch, length, 3)

    values_in_range = bool(jnp.all(ss_onehot >= -1e-6) and jnp.all(ss_onehot <= 1.0 + 1e-6))
    sums = jnp.sum(ss_onehot, axis=-1)
    sums_valid = bool(jnp.allclose(sums, 1.0, atol=1e-4))

    return {
        "ss_onehot_correct_shape": correct_shape,
        "ss_onehot_values_valid": values_in_range and sums_valid,
    }


def _validate_hbond_map(
    hbond_map: jnp.ndarray,
) -> dict[str, bool]:
    """Validate hydrogen bond map properties.

    Args:
        hbond_map: Predicted H-bond matrix.

    Returns:
        Dictionary with finiteness and non-negativity flags.
    """
    return {
        "hbond_map_finite": bool(jnp.all(jnp.isfinite(hbond_map))),
        "hbond_map_nonnegative": bool(jnp.all(hbond_map >= -1e-6)),
    }
