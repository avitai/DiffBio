#!/usr/bin/env python3
"""Molecular Dynamics Benchmark for DiffBio.

This benchmark evaluates DiffBio's molecular dynamics operators:
- ForceFieldOperator (energy and forces via Lennard-Jones potential)
- MDIntegratorOperator (velocity Verlet time integration)

Metrics:
- Force field correctness (finite energy, correct force shapes)
- Integrator trajectory shape and position evolution
- Energy conservation over NVE simulation
- Gradient flow through the force field

Usage:
    python benchmarks/molecular_dynamics/molecular_dynamics_benchmark.py
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._gradient import GradientFlowResult, check_gradient_flow
from diffbio.operators.molecular_dynamics.force_field import (
    ForceFieldConfig,
    ForceFieldOperator,
)
from diffbio.operators.molecular_dynamics.integrator import (
    MDIntegratorConfig,
    MDIntegratorOperator,
)


# ------------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------------


def _generate_lattice_positions(
    n_particles: int,
    dim: int,
    box_size: float,
    seed: int = 42,
) -> jnp.ndarray:
    """Generate lattice positions with small random perturbations.

    Places particles on a regular cubic lattice that fits inside the
    periodic box, then applies a small Gaussian perturbation so that
    no two particles overlap exactly.

    Args:
        n_particles: Number of particles.
        dim: Spatial dimension (typically 3).
        box_size: Side length of the periodic box.
        seed: Random seed for perturbation.

    Returns:
        Array of shape ``(n_particles, dim)`` with positions inside
        ``[0, box_size)``.
    """
    # Determine grid side length (ceil of n_particles^(1/dim))
    side = 1
    while side**dim < n_particles:
        side += 1

    spacing = box_size / side

    # Build full grid, then take the first n_particles points
    indices = jnp.arange(side**dim)
    coords = []
    for d in range(dim):
        coords.append((indices // (side**d)) % side)
    grid = jnp.stack(coords, axis=-1).astype(jnp.float32) * spacing
    grid = grid[:n_particles]

    # Small perturbation (10% of spacing)
    key = jax.random.key(seed)
    perturbation = jax.random.normal(key, grid.shape) * spacing * 0.1
    positions = jnp.mod(grid + perturbation, box_size)
    return positions


def _generate_maxwell_boltzmann_velocities(
    n_particles: int,
    dim: int,
    mass: float = 1.0,
    kT: float = 1.0,
    seed: int = 73,
) -> jnp.ndarray:
    """Draw velocities from the Maxwell-Boltzmann distribution.

    Each velocity component is sampled from N(0, sqrt(kT / mass)).
    The centre-of-mass velocity is then removed so the system has
    zero total momentum.

    Args:
        n_particles: Number of particles.
        dim: Spatial dimension.
        mass: Particle mass (uniform).
        kT: Thermal energy (k_B * T).
        seed: Random seed.

    Returns:
        Array of shape ``(n_particles, dim)``.
    """
    key = jax.random.key(seed)
    sigma = jnp.sqrt(kT / mass)
    velocities = jax.random.normal(key, (n_particles, dim)) * sigma
    # Remove centre-of-mass motion
    velocities = velocities - jnp.mean(velocities, axis=0, keepdims=True)
    return velocities


# ------------------------------------------------------------------
# Individual benchmark tests
# ------------------------------------------------------------------


def _test_force_field(
    positions: jnp.ndarray,
    box_size: float,
) -> dict[str, Any]:
    """Verify force field produces finite energy and correct-shape forces.

    Args:
        positions: Particle positions ``(n_particles, dim)``.
        box_size: Periodic box side length.

    Returns:
        Dictionary of force field metrics.
    """
    print("\n  Testing ForceFieldOperator...")

    config = ForceFieldConfig(
        potential_type="lennard_jones",
        sigma=1.0,
        epsilon=1.0,
        cutoff=2.5,
        box_size=box_size,
    )
    operator = ForceFieldOperator(config, rngs=nnx.Rngs(42))

    data: dict[str, Any] = {"positions": positions}
    result, _, _ = operator.apply(data, {}, None)

    energy_val = result["energy"]
    forces = result["forces"]

    energy_is_finite = bool(jnp.isfinite(energy_val))
    forces_shape_correct = forces.shape == positions.shape
    forces_are_finite = bool(jnp.all(jnp.isfinite(forces)))

    print(f"    Energy: {float(energy_val):.6f} (finite={energy_is_finite})")
    print(f"    Forces shape: {forces.shape} (correct={forces_shape_correct})")
    print(f"    Forces finite: {forces_are_finite}")

    return {
        "energy_is_finite": energy_is_finite,
        "energy_value": float(energy_val),
        "forces_shape_correct": forces_shape_correct,
        "forces_are_finite": forces_are_finite,
    }


def _test_integrator(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    box_size: float,
    n_steps: int,
) -> dict[str, Any]:
    """Verify integrator produces correct trajectory and updates positions.

    Args:
        positions: Initial particle positions ``(n_particles, dim)``.
        velocities: Initial particle velocities ``(n_particles, dim)``.
        box_size: Periodic box side length.
        n_steps: Number of integration steps.

    Returns:
        Dictionary of integrator metrics.
    """
    print("\n  Testing MDIntegratorOperator...")

    config = MDIntegratorConfig(
        integrator_type="velocity_verlet",
        dt=0.001,
        n_steps=n_steps,
        box_size=box_size,
        potential_type="lennard_jones",
        sigma=1.0,
        epsilon=1.0,
        mass=1.0,
    )
    integrator = MDIntegratorOperator(config, rngs=nnx.Rngs(42))

    data: dict[str, Any] = {
        "positions": positions,
        "velocities": velocities,
    }
    result, _, _ = integrator.apply(data, {}, None)

    trajectory = result["trajectory"]
    final_positions = result["positions"]

    n_particles, dim = positions.shape
    expected_traj_shape = (n_steps + 1, n_particles, dim)
    trajectory_shape_correct = trajectory.shape == expected_traj_shape

    # Positions should evolve over time
    max_displacement = float(jnp.max(jnp.abs(final_positions - positions)))
    positions_changed = max_displacement > 1e-6

    print(
        f"    Trajectory shape: {trajectory.shape} "
        f"(expected {expected_traj_shape}, "
        f"correct={trajectory_shape_correct})"
    )
    print(f"    Max displacement: {max_displacement:.6f} (changed={positions_changed})")

    return {
        "trajectory_shape_correct": trajectory_shape_correct,
        "positions_changed": positions_changed,
    }


def _test_energy_conservation(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    box_size: float,
    n_steps: int,
) -> dict[str, float]:
    """Check energy conservation in NVE simulation.

    For velocity Verlet integration, total energy (kinetic + potential)
    should be approximately conserved. We compare the initial and final
    total energies.

    Args:
        positions: Initial particle positions ``(n_particles, dim)``.
        velocities: Initial particle velocities ``(n_particles, dim)``.
        box_size: Periodic box side length.
        n_steps: Number of integration steps.

    Returns:
        Dictionary with initial_energy, final_energy, and energy_drift.
    """
    print("\n  Testing energy conservation...")

    mass = 1.0

    # Set up force field for energy evaluation
    ff_config = ForceFieldConfig(
        potential_type="lennard_jones",
        sigma=1.0,
        epsilon=1.0,
        cutoff=2.5,
        box_size=box_size,
    )
    force_field = ForceFieldOperator(ff_config, rngs=nnx.Rngs(42))

    # Compute initial total energy
    init_result, _, _ = force_field.apply({"positions": positions}, {}, None)
    init_pe = float(init_result["energy"])
    init_ke = float(0.5 * mass * jnp.sum(velocities**2))
    initial_energy = init_pe + init_ke

    # Run integrator
    int_config = MDIntegratorConfig(
        integrator_type="velocity_verlet",
        dt=0.001,
        n_steps=n_steps,
        box_size=box_size,
        potential_type="lennard_jones",
        sigma=1.0,
        epsilon=1.0,
        mass=mass,
    )
    integrator = MDIntegratorOperator(int_config, rngs=nnx.Rngs(42))

    int_result, _, _ = integrator.apply(
        {"positions": positions, "velocities": velocities}, {}, None
    )

    # Compute final total energy
    final_result, _, _ = force_field.apply({"positions": int_result["positions"]}, {}, None)
    final_pe = float(final_result["energy"])
    final_velocities = int_result["velocities"]
    final_ke = float(0.5 * mass * jnp.sum(final_velocities**2))
    final_energy = final_pe + final_ke

    energy_drift = abs(final_energy - initial_energy)

    print(f"    Initial energy: {initial_energy:.6f}")
    print(f"    Final energy:   {final_energy:.6f}")
    print(f"    Energy drift:   {energy_drift:.6f}")

    return {
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "energy_drift": energy_drift,
    }


def _test_gradient_flow(
    positions: jnp.ndarray,
    box_size: float,
) -> GradientFlowResult:
    """Verify gradients flow through the force field operator.

    Args:
        positions: Particle positions ``(n_particles, dim)``.
        box_size: Periodic box side length.

    Returns:
        :class:`GradientFlowResult` with gradient_norm and gradient_nonzero.
    """
    print("\n  Testing gradient flow...")

    config = ForceFieldConfig(
        potential_type="lennard_jones",
        sigma=1.0,
        epsilon=1.0,
        cutoff=2.5,
        box_size=box_size,
    )
    operator = ForceFieldOperator(config, rngs=nnx.Rngs(42))

    def loss_fn(
        model: ForceFieldOperator,
        data: dict[str, Any],
    ) -> jnp.ndarray:
        """Sum of forces magnitude as a scalar loss."""
        result, _, _ = model.apply(data, {}, None)
        return jnp.sum(result["forces"] ** 2)

    data: dict[str, Any] = {"positions": positions}
    grad_info = check_gradient_flow(loss_fn, operator, data)

    print(f"    Gradient norm: {grad_info.gradient_norm:.6f}")
    print(f"    Gradient nonzero: {grad_info.gradient_nonzero}")

    return grad_info
