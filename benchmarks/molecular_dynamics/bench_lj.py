#!/usr/bin/env python3
"""Lennard-Jones molecular dynamics benchmark: DiffBio vs jax-md.

Evaluates DiffBio's ForceFieldOperator and MDIntegratorOperator on a
standard LJ fluid system, comparing throughput and energy conservation
against a direct jax-md baseline.

Protocol:
  1. Generate FCC lattice of particles (4096 quick / 64000 full)
  2. Compute energy + forces via DiffBio ForceFieldOperator
  3. Run NVE integration via DiffBio MDIntegratorOperator
  4. Run equivalent jax-md simulation for direct comparison
  5. Report steps/sec, energy drift, gradient flow

Usage:
    python benchmarks/molecular_dynamics/bench_lj.py
    python benchmarks/molecular_dynamics/bench_lj.py --quick
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_md import energy, quantity, simulate, space

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.md import MD_BASELINES
from diffbio.operators.molecular_dynamics.force_field import (
    ForceFieldConfig,
    ForceFieldOperator,
)
from diffbio.operators.molecular_dynamics.integrator import (
    MDIntegratorConfig,
    MDIntegratorOperator,
)

logger = logging.getLogger(__name__)

# LJ parameters (reduced units)
_SIGMA = 1.0
_EPSILON = 1.0
_CUTOFF = 2.5
_DT = 0.001
_MASS = 1.0
_DIMENSION = 3

_CONFIG = DiffBioBenchmarkConfig(
    name="molecular_dynamics/lj",
    domain="molecular_dynamics",
    n_iterations_quick=5,
    n_iterations_full=20,
)


def _generate_fcc_lattice(
    n_particles: int,
    sigma: float = _SIGMA,
) -> tuple[jnp.ndarray, float]:
    """Generate an FCC lattice of particles in a periodic box.

    The lattice spacing is set to ``sigma * 2^(1/6)`` (LJ equilibrium
    distance) so particles start near the energy minimum.

    Args:
        n_particles: Desired number of particles (rounded up to
            nearest multiple of 4 for FCC).
        sigma: LJ sigma parameter.

    Returns:
        Tuple of (positions, box_size) where positions has shape
        ``(n_actual, 3)`` and box_size is the cubic box edge length.
    """
    n_cells_per_dim = max(1, math.ceil((n_particles / 4) ** (1.0 / 3.0)))
    spacing = sigma * (2.0 ** (1.0 / 6.0))
    box_size = n_cells_per_dim * spacing

    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )

    positions = []
    for ix in range(n_cells_per_dim):
        for iy in range(n_cells_per_dim):
            for iz in range(n_cells_per_dim):
                cell_origin = np.array([ix, iy, iz], dtype=np.float64)
                for b in basis:
                    pos = (cell_origin + b) * spacing
                    positions.append(pos)

    positions_arr = jnp.array(np.array(positions), dtype=jnp.float32)
    return positions_arr, float(box_size)


def _generate_velocities(
    n_particles: int,
    temperature: float,
    mass: float,
    key: jax.Array,
) -> jnp.ndarray:
    """Generate Maxwell-Boltzmann velocities at a given temperature.

    Velocities are drawn from a Gaussian and shifted so the center
    of mass velocity is zero.

    Args:
        n_particles: Number of particles.
        temperature: Target temperature (kT in reduced units).
        mass: Particle mass.
        key: JAX PRNG key.

    Returns:
        Velocities array of shape ``(n_particles, 3)``.
    """
    std = jnp.sqrt(temperature / mass)
    velocities = jax.random.normal(key, shape=(n_particles, _DIMENSION)) * std
    velocities = velocities - jnp.mean(velocities, axis=0)
    return velocities


def _run_jaxmd_baseline(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    box_size: float,
    n_steps: int,
    dt: float,
) -> dict[str, float]:
    """Run jax-md NVE simulation and measure throughput.

    Args:
        positions: Initial positions ``(n_particles, 3)``.
        velocities: Initial velocities ``(n_particles, 3)``.
        box_size: Periodic box size.
        n_steps: Number of integration steps.
        dt: Time step.

    Returns:
        Dict with ``steps_per_sec``, ``initial_energy``, and
        ``final_energy`` keys.
    """
    displacement_fn, shift_fn = space.periodic(jnp.array(box_size))
    energy_fn = energy.lennard_jones_pair(
        displacement_fn,
        sigma=jnp.array(_SIGMA),
        epsilon=jnp.array(_EPSILON),
        r_cutoff=jnp.array(_CUTOFF * _SIGMA),
    )

    init_fn, step_fn = simulate.nve(energy_fn, shift_fn, dt=dt)

    key = jax.random.PRNGKey(0)
    sim_state = init_fn(key, positions, kT=1.0)

    pe_init = float(energy_fn(sim_state.position))
    ke_init = float(quantity.kinetic_energy(momentum=sim_state.momentum))
    total_init = pe_init + ke_init

    jit_step = jax.jit(step_fn)

    # Warmup
    sim_state = jit_step(sim_state)
    sim_state.position.block_until_ready()

    # Timed run
    start = time.perf_counter()
    for _ in range(n_steps - 1):
        sim_state = jit_step(sim_state)
    sim_state.position.block_until_ready()
    elapsed = time.perf_counter() - start

    pe_final = float(energy_fn(sim_state.position))
    ke_final = float(quantity.kinetic_energy(momentum=sim_state.momentum))
    total_final = pe_final + ke_final

    steps_per_sec = (n_steps - 1) / elapsed if elapsed > 0 else 0.0

    return {
        "steps_per_sec": steps_per_sec,
        "initial_energy": total_init,
        "final_energy": total_final,
    }


class LJBenchmark(DiffBioBenchmark):
    """Lennard-Jones molecular dynamics benchmark.

    Compares DiffBio's ForceFieldOperator + MDIntegratorOperator
    against a direct jax-md baseline on a standard LJ fluid system.
    """

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "",
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        """Run DiffBio and jax-md LJ simulations, compare."""
        n_particles = 4096 if self.quick else 64000
        n_steps = 100 if self.quick else 1000
        temperature = 0.5  # Below LJ melting ~0.7

        # 1. Generate FCC lattice
        logger.info("Generating FCC lattice...")
        positions, box_size = _generate_fcc_lattice(n_particles, sigma=_SIGMA)
        n_actual = positions.shape[0]
        logger.info("  %d particles, box_size=%.3f", n_actual, box_size)

        key = jax.random.PRNGKey(42)
        velocities = _generate_velocities(n_actual, temperature, _MASS, key)

        # 2. DiffBio ForceFieldOperator: energy + forces
        logger.info("Running DiffBio ForceFieldOperator...")
        ff_config = ForceFieldConfig(
            potential_type="lennard_jones",
            sigma=_SIGMA,
            epsilon=_EPSILON,
            cutoff=_CUTOFF,
            box_size=box_size,
        )
        rngs = nnx.Rngs(42)
        ff_operator = ForceFieldOperator(ff_config, rngs=rngs)

        ff_input = {"positions": positions}
        ff_result, _, _ = ff_operator.apply(ff_input, {}, None)

        diffbio_energy = float(ff_result["energy"])
        logger.info("  Energy: %.4f", diffbio_energy)

        # 3. DiffBio MDIntegratorOperator: NVE simulation
        logger.info("Running DiffBio MDIntegratorOperator...")
        int_config = MDIntegratorConfig(
            integrator_type="velocity_verlet",
            dt=_DT,
            n_steps=n_steps,
            box_size=box_size,
            potential_type="lennard_jones",
            sigma=_SIGMA,
            epsilon=_EPSILON,
            mass=_MASS,
        )
        integrator = MDIntegratorOperator(int_config, rngs=rngs)

        int_input = {
            "positions": positions,
            "velocities": velocities,
        }
        int_result, _, _ = integrator.apply(int_input, {}, None)

        # 4. Energy conservation (drift)
        trajectory = int_result["trajectory"]
        initial_pe = float(
            ff_operator._energy_fn(  # noqa: SLF001
                trajectory[0]
            )
        )
        final_pe = float(
            ff_operator._energy_fn(  # noqa: SLF001
                trajectory[-1]
            )
        )

        ke_init = float(quantity.kinetic_energy(velocity=velocities, mass=jnp.array(_MASS)))
        final_vel = int_result["velocities"]
        ke_final = float(quantity.kinetic_energy(velocity=final_vel, mass=jnp.array(_MASS)))

        total_init = initial_pe + ke_init
        total_final = final_pe + ke_final
        energy_drift = (
            abs(total_final - total_init) / abs(total_init) if abs(total_init) > 1e-10 else 0.0
        )

        logger.info("  Energy drift: %.6f", energy_drift)

        # 5. jax-md direct baseline
        logger.info("Running jax-md baseline...")
        jaxmd_results = _run_jaxmd_baseline(positions, velocities, box_size, n_steps, _DT)
        jaxmd_steps_per_sec = jaxmd_results["steps_per_sec"]
        jaxmd_drift = (
            abs(jaxmd_results["final_energy"] - jaxmd_results["initial_energy"])
            / abs(jaxmd_results["initial_energy"])
            if abs(jaxmd_results["initial_energy"]) > 1e-10
            else 0.0
        )
        logger.info(
            "  jax-md: %.1f steps/sec, drift=%.6f",
            jaxmd_steps_per_sec,
            jaxmd_drift,
        )

        # Loss function for gradient check
        def loss_fn(
            model: ForceFieldOperator,
            d: dict[str, Any],
        ) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["energy"])

        # Throughput iterate uses the integrator
        timing_integrator = MDIntegratorOperator(int_config, rngs=rngs)

        quality: dict[str, float] = {
            "energy_drift": energy_drift,
            "initial_energy": total_init,
            "final_energy": total_final,
            "jaxmd_steps_per_sec": jaxmd_steps_per_sec,
            "jaxmd_energy_drift": jaxmd_drift,
        }

        return {
            "metrics": quality,
            "operator": ff_operator,
            "input_data": ff_input,
            "loss_fn": loss_fn,
            "n_items": n_steps,
            "iterate_fn": lambda: timing_integrator.apply(int_input, {}, None),
            "baselines": MD_BASELINES,
            "dataset_info": {
                "n_particles": n_actual,
                "dimension": _DIMENSION,
                "box_size": box_size,
                "lattice": "fcc",
            },
            "operator_config": {
                "n_particles": n_actual,
                "n_steps": n_steps,
                "dt": _DT,
                "sigma": _SIGMA,
                "epsilon": _EPSILON,
                "cutoff": _CUTOFF,
                "box_size": box_size,
                "temperature": temperature,
                "mass": _MASS,
            },
            "operator_name": ("ForceFieldOperator+MDIntegratorOperator"),
            "dataset_name": "lj_fcc",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(LJBenchmark, _CONFIG)


if __name__ == "__main__":
    main()
