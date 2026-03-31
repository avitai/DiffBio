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
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult
from calibrax.profiling.timing import TimingCollector
from flax import nnx
from jax_md import energy, quantity, simulate, space

from benchmarks._baselines.md import MD_BASELINES
from benchmarks._gradient import check_gradient_flow
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
    # FCC has 4 atoms per unit cell
    n_cells_per_dim = max(
        1, math.ceil((n_particles / 4) ** (1.0 / 3.0))
    )
    spacing = sigma * (2.0 ** (1.0 / 6.0))
    box_size = n_cells_per_dim * spacing

    # FCC basis vectors (fractional coordinates)
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])

    positions = []
    for ix in range(n_cells_per_dim):
        for iy in range(n_cells_per_dim):
            for iz in range(n_cells_per_dim):
                cell_origin = np.array(
                    [ix, iy, iz], dtype=np.float64
                )
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
    velocities = jax.random.normal(
        key, shape=(n_particles, _DIMENSION)
    ) * std
    # Remove center-of-mass drift
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
        sigma=_SIGMA,
        epsilon=_EPSILON,
        r_cutoff=_CUTOFF * _SIGMA,
    )

    init_fn, step_fn = simulate.nve(energy_fn, shift_fn, dt=dt)

    key = jax.random.PRNGKey(0)
    sim_state = init_fn(key, positions, kT=1.0)

    # Compute initial total energy
    pe_init = float(energy_fn(sim_state.position))
    ke_init = float(
        quantity.kinetic_energy(momentum=sim_state.momentum)
    )
    total_init = pe_init + ke_init

    # JIT-compile step function
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

    # Compute final total energy
    pe_final = float(energy_fn(sim_state.position))
    ke_final = float(
        quantity.kinetic_energy(momentum=sim_state.momentum)
    )
    total_final = pe_final + ke_final

    steps_per_sec = (n_steps - 1) / elapsed if elapsed > 0 else 0.0

    return {
        "steps_per_sec": steps_per_sec,
        "initial_energy": total_init,
        "final_energy": total_final,
    }


class LJBenchmark:
    """Lennard-Jones molecular dynamics benchmark.

    Compares DiffBio's ForceFieldOperator + MDIntegratorOperator
    against a direct jax-md baseline on a standard LJ fluid system.

    Args:
        quick: If True, use 4096 particles and 100 steps.
            Otherwise 64000 particles and 1000 steps.
    """

    def __init__(self, *, quick: bool = False) -> None:
        self.quick = quick

    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return a calibrax result."""
        n_particles = 4096 if self.quick else 64000
        n_steps = 100 if self.quick else 1000
        temperature = 0.5  # Reduced units (below LJ melting ~0.7)

        # 1. Generate FCC lattice
        print("Generating FCC lattice...")
        positions, box_size = _generate_fcc_lattice(
            n_particles, sigma=_SIGMA
        )
        n_actual = positions.shape[0]
        print(f"  {n_actual} particles, box_size={box_size:.3f}")

        key = jax.random.PRNGKey(42)
        velocities = _generate_velocities(
            n_actual, temperature, _MASS, key
        )

        # 2. DiffBio ForceFieldOperator: energy + forces
        print("Running DiffBio ForceFieldOperator...")
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
        start = time.perf_counter()
        ff_result, _, _ = ff_operator.apply(ff_input, {}, None)
        ff_result["energy"].block_until_ready()
        ff_time = time.perf_counter() - start

        diffbio_energy = float(ff_result["energy"])
        print(f"  Energy: {diffbio_energy:.4f} ({ff_time:.3f}s)")

        # 3. DiffBio MDIntegratorOperator: NVE simulation
        print("Running DiffBio MDIntegratorOperator...")
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
        start = time.perf_counter()
        int_result, _, _ = integrator.apply(int_input, {}, None)
        int_result["positions"].block_until_ready()
        diffbio_wall = time.perf_counter() - start

        diffbio_steps_per_sec = n_steps / diffbio_wall

        # 4. Energy conservation (drift)
        trajectory = int_result["trajectory"]  # (n_steps+1, N, 3)
        initial_pe = float(
            ff_operator._energy_fn(trajectory[0])  # noqa: SLF001
        )
        final_pe = float(
            ff_operator._energy_fn(trajectory[-1])  # noqa: SLF001
        )

        ke_init = float(
            quantity.kinetic_energy(
                velocity=velocities, mass=_MASS
            )
        )
        final_vel = int_result["velocities"]
        ke_final = float(
            quantity.kinetic_energy(
                velocity=final_vel, mass=_MASS
            )
        )

        total_init = initial_pe + ke_init
        total_final = final_pe + ke_final
        energy_drift = (
            abs(total_final - total_init) / abs(total_init)
            if abs(total_init) > 1e-10
            else 0.0
        )

        print(
            f"  DiffBio: {diffbio_steps_per_sec:.1f} steps/sec, "
            f"drift={energy_drift:.6f}"
        )

        # 5. jax-md direct baseline
        print("Running jax-md baseline...")
        jaxmd_results = _run_jaxmd_baseline(
            positions, velocities, box_size, n_steps, _DT
        )
        jaxmd_steps_per_sec = jaxmd_results["steps_per_sec"]
        jaxmd_drift = (
            abs(
                jaxmd_results["final_energy"]
                - jaxmd_results["initial_energy"]
            )
            / abs(jaxmd_results["initial_energy"])
            if abs(jaxmd_results["initial_energy"]) > 1e-10
            else 0.0
        )
        print(
            f"  jax-md:  {jaxmd_steps_per_sec:.1f} steps/sec, "
            f"drift={jaxmd_drift:.6f}"
        )

        # 6. Gradient flow through force field
        print("Checking gradient flow...")

        def loss_fn(
            model: ForceFieldOperator,
            d: dict[str, Any],
        ) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["energy"])

        grad = check_gradient_flow(loss_fn, ff_operator, ff_input)
        print(
            f"  Gradient norm: {grad.gradient_norm:.4f}, "
            f"nonzero: {grad.gradient_nonzero}"
        )

        # 7. Throughput profiling with TimingCollector
        print("Measuring DiffBio throughput...")
        n_timing_iters = 5 if self.quick else 20
        collector = TimingCollector(warmup_iterations=2)

        # Re-create integrator for timing measurement
        timing_config = MDIntegratorConfig(
            integrator_type="velocity_verlet",
            dt=_DT,
            n_steps=n_steps,
            box_size=box_size,
            potential_type="lennard_jones",
            sigma=_SIGMA,
            epsilon=_EPSILON,
            mass=_MASS,
        )
        timing_integrator = MDIntegratorOperator(
            timing_config, rngs=rngs
        )

        timing = collector.measure_iteration(
            iterator=iter(range(n_timing_iters)),
            num_batches=n_timing_iters,
            process_fn=lambda _: timing_integrator.apply(
                int_input, {}, None
            ),
            count_fn=lambda _: n_steps,
        )
        profiled_steps_per_sec = (
            timing.num_elements / timing.wall_clock_sec
        )
        print(f"  Profiled: {profiled_steps_per_sec:.1f} steps/sec")

        # 8. Comparison table
        baselines = MD_BASELINES
        print("\nComparison Table:")
        header = (
            f"{'Method':<20} {'steps/sec':>12} "
            f"{'Energy Drift':>14}"
        )
        print(header)
        print("-" * len(header))
        print(
            f"{'DiffBio':<20} "
            f"{profiled_steps_per_sec:>12.1f} "
            f"{energy_drift:>14.6f}"
        )
        print(
            f"{'jax-md (local)':<20} "
            f"{jaxmd_steps_per_sec:>12.1f} "
            f"{jaxmd_drift:>14.6f}"
        )
        for name, point in baselines.items():
            sps = point.metrics.get(
                "steps_per_sec", Metric(value=0)
            ).value
            print(
                f"{name + ' (pub.)':<20} "
                f"{sps:>12.1f} "
                f"{'N/A':>14}"
            )

        # 9. Build calibrax BenchmarkResult
        metrics = {
            "steps_per_sec": Metric(value=profiled_steps_per_sec),
            "energy_drift": Metric(value=energy_drift),
            "initial_energy": Metric(value=total_init),
            "final_energy": Metric(value=total_final),
            "jaxmd_steps_per_sec": Metric(
                value=jaxmd_steps_per_sec
            ),
            "jaxmd_energy_drift": Metric(value=jaxmd_drift),
            "force_field_time_sec": Metric(value=ff_time),
            "gradient_norm": Metric(value=grad.gradient_norm),
            "gradient_nonzero": Metric(
                value=1.0 if grad.gradient_nonzero else 0.0
            ),
        }

        return BenchmarkResult(
            name="molecular_dynamics/lj",
            domain="diffbio_benchmarks",
            tags={
                "operator": "ForceFieldOperator+MDIntegratorOperator",
                "potential": "lennard_jones",
                "framework": "diffbio",
            },
            timing=timing,
            metrics=metrics,
            config={
                "n_particles": n_actual,
                "n_steps": n_steps,
                "dt": _DT,
                "sigma": _SIGMA,
                "epsilon": _EPSILON,
                "cutoff": _CUTOFF,
                "box_size": box_size,
                "temperature": temperature,
                "mass": _MASS,
                "quick": self.quick,
            },
            metadata={
                "system_info": {
                    "n_particles": n_actual,
                    "dimension": _DIMENSION,
                    "box_size": box_size,
                    "lattice": "fcc",
                },
            },
        )


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("DiffBio Benchmark: Lennard-Jones Molecular Dynamics")
    mode = "quick (4K particles)" if quick else "full (64K)"
    print(f"Mode: {mode}")
    print("=" * 60)

    bench = LJBenchmark(quick=quick)
    result = bench.run()

    # Save result
    from pathlib import Path  # noqa: PLC0415

    output_dir = Path("benchmarks/results/molecular_dynamics")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(output_dir / "lj.json")
    print(f"\nResult saved to: {output_dir / 'lj.json'}")

    print("\n" + "=" * 60)
    sps = result.metrics["steps_per_sec"].value
    drift = result.metrics["energy_drift"].value
    print(f"DiffBio: {sps:.1f} steps/sec, drift={drift:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
