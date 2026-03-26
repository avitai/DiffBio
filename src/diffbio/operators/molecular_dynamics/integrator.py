"""MD integrator operators wrapping JAX-MD.

This module provides differentiable MD integration operators that evolve
particle positions and velocities over time using JAX-MD's simulators.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jax_md import quantity, simulate

from diffbio.operators.molecular_dynamics.primitives import (
    PotentialType,
    create_displacement_fn,
    create_energy_fn,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MDIntegratorConfig(OperatorConfig):
    # pylint: disable=too-many-instance-attributes
    """Configuration for MD integrator operator.

    Attributes:
        integrator_type: Type of integrator ("velocity_verlet", "nvt_langevin").
        dt: Time step for integration.
        n_steps: Number of integration steps.
        box_size: Size of periodic box. None for non-periodic.
        potential_type: Type of potential ("lennard_jones", "morse", "soft_sphere").
        sigma: Sigma parameter for potential (length scale).
        epsilon: Epsilon parameter for potential (energy scale).
        mass: Particle mass (uniform for all particles).
        kT: Thermal energy for Langevin thermostat.
        gamma: Friction coefficient for Langevin dynamics.
    """

    integrator_type: str = "velocity_verlet"
    dt: float = 0.001
    n_steps: int = 100
    box_size: float | None = 10.0
    potential_type: str = "lennard_jones"
    sigma: float = 1.0
    epsilon: float = 1.0
    mass: float = 1.0
    kT: float = 1.0
    gamma: float = 1.0


@dataclass(frozen=True)
class _MDRuntime:
    """Precomputed JAX-MD runtime callables."""

    displacement_fn: Any
    shift_fn: Any
    energy_fn: Any
    force_fn: Any
    step_fn: Any


class MDIntegratorOperator(OperatorModule):
    """Differentiable MD integrator operator using JAX-MD.

    Evolves particle positions and velocities over time using classical
    molecular dynamics integration schemes.

    Supported integrators:
        - velocity_verlet: Symplectic velocity Verlet (NVE)
        - nvt_langevin: Langevin dynamics for NVT ensemble

    Example:
        ```python
        config = MDIntegratorConfig(dt=0.001, n_steps=1000, box_size=10.0)
        integrator = MDIntegratorOperator(config, rngs=nnx.Rngs(42))
        data = {"positions": positions, "velocities": velocities}
        result, state, meta = integrator.apply(data, {}, None)
        final_positions = result["positions"]
        trajectory = result["trajectory"]
        ```
    """

    def __init__(
        self,
        config: MDIntegratorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize MD integrator operator.

        Args:
            config: Integrator configuration.
            rngs: Flax NNX random number generators.
            name: Optional name for the operator.
        """
        super().__init__(config, rngs=rngs, name=name)
        # Pre-create displacement, energy, and force functions (efficiency: only created once)
        displacement_fn, shift_fn = create_displacement_fn(config.box_size)
        energy_fn = create_energy_fn(
            displacement_fn,
            potential_type=config.potential_type,
            sigma=config.sigma,
            epsilon=config.epsilon,
        )
        force_fn = quantity.force(energy_fn)

        # Pre-create step function based on integrator type
        if config.integrator_type == "velocity_verlet":
            _, step_fn = simulate.nve(energy_fn, shift_fn, dt=config.dt)
        elif config.integrator_type == "nvt_langevin":
            _, step_fn = simulate.nvt_langevin(
                energy_fn,
                shift_fn,
                dt=config.dt,
                kT=config.kT,
                gamma=config.gamma,
            )
        else:
            raise ValueError(f"Unknown integrator type: {config.integrator_type}")

        self._runtime = nnx.static(
            _MDRuntime(
                displacement_fn=displacement_fn,
                shift_fn=shift_fn,
                energy_fn=energy_fn,
                force_fn=force_fn,
                step_fn=step_fn,
            )
        )

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Run MD simulation for specified number of steps.

        Args:
            data: Input data containing:
                - positions: Initial particle positions (n_particles, dim)
                - velocities: Initial particle velocities (n_particles, dim)
            state: Per-element state (passed through).
            metadata: Optional metadata.
            random_params: Unused random parameters.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of:
                - data with updated positions/velocities and trajectory
                - unchanged state
                - unchanged metadata
        """
        positions = data["positions"]
        velocities = data["velocities"]
        config = self.config
        runtime = self._runtime

        # Use pre-created functions from __init__
        # Initialize state with user-provided velocities
        # JAX-MD uses momentum = mass * velocity internally
        initial_force = runtime.force_fn(positions)
        mass = config.mass  # JAX-MD works with scalar mass
        momentum = velocities * mass

        # Create appropriate state based on integrator type
        if config.integrator_type == "velocity_verlet":
            sim_state = simulate.NVEState(
                position=positions,  # pyright: ignore[reportCallIssue]
                momentum=momentum,  # pyright: ignore[reportCallIssue]
                force=initial_force,  # pyright: ignore[reportCallIssue]
                mass=mass,  # pyright: ignore[reportCallIssue]
            )
        elif config.integrator_type == "nvt_langevin":
            # Langevin dynamics requires rng for stochastic forces
            rng_key = jax.random.PRNGKey(42)  # Deterministic for reproducibility
            sim_state = simulate.NVTLangevinState(
                position=positions,  # pyright: ignore[reportCallIssue]
                momentum=momentum,  # pyright: ignore[reportCallIssue]
                force=initial_force,  # pyright: ignore[reportCallIssue]
                mass=mass,  # pyright: ignore[reportCallIssue]
                rng=rng_key,  # pyright: ignore[reportCallIssue]
            )
        else:
            raise ValueError(f"Unknown integrator type: {config.integrator_type}")

        # Run simulation using scan for efficiency
        step_fn = runtime.step_fn  # Capture for use in nested function

        def scan_step(carry, _):
            sim_state = carry
            sim_state = step_fn(sim_state)
            return sim_state, sim_state.position

        final_state, traj_positions = jax.lax.scan(
            scan_step, sim_state, None, length=config.n_steps
        )

        # Stack trajectory (including initial position)
        full_trajectory = jnp.concatenate([positions[jnp.newaxis, ...], traj_positions], axis=0)

        result = {
            **data,
            "positions": final_state.position,
            "velocities": final_state.velocity,
            "trajectory": full_trajectory,
        }

        return result, state, metadata


def create_integrator(
    integrator_type: str = "velocity_verlet",
    dt: float = 0.001,
    n_steps: int = 100,
    box_size: float | None = 10.0,
    potential_type: str | PotentialType = PotentialType.LENNARD_JONES,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    mass: float = 1.0,
    kT: float = 1.0,
    gamma: float = 1.0,
    seed: int = 42,
) -> MDIntegratorOperator:
    """Create an MD integrator operator.

    Args:
        integrator_type: Type of integrator ("velocity_verlet", "nvt_langevin").
        dt: Time step for integration.
        n_steps: Number of integration steps.
        box_size: Periodic box size. None for non-periodic.
        potential_type: Type of potential ("lennard_jones", "morse", "soft_sphere")
            or PotentialType enum.
        sigma: Sigma parameter for potential (length scale).
        epsilon: Epsilon parameter for potential (energy scale).
        mass: Particle mass.
        kT: Thermal energy for Langevin thermostat.
        gamma: Friction coefficient for Langevin dynamics.
        seed: Random seed for initialization.

    Returns:
        Configured MDIntegratorOperator.
    """
    # Convert enum to string if needed
    if isinstance(potential_type, PotentialType):
        potential_type = potential_type.value

    config = MDIntegratorConfig(
        integrator_type=integrator_type,
        dt=dt,
        n_steps=n_steps,
        box_size=box_size,
        potential_type=potential_type,
        sigma=sigma,
        epsilon=epsilon,
        mass=mass,
        kT=kT,
        gamma=gamma,
    )
    return MDIntegratorOperator(config, rngs=nnx.Rngs(seed))


def create_verlet_integrator(
    dt: float = 0.001,
    n_steps: int = 100,
    box_size: float | None = 10.0,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    seed: int = 42,
) -> MDIntegratorOperator:
    """Create a velocity Verlet integrator operator.

    This is a convenience function for creating an NVE integrator
    with velocity Verlet algorithm.

    Args:
        dt: Time step for integration.
        n_steps: Number of integration steps.
        box_size: Periodic box size. None for non-periodic.
        sigma: Sigma parameter for potential.
        epsilon: Epsilon parameter for potential.
        seed: Random seed for initialization.

    Returns:
        Configured MDIntegratorOperator.
    """
    return create_integrator(
        integrator_type="velocity_verlet",
        dt=dt,
        n_steps=n_steps,
        box_size=box_size,
        sigma=sigma,
        epsilon=epsilon,
        seed=seed,
    )
