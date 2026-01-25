"""Force field operators wrapping JAX-MD.

This module provides differentiable force field operators that compute
molecular energies and forces using JAX-MD's efficient implementations.
"""

from dataclasses import dataclass
from typing import Any

import jax
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array

from diffbio.operators.molecular_dynamics.primitives import (
    PotentialType,
    create_displacement_fn,
    create_energy_fn,
    create_force_fn,
)


@dataclass
class ForceFieldConfig(OperatorConfig):
    """Configuration for force field operator.

    Attributes:
        potential_type: Type of potential ("lennard_jones", "morse", "soft_sphere").
        sigma: Length scale parameter (particle diameter).
        epsilon: Energy scale parameter (well depth).
        cutoff: Cutoff distance for interactions (in units of sigma). None for no cutoff.
        box_size: Size of periodic box. None for non-periodic.
        alpha: Morse potential width parameter (only for morse).
        stochastic: Whether operator uses random sampling.
        stream_name: Optional stream name for data routing.
    """

    potential_type: str = "lennard_jones"
    sigma: float = 1.0
    epsilon: float = 1.0
    cutoff: float | None = 2.5
    box_size: float | None = None
    alpha: float = 5.0  # Morse potential parameter
    stochastic: bool = False
    stream_name: str | None = None


class ForceFieldOperator(OperatorModule):
    """Differentiable force field operator using JAX-MD.

    Computes potential energy and forces for a system of particles using
    classical pairwise potentials. Forces are computed automatically via
    JAX's automatic differentiation.

    Supported potentials:
        - lennard_jones: Standard 12-6 LJ potential
        - morse: Morse potential for bonded interactions
        - soft_sphere: Soft repulsive potential

    Example:
        ```python
        config = ForceFieldConfig(potential_type="lennard_jones", box_size=10.0)
        operator = ForceFieldOperator(config, rngs=nnx.Rngs(42))
        data = {"positions": positions}  # (n_particles, dim)
        result, state, meta = operator.apply(data, {}, None)
        energy = result["energy"]  # scalar
        forces = result["forces"]  # (n_particles, dim)
        ```
    """

    def __init__(self, config: ForceFieldConfig, *, rngs: nnx.Rngs | None = None):
        """Initialize force field operator.

        Args:
            config: Force field configuration.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)
        self.config: ForceFieldConfig = config

        # Pre-create energy and force functions (efficiency: only created once)
        self._displacement_fn, _ = create_displacement_fn(config.box_size)
        self._energy_fn = create_energy_fn(
            self._displacement_fn,
            potential_type=config.potential_type,
            sigma=config.sigma,
            epsilon=config.epsilon,
            cutoff=config.cutoff,
            alpha=config.alpha,
        )
        self._force_fn = create_force_fn(self._energy_fn)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Compute energy and forces for particle positions.

        Args:
            data: Input data containing:
                - positions: Particle positions (n_particles, dim) or
                             (batch, n_particles, dim)
            state: Per-element state (passed through).
            metadata: Optional metadata.
            random_params: Unused random parameters.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of:
                - data with added "energy" and "forces" keys
                - unchanged state
                - unchanged metadata
        """
        positions = data["positions"]

        # Handle batched input
        if positions.ndim == 3:
            # Batched: (batch, n_particles, dim)
            batch_apply = jax.vmap(self._compute_single)
            energy_vals, forces = batch_apply(positions)
        else:
            # Single: (n_particles, dim)
            energy_vals, forces = self._compute_single(positions)

        result = {
            **data,
            "energy": energy_vals,
            "forces": forces,
        }

        return result, state, metadata

    def _compute_single(self, positions: Array) -> tuple[Array, Array]:
        """Compute energy and forces for a single configuration.

        Args:
            positions: Particle positions (n_particles, dim).

        Returns:
            Tuple of (energy, forces).
        """
        # Use pre-created functions from __init__
        total_energy = self._energy_fn(positions)
        forces = self._force_fn(positions)

        return total_energy, forces


def create_force_field(
    potential_type: str | PotentialType = PotentialType.LENNARD_JONES,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    cutoff: float | None = 2.5,
    box_size: float | None = None,
    alpha: float = 5.0,
    seed: int = 42,
) -> ForceFieldOperator:
    """Create a force field operator with specified potential.

    Args:
        potential_type: Type of potential ("lennard_jones", "morse", "soft_sphere")
            or PotentialType enum.
        sigma: Particle diameter (length scale).
        epsilon: Well depth (energy scale).
        cutoff: Cutoff distance in units of sigma. None for no cutoff.
        box_size: Periodic box size. None for non-periodic.
        alpha: Morse potential width parameter.
        seed: Random seed for initialization.

    Returns:
        Configured ForceFieldOperator.
    """
    # Convert enum to string if needed
    if isinstance(potential_type, PotentialType):
        potential_type = potential_type.value

    config = ForceFieldConfig(
        potential_type=potential_type,
        sigma=sigma,
        epsilon=epsilon,
        cutoff=cutoff,
        box_size=box_size,
        alpha=alpha,
    )
    return ForceFieldOperator(config, rngs=nnx.Rngs(seed))


def create_lennard_jones_operator(
    sigma: float = 1.0,
    epsilon: float = 1.0,
    cutoff: float | None = 2.5,
    box_size: float | None = None,
    seed: int = 42,
) -> ForceFieldOperator:
    """Create a Lennard-Jones force field operator.

    This is a convenience function for creating a force field operator
    with Lennard-Jones potential.

    Args:
        sigma: Particle diameter (length scale).
        epsilon: Well depth (energy scale).
        cutoff: Cutoff distance in units of sigma. None for no cutoff.
        box_size: Periodic box size. None for non-periodic.
        seed: Random seed for initialization.

    Returns:
        Configured ForceFieldOperator.
    """
    return create_force_field(
        potential_type=PotentialType.LENNARD_JONES,
        sigma=sigma,
        epsilon=epsilon,
        cutoff=cutoff,
        box_size=box_size,
        seed=seed,
    )
