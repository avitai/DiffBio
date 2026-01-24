"""Shared JAX-MD primitives for molecular dynamics operators.

This module provides common functions for creating JAX-MD primitives,
following the DRY principle by centralizing displacement, energy, and
force function creation.
"""

from enum import Enum
from typing import Callable

from jax_md import energy, quantity, space


class PotentialType(str, Enum):
    """Enumeration of supported potential types."""

    LENNARD_JONES = "lennard_jones"
    SOFT_SPHERE = "soft_sphere"
    MORSE = "morse"


def create_displacement_fn(
    box_size: float | None = None,
) -> tuple[Callable, Callable]:
    """Create displacement and shift functions based on boundary conditions.

    Args:
        box_size: Size of periodic box. None for non-periodic (free) boundaries.

    Returns:
        Tuple of (displacement_fn, shift_fn) where:
            - displacement_fn: computes displacement vector between two points
            - shift_fn: applies displacement to a position respecting boundaries
    """
    if box_size is not None:
        return space.periodic(box_size)
    else:
        return space.free()


def create_energy_fn(
    displacement_fn: Callable,
    potential_type: PotentialType | str = PotentialType.LENNARD_JONES,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    cutoff: float | None = None,
    alpha: float = 5.0,
) -> Callable:
    """Create energy function for the specified potential.

    Args:
        displacement_fn: Displacement function from create_displacement_fn.
        potential_type: Type of potential to use.
        sigma: Length scale parameter (particle diameter).
        epsilon: Energy scale parameter (well depth).
        cutoff: Cutoff distance for interactions. None for no cutoff.
        alpha: Morse potential width parameter (only for morse).

    Returns:
        Energy function that takes positions and returns total energy.

    Raises:
        ValueError: If potential_type is not recognized.
    """
    # Convert string to enum if needed
    if isinstance(potential_type, str):
        try:
            potential_type = PotentialType(potential_type)
        except ValueError as err:
            raise ValueError(f"Unknown potential type: {potential_type}") from err

    if potential_type == PotentialType.LENNARD_JONES:
        kwargs = {
            "displacement_or_metric": displacement_fn,
            "sigma": sigma,
            "epsilon": epsilon,
        }
        if cutoff is not None:
            kwargs["r_cutoff"] = cutoff * sigma
        return energy.lennard_jones_pair(**kwargs)

    elif potential_type == PotentialType.SOFT_SPHERE:
        return energy.soft_sphere_pair(
            displacement_fn,
            sigma=sigma,
            epsilon=epsilon,
        )

    elif potential_type == PotentialType.MORSE:
        return energy.morse_pair(
            displacement_fn,
            sigma=sigma,
            epsilon=epsilon,
            alpha=alpha,
        )

    else:
        raise ValueError(f"Unknown potential type: {potential_type}")


def create_force_fn(energy_fn: Callable) -> Callable:
    """Create force function from energy function.

    Forces are computed as the negative gradient of the energy.

    Args:
        energy_fn: Energy function that takes positions and returns energy.

    Returns:
        Force function that takes positions and returns forces.
    """
    return quantity.force(energy_fn)
