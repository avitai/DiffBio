"""Molecular dynamics operators for DiffBio.

This module provides differentiable operators for molecular dynamics simulations,
wrapping JAX-MD functionality for seamless integration with DiffBio pipelines.

Operators:
    ForceFieldOperator: Compute energies and forces from particle positions
    MDIntegratorOperator: Time integration for MD simulations

Factory Functions:
    create_force_field: Create force field operator with specified potential
    create_integrator: Create integrator operator with specified type
    create_lennard_jones_operator: Create LJ force field operator (convenience)
    create_verlet_integrator: Create velocity Verlet integrator (convenience)

Enums:
    PotentialType: Enumeration of supported potential types

References:
    Schoenholz & Cubuk (2020). JAX, M.D.: A Framework for Differentiable Physics.
    NeurIPS 2020.
"""

from diffbio.operators.molecular_dynamics.force_field import (
    ForceFieldConfig,
    ForceFieldOperator,
    create_force_field,
    create_lennard_jones_operator,
)
from diffbio.operators.molecular_dynamics.integrator import (
    MDIntegratorConfig,
    MDIntegratorOperator,
    create_integrator,
    create_verlet_integrator,
)
from diffbio.operators.molecular_dynamics.primitives import PotentialType

__all__ = [
    # Enums
    "PotentialType",
    # Force field
    "ForceFieldConfig",
    "ForceFieldOperator",
    "create_force_field",
    "create_lennard_jones_operator",
    # Integrator
    "MDIntegratorConfig",
    "MDIntegratorOperator",
    "create_integrator",
    "create_verlet_integrator",
]
