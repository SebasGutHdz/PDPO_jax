"""
Energy model components for density path optimization.

This module provides:
- Potential functionals (linear, internal, interaction)
- Pre-defined potential functions
- Kinetic and potential energy computations
- Lagrangian formulation
"""

# Only import the functionals to avoid circular imports
# kinetic, potential, and lagrangian should be imported directly when needed
from pdpo.energy_model.functionals import (
    LinearPotential,
    InternalPotential,
    InteractionPotential,
    Potential
)

from pdpo.energy_model.functionals.functions import (
    quadratic_potential_fn,
    double_well_potential_fn,
    four_well_potential_fn,
    quartic_potential_fn,
    styblinski_tang_potential_fn,
    aggregation_potential_fn,
    zero_potential_fn,
    harmonic_potential_fn,
    obstacle_cost_gmm_fn,
    obstacle_cost_vneck_fn,
    obstacle_cost_stunnel_fn,
    gaussian_well_potential_fn,
    congestion_kernel_fn,
    gaussian_kernel_fn,
    coulomb_kernel_fn,
    power_law_kernel_fn,
    create_potentials,
    create_interaction_kernels
)

__all__ = [
    # Functional classes
    'LinearPotential',
    'InternalPotential',
    'InteractionPotential',
    'Potential',
    # Potential functions
    'quadratic_potential_fn',
    'double_well_potential_fn',
    'four_well_potential_fn',
    'quartic_potential_fn',
    'styblinski_tang_potential_fn',
    'aggregation_potential_fn',
    'zero_potential_fn',
    'harmonic_potential_fn',
    # Obstacle potentials
    'obstacle_cost_gmm_fn',
    'obstacle_cost_vneck_fn',
    'obstacle_cost_stunnel_fn',
    'gaussian_well_potential_fn',
    # Interaction kernels
    'congestion_kernel_fn',
    'gaussian_kernel_fn',
    'coulomb_kernel_fn',
    'power_law_kernel_fn',
    # Factory functions
    'create_potentials',
    'create_interaction_kernels',
]
