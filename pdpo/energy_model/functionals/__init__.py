"""
Functional classes for density path optimization.

This module provides classes for computing various energy functionals:
- LinearPotential: Linear potential energy F(ρ) = ∫ U(x)ρ(x)dx
- InternalPotential: Internal energy F(ρ) = ∫ f(ρ(x))dx (entropy, Fisher info)
- InteractionPotential: Interaction energy F(ρ) = ∫∫ W(x-y)ρ(x)ρ(y)dxdy
- Potential: Combined potential managing all three types
"""

from pdpo.energy_model.functionals.linear_functional import LinearPotential
from pdpo.energy_model.functionals.internal_functional import InternalPotential
from pdpo.energy_model.functionals.interaction_functional import InteractionPotential
from pdpo.energy_model.functionals.functional import Potential

__all__ = [
    'LinearPotential',
    'InternalPotential',
    'InteractionPotential',
    'Potential'
]
