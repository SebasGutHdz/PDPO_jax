
from typing import List, Optional, Tuple, Callable
import jax
from functools import partial
import jax.numpy as jnp


from jaxtyping import Array, PyTree


from pdpo.spline.interpolation import linear_interpolation_states, cubic_interp, linear_interp
from pdpo.spline.types_interpolation import  ProblemConfig
from pdpo.core.types import (
 TrajectoryArray, TimeStepsArray, EnergyArray,
    ScalarArray
)

from pdpo.energy_model.kinetic import kinetic_energy
from pdpo.energy_model.potential import potential_energy




def lagrangian(
    samples_path: TrajectoryArray,
    times_path: TimeStepsArray,
    problem_config: ProblemConfig,
    key: Optional[Array] = None,
    log_density: Optional[Array] = None,
    score: Optional[Array] = None
) -> Tuple[ScalarArray, ScalarArray, ScalarArray]:
    """
    Compute the action functional (Lagrangian).
    
    Args:
        samples_path: Sample trajectories
        times_path: Time points
        spline_state: Spline configuration
        log_density: Optional log density values (placeholder)
        score: Optional score values (placeholder)
        
    Returns:
        total_action: Total action value
        ke: Kinetic energy component
        pe: Potential energy component
    """
    # Ensure correct time ordering
    # if times_path[0] > times_path[1]:
    #     times_path = jnp.flip(times_path)
    #     samples_path = jnp.flip(samples_path, axis=1)
    p_int = int(problem_config.p)
    
    # Compute kinetic energy
    ke = kinetic_energy(samples_path, times_path, p=p_int, ke_modifier=problem_config.ke_modifier)
    ke_integrated = jnp.trapezoid(ke, times_path) / 2
    
    # Compute potential energy
    if problem_config.potential is None:
        pe = jnp.zeros_like(ke)
    else: 
        pe = potential_energy(samples_path=samples_path,
                          fn_ids=problem_config.potential,
                          coeffs=problem_config.potential_coefficients,
                          key=key)

    # Add entropy/Fisher information terms (placeholders)
    if problem_config.entropy > 0:
        pe = pe + log_density * problem_config.entropy
    if problem_config.fisher > 0:
        pe = pe + score * (problem_config.fisher ** 4 / 8)

    pe_integrated = jnp.trapezoid(pe, times_path)
    
    total_action = ke_integrated + pe_integrated
    
    return total_action, ke_integrated, pe_integrated