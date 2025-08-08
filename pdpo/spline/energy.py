
from typing import List, Optional, Tuple, Callable
import jax
import jax.numpy as jnp


from jaxtyping import Array, PyTree


from pdpo.spline.interpolation import linear_interpolation_states, cubic_interp, linear_interp
from pdpo.spline.types_interpolation import  ProblemConfig
from pdpo.core.types import (
 TrajectoryArray, TimeStepsArray, EnergyArray,
    ScalarArray
)




# =============================================================================
# Energy Functions
# =============================================================================

def kinetic_energy(
    samples_path: TrajectoryArray,
    times_path: TimeStepsArray,
    p: int = 2,
    ke_modifier: Optional[List[Callable]] = None
) -> EnergyArray:
    """
    Computes the kinetic energy along a path of samples using finite differences.
    
    Args:
        samples_path: (batch_size, num_timesteps, dim) tensor of sample trajectories
        times_path: (num_timesteps,) tensor of timepoints
        p: Norm for kinetic energy (default: 2)
        ke_modifier: Optional list of functions to modify kinetic energy
        
    Returns:
        ke: (num_timesteps,) tensor containing kinetic energy at each timestep
    """
    # Compute forward differences
    difference = samples_path[:, 1:, :] - samples_path[:, :-1, :]
    dt = (times_path[1:] - times_path[:-1] + 1e-6).reshape(-1, 1)
    
    # Compute velocity
    velocity = difference / dt
    
    # Compute centered differences for interior points
    m = (velocity[:, 1:, :] + velocity[:, :-1, :]) / 2
    
    # Velocity estimate for trajectory (forward, centered, backward)
    m = jnp.concatenate([
        velocity[:, :1, :],
        m,
        velocity[:, -1:, :]
    ], axis=1)
    
    # Apply KE modifiers if provided (placeholder for opinion dynamics)
    if ke_modifier is not None:
        for modifier_fn in ke_modifier:
            # Placeholder: modifier_fn should take (samples_path, times_path) -> modified_velocity
            pass
    
    # Compute kinetic energy
    ke = jnp.linalg.norm(m, ord=p, axis=-1) ** p
    ke = jnp.mean(ke, axis=0)  # Mean over samples at each time step
    
    return ke


def potential_energy(
    samples_path: TrajectoryArray,
    potential_fns: Optional[List[Callable]] = None
) -> EnergyArray:
    """
    Computes the potential energy along a path of samples.
    
    Args:
        samples_path: (batch_size, num_timesteps, dim) tensor of sample trajectories
        potential_fns: List of potential functions (currently returns zeros)
        
    Returns:
        potential_energy: (num_timesteps,) tensor containing potential energy
    """
    # Placeholder implementation - return zeros
    num_timesteps = samples_path.shape[1]
    return jnp.zeros(num_timesteps)


def lagrangian(
    samples_path: TrajectoryArray,
    times_path: TimeStepsArray,
    problem_config: ProblemConfig,
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
    if times_path[0] > times_path[1]:
        times_path = jnp.flip(times_path)
        samples_path = jnp.flip(samples_path, axis=1)
    
    # Compute kinetic energy
    ke = kinetic_energy(samples_path, times_path, p=problem_config.p)
    ke_integrated = jnp.trapezoid(ke, times_path) / 2
    
    # Compute potential energy
    pe = potential_energy(samples_path)
    
    # Add entropy/Fisher information terms (placeholders)
    if problem_config.entropy > 0:
        pe = pe + log_density * problem_config.entropy
    if problem_config.fisher > 0:
        pe = pe + score * (problem_config.fisher ** 4 / 8)

    pe_integrated = jnp.trapezoid(pe, times_path)
    
    total_action = ke_integrated + pe_integrated
    
    return total_action, ke_integrated, pe_integrated