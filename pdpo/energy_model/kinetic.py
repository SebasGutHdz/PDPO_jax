
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
