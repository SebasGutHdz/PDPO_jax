
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Callable, NamedTuple
import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import lax
from jaxtyping import Array, Float, PyTree
import optax
from flax import nnx

from pdpo.spline.interpolation import linear_interpolation_states, cubic_interp, linear_interp
from pdpo.models.builder import create_model
from pdpo.ode.solvers import ODESolver, EulerSolver, MidpointSolver,sample_trajectory
from pdpo.core.types import (
    SampleArray, TrajectoryArray, TimeStepsArray, EnergyArray,
    ScalarArray, PRNGKeyArray
)

from pdpo.spline.types_interpolation import SplineState, SplineConfig, ProblemConfig
from pdpo.data.toy_datasets import inf_train_gen
from pdpo.spline.interpolation import interp
from pdpo.models.builder import create_model


# =============================================================================
# Sample Generation and Trajectory Functions
# =============================================================================

def gen_sample_trajectory(
    spline_state: SplineState,
    key: PRNGKeyArray,
    x0: Optional[SampleArray] = None,
    num_samples: int = 1000,
    t_traj: Optional[TimeStepsArray] = None,
    time_steps_node: int = 10,
    solver: ODESolver = MidpointSolver
) -> TrajectoryArray:
    """
    Generates samples along the interpolated path by pushing forward samples
    through the sequence of transport maps.
    
    Args:
        spline_state: Spline state
        key: JAX random key
        x0: Optional initial samples (if None, samples from prior_dist)
        num_samples: Number of samples to generate
        t_traj: Time points to evaluate samples at
        time_steps_node: Number of integration steps for NODE solver
        solver: ODE solver type ('euler' or 'midpoint')
        
    Returns:
        samples_path: Shape (num_samples, time_steps, dim)
        
    Note: Placeholders for entropy and Fisher information are included but not implemented.
    """
    dim = spline_state.config.architecture[0]
    
    # Sample from prior if x0 not provided
    if x0 is None:
        z = inf_train_gen(
            data_type = spline_state.prior,
            key=key,
            batch_size=num_samples,
            dim=dim
        )
    else:
        z = x0
    
    # Default time trajectory
    if t_traj is None:
        t_traj = jnp.linspace(0, 1, 10)
    
    time_steps_traj = len(t_traj)
    
    # Get interpolated parameters at trajectory times
    theta_t_list = interp(spline_state, t_traj)
    # Initialize output array
    samples_path = jnp.zeros((num_samples, time_steps_traj, dim))
    
    # Placeholder for entropy/Fisher information
    # In future: augment z with log_density and/or score
    
    # Build parametric map
    key, subkey = jrn.split(key)
    arch = spline_state.config.architecture +[subkey]
    vf = create_model(
        type=spline_state.config.type_architecture,
        args_arch=arch
    )

    # Generate trajectory by pushing forward through each time step
    for i in range(time_steps_traj):
        theta = theta_t_list[i]
        nnx.update(vf,theta)
        # Push forward through NODE
        samples_i = pushforward(
            vf=vf,
            z=z.copy(),
            t_node=time_steps_node,
            solver=solver
        )
        
        samples_path = samples_path.at[:, i, :].set(samples_i)
    
    return samples_path

# =============================================================================
# Pushforward and Pullback Functions
# =============================================================================

def pushforward(
    vf: nnx.Module,
    z: SampleArray,
    t_node: int = 10,
    solver: ODESolver = MidpointSolver
) -> SampleArray:
    """
    Pushes samples forward through a neural ODE with given parameters.
    
    Args:
        vf: nnx.Module representing the velocity field
        z: (batch_size, dim) tensor of input samples
        t_node: Number of integration steps
        solver: ODE solver type
        
    Returns:
        Transformed samples after flowing through the ODE
    """
    
    
        
    # Integrate forward
    x = sample_trajectory(
        vf=vf,
        x0=z,
        ode_solver=solver,
        n_steps=t_node
    )
    return x


def pull_back(
    model: nnx.Module,
    x: SampleArray,
    t_node: int = 10,
    solver: ODESolver = MidpointSolver
) -> SampleArray:
    """
    Pulls samples backward through a neural ODE with given parameters.
    
    Args:
        model: nnx.Module representing the velocity field
        x: (batch_size, dim) tensor of input samples
        t_node: Number of integration steps
        solver: ODE solver type
        
    Returns:
        Original samples obtained by running ODE backward
    """
        
  
    # Integrate backward
    z  = sample_trajectory(
        vf=model,
        x0=x,
        ode_solver=solver,
        n_steps=t_node,
        backward=True
    )
    
    return z
