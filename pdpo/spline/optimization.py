
from typing import List, Optional, Tuple, Callable
from dataclasses import replace
import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import lax
import optax
from jaxtyping import Array, Float, PyTree
from flax import nnx

from jaxtyping import Array, PyTree


from pdpo.spline.interpolation import linear_interpolation_states, cubic_interp, linear_interp
from pdpo.spline.types_interpolation import  ProblemConfig
from pdpo.data.toy_datasets import inf_train_gen
from pdpo.spline.energy import kinetic_energy, potential_energy, lagrangian
from pdpo.spline.dynamics import gen_sample_trajectory
from pdpo.core.types import (
 TrajectoryArray, TimeStepsArray, EnergyArray,
    ScalarArray, PRNGKeyArray, SampleArray
)
from pdpo.spline.types_interpolation import SplineState, SplineConfig,OptimizationHistory,ProblemConfig
from pdpo.core.config import setup_device
from pdpo.models.builder import create_model



# =============================================================================
# Optimization Functions
# =============================================================================

def optimize_path(
    problem_config: ProblemConfig,
    key: PRNGKeyArray,
    epochs: int,
    learning_rate: float = 1e-3,
    t_node: int = 10,
    bs: int = 1000,
    x0: Optional[SampleArray] = None
) -> Tuple[SplineState, OptimizationHistory]:
    """
    Optimizes the interior points of the spline path while keeping endpoints fixed.
    
    Args:
        problem_config: Configuration for the optimization problem
        key: JAX random key
        epochs: Number of optimization iterations
        learning_rate: Learning rate for optimization
        t_node: Number of integration steps for NODE solver
        bs: Batch size for sampling
        x0: Optional fixed initial samples
        
    Returns:
        optimized_state: Updated spline state
        history: Optimization history
    """
    setup_device(problem_config.splinestate.config.device)
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    spline_state = problem_config.splinestate
    opt_state = optimizer.init(spline_state.control_points)
    t_partition = problem_config.discretization_integral
    # Time points for trajectory evaluation
    t_traj = jnp.linspace(0, 1, t_partition)
    
    # Sample initial conditions if not provided
    if x0 is None:
        x0 = inf_train_gen(
            data_type=spline_state.prior,
            key=key,
            batch_size=bs,
            dim=spline_state.config.architecture[0]
        )
        key, subkey = jrn.split(key)
    # History tracking
    history = OptimizationHistory(
        lagrangian=[],
        kinetic=[],
        potential=[],
        iterations=[]
    )

    
    arch = spline_state.config.architecture +[subkey]
    key, subkey = jrn.split(key)
    vf = create_model(
        type=spline_state.config.type_architecture,
        args_arch=arch
    )

    
    # Define loss function
    def loss_fn(control_points, subkey):
        # Create temporary spline state with updated control points
        temp_state = replace(spline_state, control_points=control_points)
        
        # Generate sample trajectory
        samples_path = gen_sample_trajectory(
            temp_state,
            vf,
            key=subkey,
            x0=x0,
            num_samples=bs,
            t_traj=t_traj,
            time_steps_node=t_node,
            solver=spline_state.config.solver
        )
        
        # Compute lagrangian
        total_cost, ke, pe = lagrangian(
            samples_path,
            t_traj,
            problem_config
        )
        
        return total_cost, (ke, pe)
    
    # Training loop
    current_control_points = spline_state.control_points
    
    for epoch in range(epochs):
        epoch_key = jrn.fold_in(key, epoch)
        
        # Compute loss and gradients
        (loss_val, (ke_val, pe_val)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(current_control_points, epoch_key)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        current_control_points = optax.apply_updates(current_control_points, updates)
        
        # Record history
        history.lagrangian.append(float(loss_val))
        history.kinetic.append(float(ke_val))
        history.potential.append(float(pe_val))
        history.iterations.append(epoch)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Lagrangian={loss_val:.4f}, KE={ke_val:.4f}, PE={pe_val:.4f}")
    
    # Create optimized state
    optimized_state = replace(spline_state, control_points=current_control_points)
    
    return optimized_state, history



















# =============================================================================
# Additional Helper Functions
# =============================================================================

def geodesic_warmup(
    spline_state: SplineState,
    key: PRNGKeyArray,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 1000
) -> SplineState:
    """
    Initializes control points by optimizing for geodesic path in Wasserstein space.
    
    Args:
        spline_state: Initial spline state
        key: JAX random key
        num_epochs: Number of warmup iterations
        learning_rate: Learning rate for warmup
        batch_size: Batch size for sampling
        
    Returns:
        Warmed-up spline state
    """
    # Simplified geodesic warmup - optimize only kinetic energy
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(spline_state.control_points)
    
    t_traj = spline_state.time_points
    
    def warmup_loss(control_points, key):
        temp_state = spline_state._replace(control_points=control_points)
        
        # Sample from source distribution
        x0 = jrn.multivariate_normal(
            key,
            spline_state.prior_mean,
            spline_state.prior_cov,
            shape=(batch_size,)
        )
        
        # Generate trajectory
        samples_path = gen_sample_trajectory(
            temp_state,
            key=key,
            x0=x0,
            num_samples=batch_size,
            t_traj=t_traj,
            time_steps_node=10,
            solver='euler'
        )
        
        # Compute only kinetic energy
        ke = kinetic_energy(samples_path, t_traj, p=spline_state.config.p)
        return jnp.trapz(ke, t_traj) / 2
    
    current_control_points = spline_state.control_points
    
    for epoch in range(num_epochs):
        epoch_key = jrn.fold_in(key, epoch)
        
        loss_val, grads = jax.value_and_grad(warmup_loss)(
            current_control_points, epoch_key
        )
        
        updates, opt_state = optimizer.update(grads, opt_state)
        current_control_points = optax.apply_updates(current_control_points, updates)
    
    return spline_state._replace(control_points=current_control_points)