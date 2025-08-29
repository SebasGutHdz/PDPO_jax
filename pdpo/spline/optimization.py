
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
# from pdpo.spline.energy import kinetic_energy, potential_energy, lagrangian
from pdpo.energy_model.lagrangian import lagrangian
from pdpo.spline.dynamics import gen_sample_trajectory
from pdpo.core.types import (
 TrajectoryArray, TimeStepsArray, EnergyArray,
    ScalarArray, PRNGKeyArray, SampleArray
)
from pdpo.spline.types_interpolation import SplineState, SplineConfig,OptimizationHistory,ProblemConfig
from pdpo.core.config import setup_device
from pdpo.models.builder import create_model
from pdpo.spline.boundary import update_boundary_parameters, create_boundary_methods
from pdpo.core.datasets import inf_train_gen
from pdpo.spline.path_optimization import optimize_path



def optimize_path_with_boundaries(
    problem_config: ProblemConfig,
    boundary_method_type: str,
    key: PRNGKeyArray,
    path_epochs: int = 100,
    boundary_epochs: int = 20,
    alternating_iterations: int = 5,
    **kwargs
) -> Tuple[SplineState, dict]:
    """
    Optimize path with alternating boundary updates using matching methods.
    
    Args:
        problem_config: Problem configuration
        boundary_method_type: Type of matching method for boundaries ('FM', 'SI')
        key: JAX random key
        path_epochs: Epochs for path optimization  
        boundary_epochs: Epochs for boundary optimization
        alternating_iterations: Number of alternating optimization cycles
        
    Returns:
        Optimized spline state and training history
    """
    spline_state = problem_config.splinestate
    key_boundary, key_path = jrn.split(key)
    
    # Create boundary matching methods
    arch_config = {
        'type': spline_state.config.type_architecture,
        'args_arch': spline_state.config.architecture + [key_boundary]
    }
    
    source_method, target_method = create_boundary_methods(
        method_type=boundary_method_type,
        architecture_config=arch_config,
        source_data_type=spline_state.config.data0,
        target_data_type=spline_state.config.data1,
        key=key_boundary
    )
    
    history = {'path_losses': [], 'boundary_losses': [], 'iterations': []}
    
    for iteration in range(alternating_iterations):
        iter_key = jrn.fold_in(key, iteration)
        key1, key2, key3 = jrn.split(iter_key, 3)
        
        # Step 1: Optimize path with fixed boundaries
        optimized_state, _, path_history = optimize_path(
            problem_config=problem_config,
            key=key1,
            epochs=path_epochs,
            **kwargs
        )
        
        # Step 2: Optimize boundaries with fixed path
        
        batch_size = kwargs.get('batch_size', 1000)
        dim = spline_state.config.architecture[0]
        
        # Sample boundary data
        source_samples = inf_train_gen(spline_state.config.data0, key2, batch_size, dim)
        target_samples = inf_train_gen(spline_state.config.data1, key3, batch_size, dim)
        reference_samples = inf_train_gen(spline_state.prior, key2, batch_size, dim)
        
        # Update boundary parameters
        updated_source_vf, updated_target_vf = update_boundary_parameters(
            source_method=source_method,
            target_method=target_method,
            problem_config=problem_config._replace(splinestate=optimized_state),
            key=key3,
            source_samples=source_samples,
            target_samples=target_samples,
            reference_samples=reference_samples,
            num_steps=boundary_epochs
        )
        
        # Update problem config with new boundaries
        new_boundary_params = (
            nnx.state(updated_source_vf),
            nnx.state(updated_target_vf)
        )
        optimized_state = optimized_state._replace(boundary_params=new_boundary_params)
        problem_config = problem_config._replace(splinestate=optimized_state)
        
        # Record history
        history['path_losses'].extend(path_history['lagrangian'])
        history['iterations'].append(iteration)
    
    return optimized_state, history