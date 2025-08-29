"""
Boundary parameter optimization for PDPO splines using generative matching methods.
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jrn
from flax import nnx

from pdpo.core.types import PRNGKeyArray, SampleArray
from pdpo.generative.models.base import MatchingMethod
from pdpo.spline.types_interpolation import SplineState
from pdpo.spline.dynamics import gen_sample_trajectory
from pdpo.energy_model.lagrangian import lagrangian
from pdpo.spline.types_interpolation import ProblemConfig


def update_boundary_parameters(
    source_method: MatchingMethod,
    target_method: MatchingMethod,
    problem_config: ProblemConfig,
    key: PRNGKeyArray,
    source_samples: SampleArray,
    target_samples: SampleArray,
    reference_samples: SampleArray,
    boundary_weights: Tuple[float, float] = (1.0, 1.0),
    action_weight: float = 0.1,
    num_steps: int = 10
) -> Tuple[nnx.Module, nnx.Module]:
    """
    Update boundary parameters θ₀, θ₁ using matching method losses and action penalty.
    
    Args:
        source_method: MatchingMethod for source boundary (θ₀)
        target_method: MatchingMethod for target boundary (θ₁)  
        problem_config: Problem configuration containing spline state
        key: JAX random key
        source_samples: Ground truth source distribution samples
        target_samples: Ground truth target distribution samples
        reference_samples: Reference distribution samples for pushforward
        boundary_weights: Weights (α₀, α₁) for boundary losses
        action_weight: Weight for action penalty term
        num_steps: Number of optimization steps
        
    Returns:
        Updated source and target velocity field models
    """
    spline_state = problem_config.splinestate
    alpha0,alpha1 = boundary_weights
    
    def boundary_loss_fn(source_vf: nnx.Module, target_vf: nnx.Module, step_key: PRNGKeyArray):
        """Compute combined boundary loss with action penalty."""
        key1, key2, key3 = jrn.split(step_key, 3)
        
        # Compute source boundary loss
        source_loss, _ = source_method.compute_loss(
            model=source_vf,
            key=key1,
            data_batch=source_samples,
            reference_samples=reference_samples
        )
        
        # Compute target boundary loss  
        target_loss, _ = target_method.compute_loss(
            model=target_vf,
            key=key2,
            data_batch=target_samples,
            reference_samples=reference_samples
        )
        
        # Compute action penalty
        action_penalty = _compute_action_penalty(
            source_vf, target_vf, problem_config, key3, reference_samples
        )
        
        # Combined objective
        total_loss = alpha0 * source_loss + alpha1 * target_loss + action_weight * action_penalty

        return total_loss, {
            'source_loss': source_loss,
            'target_loss': target_loss, 
            'action_penalty': action_penalty,
            'total_loss': total_loss
        }
    
    # Optimization loop
    keys = jrn.split(key, num_steps)
    
    for step_key in keys:
        # Compute gradients for both models
        grad_fn = nnx.value_and_grad(
            lambda src, tgt: boundary_loss_fn(src, tgt, step_key),
            argnums=(0, 1),
            has_aux=True
        )
        
        (loss, metrics), (source_grads, target_grads) = grad_fn(
            source_method.vf_model, target_method.vf_model
        )
        
        # Update parameters using method optimizers
        source_method.optimizer.update(source_method.vf_model, source_grads)
        target_method.optimizer.update(target_method.vf_model, target_grads)
    
    return source_method.vf_model, target_method.vf_model


def _compute_action_penalty(
    source_vf: nnx.Module,
    target_vf: nnx.Module, 
    problem_config: ProblemConfig,
    key: PRNGKeyArray,
    reference_samples: SampleArray
) -> float:
    """Compute action penalty for current boundary parameters."""
    spline_state = problem_config.splinestate
    
    # Update spline state with new boundary parameters
    updated_params = (
        nnx.state(source_vf),
        nnx.state(target_vf)
    )
    temp_spline_state = spline_state._replace(boundary_params=updated_params)
    
    # Generate sample trajectory with updated boundaries
    t_traj = jnp.linspace(0, 1, problem_config.discretization_integral)
    
    # Create temporary velocity field for trajectory generation
    temp_vf = source_vf  # Use source as template
    
    samples_path = gen_sample_trajectory(
        spline_state=temp_spline_state,
        vf=temp_vf,
        key=key,
        x0=reference_samples,
        num_samples=reference_samples.shape[0],
        t_traj=t_traj,
        time_steps_node=10,
        solver=spline_state.config.solver
    )
    
    # Compute action using lagrangian
    action_value, _, _ = lagrangian(samples_path, t_traj, problem_config, key=key)
    
    return action_value


def create_boundary_methods(
    method_type: str,
    architecture_config: dict,
    source_data_type: str,
    target_data_type: str,
    key: PRNGKeyArray
) -> Tuple[MatchingMethod, MatchingMethod]:
    """
    Factory function to create matching methods for boundary optimization.
    
    Args:
        method_type: Type of matching method ('FM', 'SI', etc.)
        architecture_config: Model architecture configuration
        source_data_type: Source distribution type
        target_data_type: Target distribution type  
        key: JAX random key
        
    Returns:
        Source and target matching method instances
    """
    from pdpo.generative.models.matching_methods import FlowMatching, StochasticInterpolant
    from pdpo.models.builder import create_model
    from pdpo.ode.solvers import MidpointSolver
    import optax
    
    key1, key2 = jrn.split(key)
    
    # Create models
    source_vf = create_model(**architecture_config, key=key1)
    target_vf = create_model(**architecture_config, key=key2)
    
    # Create optimizers
    source_optimizer = nnx.Optimizer(source_vf, optax.adam(1e-3))
    target_optimizer = nnx.Optimizer(target_vf, optax.adam(1e-3))
    
    # Create ODE solver
    ode_solver = MidpointSolver()
    
    if method_type == 'FM':
        source_method = FlowMatching(
            vf_model=source_vf,
            optimizer=source_optimizer,
            ode_solver=ode_solver
        )
        target_method = FlowMatching(
            vf_model=target_vf,
            optimizer=target_optimizer,
            ode_solver=ode_solver
        )
    elif method_type == 'SI':
        source_method = StochasticInterpolant(
            vf_model=source_vf,
            optimizer=source_optimizer,
            ode_solver=ode_solver
        )
        target_method = StochasticInterpolant(
            vf_model=target_vf,
            optimizer=target_optimizer,
            ode_solver=ode_solver
        )
    else:
        raise ValueError(f"Unknown method type: {method_type}")
    
    return source_method, target_method