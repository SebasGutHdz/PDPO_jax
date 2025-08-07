"""
MatchingMethod: Abstract base class for generative models used in PDPO boundary initialization.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Callable
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, PyTree

from pdpo.core.types import (
    ModelParams,
    ModelState,
    PRNGKeyArray,
    SampleArray,
    TimeArray
)
from pdpo.generative.optimization.objectives import FlowMatchingObjective,StochasticInterpolantObjective
from pdpo.ode import solvers
from pdpo.ode.solvers import ODESolver
from pdpo.generative.models.base import MatchingMethod
from pdpo.generative.models.interpolant_schedules import InterpolantSchedule




    
   
    
class FlowMatching(MatchingMethod):
    """
    Flow Matching implementation for generative modeling.
    
    Implements the standard Flow Matching objective:
    L = E[||v_θ(t, x_t) - u_t||²]
    where x_t = (1-t)x_0 + t*x_1 and u_t = x_1 - x_0
    """
    
    def __init__(
        self,
        vf_model: nnx.Module,
        optimizer: nnx.Optimizer,
        ode_solver: ODESolver,
        sigma: float = 0.1,
        time_sampling: str =  'uniform',
        scheduler: Optional[Callable] = None,
        reference_sampler: Optional[Callable] = None,
    ):
        """
        Initialize Flow Matching method.
        
        Args:
            vf_model: Velocity field neural network (nnx.Module)
            optimizer: JAX optimizer for training
            scheduler: Optional learning rate scheduler
        """
        super().__init__(
            method_name="fm",
            vf_model=vf_model,
            optimizer=optimizer,
            ode_solver = ode_solver,
            scheduler=scheduler,
            reference_sampler = reference_sampler
        )
        
        self.objective = FlowMatchingObjective(sigma = sigma,time_sampling = time_sampling)

    def compute_loss(
        self,
        vf_model: nnx.Module,
        key: PRNGKeyArray,
        data_batch: SampleArray,
        reference_samples: Optional[SampleArray] = None,
    ) -> Tuple[Float[Array, ""], Dict[str, Any]]:
        """
        Compute Flow Matching loss using existing objective.
        
        Args:
            params: Model parameters
            key: JAX random key
            data_batch: Target samples (ρ₁)
            reference_samples: Optional source samples (ρ₀). If None, use Gaussian
            
        Returns:
            loss: Scalar loss value
            metrics: Dictionary of training metrics
        """
        # Use existing flow_matching_loss from objectives.py
        # Adapt to the actual API in your objectives module
        loss, metrics = self.objective.compute_loss(
            model=vf_model,
            eval_model = self.eval_model,
            key=key,
            data_batch=data_batch,  # Target samples
            reference_samples=reference_samples,  # Source samples (None for Gaussian)
        )
        
        return loss, metrics
    
    
    
    def _velocity_field(
        self,
        params: ModelParams,
        t: float,
        x: SampleArray
    ) -> SampleArray:
        """
        Compute velocity field v_θ(t, x) using the neural network.
        
        Args:
            params: Model parameters
            t: Time value (scalar)
            x: Position samples, shape (batch_size, dim)
            
        Returns:
            velocity: Velocity field v_θ(t, x), shape (batch_size, dim)
        """
        batch_size = x.shape[0]
        
        # Time conditioning - broadcast time to match batch dimension
        t_expanded = jnp.full((batch_size, 1), t)
        
        # Concatenate position and time: input = [x, t]
        model_input = jnp.concatenate([x, t_expanded], axis=-1)
        
        # Forward pass through the velocity field network
        velocity = self.vf_model(model_input)

        return velocity




class StochasticInterpolant(MatchingMethod):
    """
    Stochastic Interpolants implementation for generative modeling.
    
    Implements the generalized SI objective:
    L = E[||v_θ(t, x_t) - u_t||²]
    where x_t = α(t)x_0 + β(t)x_1 + γ(t)ε
    and u_t = α'(t)x_0 + β'(t)x_1 + γ'(t)ε
    """
    
    def __init__(
        self,
        vf_model: nnx.Module,
        optimizer: nnx.Optimizer,
        ode_solver: ODESolver,
        interpolant_schedule: str = "linear",
        schedule_params: Optional[Dict] = None,
        time_sampling: str = 'uniform',
        scheduler: Optional[Callable] = None,
        reference_sampler: Optional[Callable] = None,
    ):
        """
        Initialize Stochastic Interpolants method.
        
        Args:
            vf_model: Velocity field neural network (nnx.Module)
            optimizer: JAX optimizer for training
            ode_solver: ODE solver for trajectory generation
            interpolant_schedule: Type of interpolant schedule ("linear", "trigonometric", "polynomial", "vp")
            schedule_params: Parameters for the interpolant schedule (e.g., {"sigma": 0.1, "power": 2.0})
            time_sampling: Time sampling strategy ('uniform', 'logit_normal')
            scheduler: Optional learning rate scheduler
            reference_sampler: Optional reference distribution sampler
        """
        super().__init__(
            method_name="si",
            vf_model=vf_model,
            optimizer=optimizer,
            ode_solver=ode_solver,
            scheduler=scheduler,
            reference_sampler=reference_sampler
        )
        
        # Setup interpolant schedule
        if schedule_params is None:
            schedule_params = {"sigma": 0.1}
        
        self.schedule = InterpolantSchedule(interpolant_schedule, **schedule_params)
        alpha_fn, beta_fn, gamma_fn = self.schedule.get_coefficient_functions()
        
        # Create objective with coefficient functions
        self.objective = StochasticInterpolantObjective(
            alpha_fn=alpha_fn,
            beta_fn=beta_fn, 
            gamma_fn=gamma_fn,
            time_sampling=time_sampling
        )
    
    def compute_loss(
        self,
        vf_model: nnx.Module,
        key: PRNGKeyArray,
        data_batch: SampleArray,
        reference_samples: Optional[SampleArray] = None,
    ) -> Tuple[Float[Array, ""], Dict[str, Any]]:
        """
        Compute Stochastic Interpolants loss using the objective.
        
        Args:
            vf_model: Velocity field model
            key: JAX random key
            data_batch: Target samples (ρ₁)
            reference_samples: Optional source samples (ρ₀). If None, use Gaussian
            
        Returns:
            loss: Scalar loss value
            metrics: Dictionary of training metrics
        """
        loss, metrics = self.objective.compute_loss(
            model=vf_model,
            eval_model=self.eval_model,
            key=key,
            data_batch=data_batch,
            reference_samples=reference_samples,
        )
        
        return loss, metrics
