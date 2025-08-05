"""
This file contains the MatchingMethods abstract class and the particular FM, CFM, and SI matching methods
"""

from abc import ABC,abstractmethod
from typing import Callable,Dict,Any,Optional
import jax
import jax.numpy as jnp
import jax.random as jrn
from jaxtyping import Array,Float,PRNGKeyArray
from flax import nnx

from base import GenerativeModel
from pdpo.core.types import(
    SampleArray,
    TimeArray, 
    VelocityArray,
    TrajectoryArray,
    ObjectiveFunction
)


class MatchingMethod(GenerativeModel):
    """Abstract base class for training NODEs using Matching Methods"""

    def __init__(
            self,
            vf_model: nnx.Module,
            dim: int = 2,
            time_varying: bool = True,
            solver_config: Dict = {"type": 'midpoint', 'num_steps': 100}
    ):
        """
        Define MatchingMethod

        Args: 
            vf_model: Initialized velocity field
            dim: Spatial dimension of the problem
            time_varying: Whethere the vf is time dependent or not. 
        """

        if not isinstance(vf_model, nnx.Module):
            raise ValueError(f"The velocity field has to be an nnx.Module")
        else:
            self.vf_model = vf_model
            self.dim = dim
            self.time_varying = time_varying
            self.solver_config =  solver_config

        # Trainig state
        self.is_trained = False
        self.training_metrics = []

    def __call__(
        self, 
        t: TimeArray, 
        x: SampleArray
    ) -> VelocityArray:
        """Evaluate velocity field v_{theta}(t,x).
        
        Args:
            t: Time points [batch_size] or [batch_size, 1]
            x: Spatial coordinates [batch_size, dim]
            
        Returns:
            Velocity field values [batch_size, dim]
        """
        self._validate_inputs(t, x)
        
        # Ensure t has correct shape
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        elif t.ndim != 2:
            raise ValueError("The time variable does not have the right shape, it should be: (bs,) or (bs,1)")
        
        # Call the velocity field model
        if self.time_varying:
            return self.vf_model(t, x)
        else:
            return self.vf_model(x)
    
    def _ode_func(self, t: float, x: SampleArray) -> VelocityArray:
        """ODE right-hand side function for integration.
        
        Args:
            t: Scalar time point
            x: Current state [batch_size, dim]
            
        Returns:
            Velocity at v(t,\vec{x}) current state [batch_size, dim]
        """
        # Convert scalar time to batch
        t_batch = jnp.full((x.shape[0],), t)
        return self(t_batch, x)
    
    def sample(
        self,
        key: PRNGKeyArray,
        num_samples: int,
        prior_samples: Optional[SampleArray] = None
    ) -> SampleArray:
        """Sample from the learned distribution.
        
        Args:
            key: Random key for sampling
            num_samples: Number of samples to generate
            prior_samples: Optional prior samples (default: standard normal)
            
        Returns:
            Generated samples at t=1 [num_samples, dim]
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before sampling")
        
        # Sample from prior if not provided
        if prior_samples is None:
            prior_samples = jax.random.normal(key, (num_samples, self.dim))
        
        # Integration using configured solver
        num_steps = self.solver_config["num_steps"]
        dt = 1.0 / num_steps
        x = prior_samples
        
        # Simple Euler integration (can be extended for other solvers)
        for i in range(num_steps):
            t = i / num_steps
            v = self._ode_func(t, x)
            x = x + dt * v
        
        return x
    
    def sample_trajectory(
        self,
        key: PRNGKeyArray,
        num_samples: int,
        prior_samples: Optional[SampleArray] = None,
        save_every: int = 1
    ) -> TrajectoryArray:
        """Sample complete trajectories from t=0 to t=1.
        
        Args:
            key: Random key for sampling
            num_samples: Number of samples
            prior_samples: Optional prior samples
            save_every: Save trajectory every N steps
            
        Returns:
            Complete trajectories [num_samples, num_saved_steps, dim]
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before sampling")
        
        if prior_samples is None:
            prior_samples = jax.random.normal(key, (num_samples, self.dim))
        
        num_steps = self.solver_config["num_steps"]
        num_saved = (num_steps // save_every) + 1
        
        # Initialize trajectory storage
        trajectory = jnp.zeros((num_samples, num_saved, self.dim))
        trajectory = trajectory.at[:, 0, :].set(prior_samples)
        
        # Integration with trajectory saving
        dt = 1.0 / num_steps
        x = prior_samples
        save_idx = 1
        
        for i in range(num_steps):
            t = i / num_steps
            v = self._ode_func(t, x)
            x = x + dt * v
            
            # Save trajectory point if needed
            if (i + 1) % save_every == 0 and save_idx < num_saved:
                trajectory = trajectory.at[:, save_idx, :].set(x)
                save_idx += 1
        
        return trajectory
    

