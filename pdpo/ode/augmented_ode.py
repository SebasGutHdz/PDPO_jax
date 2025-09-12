from typing import Tuple,Optinal,Callable,Union
import jax
import jax.numpy as jnp
from jaxtyping import Array,Float
from flax import nnx
from functools import partial 


from pdpo.core.types import (SampleArray, VelocityArray, TimeArray, PRNGKeyArray,
    TrajectoryArray, ScoreArray, LogDensityArray   
)
from pdpo.ode.solvers import eval_model

def divergence_vf_hutchinson(
   vf_model: nnx.Module,
   t: TimeArray,
   x_in: SampleArray,
   num_samples: int = 1
) :
   """
   Compute the divergence ∇ · v_θ(t, x) using Hutchinson trace estimator.
   
   This is a stochastic estimator that approximates the trace without computing
   the full Jacobian matrix. Useful for high-dimensional problems.
   
   Args:
       vf_model: Velocity field neural network
       t: Time value (scalar or array)
       x_in: Sample positions, shape (batch_size, dim)
       num_samples: Number of random vectors for trace estimation
       
   Returns:
       Divergence values, shape (batch_size,)
       
   Mathematical Details:
       Tr(J) ≈ (1/num_samples) * Σ εᵢᵀ J εᵢ
       where εᵢ are random vectors (typically Rademacher: ±1)
   """
   batch_size, dim = x_in.shape
   
   def velocity_field(x):
       return eval_model(vf_model, t, x)
   
   def compute_divergence_hutchinson_single(x_single, sample_key):
       """Compute Hutchinson trace estimate for a single sample point."""
       
       def jvp_fn(eps):
           """Compute Jacobian-vector product J @ eps."""
           _, jvp = jax.jvp(velocity_field, (x_single[None, :],), (eps[None, :],))
           return jvp[0]  # Remove batch dimension
       # Generate random vectors for trace estimation
       keys = jax.random.split(sample_key, num_samples)
       
       def single_estimate(eps_key):
           # Sample Rademacher random vector (±1 entries)
           eps = jax.random.rademacher(eps_key, (dim,), dtype=x_single.dtype)
           
           # Compute εᵀ J ε using JVP
           jvp_result = jvp_fn(eps)
           return jnp.dot(eps, jvp_result)
       
       # Average over multiple random samples
       estimates = jax.vmap(single_estimate)(keys)
       return jnp.mean(estimates)
   key = jax.random.PRNGKey(0)
   # Split keys for each batch element
   batch_keys = jax.random.split(key, batch_size)
   
   # Vectorize over batch dimension
   divergence = jax.vmap(compute_divergence_hutchinson_single)(x_in, batch_keys)
   
   return divergence

def divergence_vf(
    vf_model: nnx.Module,
    t: TimeArray,
    x_in: SampleArray
):
    """
    Compute the divergence 
    
    Args:
        vf_model: Velocity field neural network that takes (t, x_in)
        t: Time value (scalar)
        x_in: Sample positions, shape (batch_size, dim)
        
    Returns:
        Divergence values, shape (batch_size,)
    """
    def velocity_field(x):
        return eval_model(vf_model, t,x)
    
    # Compute divergence using JAX's efficient diagonal Jacobian extraction
    def compute_divergence_single(x_single):
        """Compute divergence for a single sample point."""
        jac_fn = jax.jacfwd(velocity_field)
        jacobian = jac_fn(x_single[None, :])  # Add batch dim, compute Jacobian
        return jnp.trace(jacobian[0].squeeze())  # Extract trace (divergence)

    # Vectorize over batch dimension
    divergence = jax.vmap(compute_divergence_single)(x_in)
    
    return divergence
def jacobian_vf(
    vf_model: nnx.Module,
    t: TimeArray,
    x_in: SampleArray
):
    """
    Compute the jacobian
    Args:
        vf_model: Velocity field neural network that takes (t, x_in)
        t: Time value (scalar)
        x_in: Sample positions, shape (batch_size, dim)
        
    Returns:
        Divergence values, shape (batch_size,)
    """
    def velocity_field(x):
        return eval_model(vf_model, t,x)
    
    # Compute divergence using JAX's efficient diagonal Jacobian extraction
    def compute_divergence_single(x_single):
        """Compute divergence for a single sample point."""
        jac_fn = jax.jacfwd(velocity_field)
        jacobian = jac_fn(x_single[None, :])  # Add batch dim, compute Jacobian
        return jacobian[0].squeeze()  # Remove singleton dimensions
    
    # Vectorize over batch dimension
    jacobian = jax.vmap(compute_divergence_single)(x_in)
    return jacobian