"""
Objective functions for generative models training in PDPO.
"""
from pathlib import Path
project_root = Path(__file__).parent.absolute()
import sys
sys.path.append(str(project_root))


from abc import ABC,abstractmethod
from typing import Tuple,Dict,Any,Optional,Callable
import jax
import jax.numpy as jnp
import jax.random as jrn
from jax.nn import sigmoid
from flax import nnx
from jaxtyping  import Array,Float, PRNGKeyArray


from pdpo.core.types import (
    ModelParams,
    SampleArray,
    TimeArray,
    VelocityArray,
    ScoreArray
)


class ObjectiveFunction(ABC):
    """
    Abstract base class for all objective functions.
    """

    @abstractmethod
    def compute_loss(
        self,
        model: nnx.Module,
        eval_nf: Callable,
        key: PRNGKeyArray,
        data_batch: SampleArray,
        refrence_samples: Optional[SampleArray] = None,
        **kwargs
    )-> Tuple[Float[Array,""],Dict[str,Any]]:
        
        """Compute loss and metrics for a batch of data.
        
        Args:
            params: Model parameters
            key: JAX random key
            data_batch: Batch of target samples
            **kwargs: Additional objective-specific arguments
            
        Returns:
            loss: Scalar loss value
            metrics: Dictionary of additional metrics
        """
        pass



class FlowMatchingObjective(ObjectiveFunction):
    '''Flow Matching for learing transport maps
    
    Implements the standard FM loss: L_FM = E[||v_{theta}(t,x_t) - u_t||²]
    where x_t = (1-t)x_0 + tx_1 + sig_t ε and u_t = x_1 - x_0 + sig_t' ε
    '''

    def __init__(
            self,
            sigma: float = 0.0,
            time_sampling: str = "uniform"
    ):
        '''
        Initialize FM        
        Args:
            sigma: Noise level for conditional FM (0.0 for deterministic)
            time_sampling: Time sampling strategy ("uniform", "logit_normal")
        '''
        self.sigma = sigma
        self.time_sampling = time_sampling

    def sample_time(self,key:PRNGKeyArray,batch_size: int)-> TimeArray:

        '''Sample time points for training'''

        if self.time_sampling == "uniform":
            return jrn.uniform(key,(batch_size,), minval=1e-6,maxval=1.0)
        elif self.time_sampling == "logit_normal":
            normal_samples = jrn.normal(key=key, shape=(batch_size,))
            return sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampling: {self.time_sampling}")
        
    def interpolate(
            self,
            t: TimeArray,
            x0:SampleArray,
            x1:SampleArray,
            noise:Optional[SampleArray] = None
    )-> Tuple[SampleArray,VelocityArray]:
        """Compute interpolated samples and target velocities.
        
        Args:
            t: Time points [batch_size]
            x0: Source samples [batch_size, dim]  
            x1: Target samples [batch_size, dim]
            noise: Optional noise samples [batch_size, dim] (This is passed for jax key control management)
            
        Returns:
            x_t: Interpolated samples [batch_size, dim]
            u_t: Target velocity [batch_size, dim]
        """

        t = t.reshape(-1,1) # (bs,) -> (bs,1)

        # Linear interpolation 
        x_t = (1-t)*x0+t*x1
        u_t = x1-x0

        # Add noise for cfm
        if self.sigma>0:
            if noise is None:
                raise ValueError("Noise required for CFM")
            sigma_t = self.sigma*jnp.sqrt(t*(1-t))
            x_t = x_t + sigma_t*noise
        
        return x_t,u_t
    
    def compute_loss(
            self,
            model: nnx.Module,
            eval_model: Callable,
            key: PRNGKeyArray,
            data_batch: SampleArray,
            reference_samples: Optional[SampleArray] = None
    )-> Tuple[Float[Array,""],Dict[str,Any]]:
        '''Compute FM loss'''

        batch_size,dim = data_batch.shape           

        # Sample time points
        key_time,key = jrn.split(key)
        t = self.sample_time(key= key_time,batch_size=batch_size)

        # Noise for CFM
        noise = None
        if self.sigma>0:
            key_noise,key = jrn.split(key)
            noise = jrn.normal(key= key_noise,shape = (batch_size,dim))
        
        # Compute interpolation and target velocity
        x_t,u_t = self.interpolate(t = t, x0 = reference_samples, x1 = data_batch,noise=noise)

        # Predict velocity
        v_pred = eval_model(model,t,x_t)

        l2_error = jnp.linalg.norm(v_pred-u_t,axis = 1)**2 # Norm in the spatial dimension
        # Loss fn 
        loss = jnp.mean(l2_error)

        metrics = {
            "mse_loss": loss,
            "velocity_norm": jnp.mean(jnp.linalg.norm(v_pred, axis=1)),
            "target_velocity_norm": jnp.mean(jnp.linalg.norm(u_t, axis=1)),
        }
        
        return loss, metrics
    
class ConditionalFlowMatchingObjective(FlowMatchingObjective):
    """Conditional Flow Matching with learned conditional distributions."""
    
    def __init__(
        self,
        model_fn: Callable,
        sigma: float = 0.1,
        time_sampling: str = "uniform",
        interpolation_type: str = "linear"
    ):
        super().__init__(model_fn, sigma, time_sampling)
        self.interpolation_type = interpolation_type
    
    def interpolate(
        self,
        t: TimeArray,
        x0: SampleArray,
        x1: SampleArray,
        noise: Optional[SampleArray] = None
    ) -> Tuple[SampleArray, VelocityArray]:
        """Enhanced interpolation with multiple schemes."""
        t = t.reshape(-1, 1)
        
        if self.interpolation_type == "linear":
            return super().interpolate(t.squeeze(), x0, x1, noise)
        elif self.interpolation_type == "trigonometric":
            # Smoother interpolation using cosine
            alpha_t = 0.5 * (1 - jnp.cos(jnp.pi * t))
            x_t = (1 - alpha_t) * x0 + alpha_t * x1
            u_t = 0.5 * jnp.pi * jnp.sin(jnp.pi * t) * (x1 - x0)
        else:
            raise ValueError(f"Unknown interpolation type: {self.interpolation_type}")
        
        if self.sigma > 0 and noise is not None:
            sigma_t = self.sigma * jnp.sqrt(t * (1 - t))
            x_t = x_t + sigma_t * noise
            
        return x_t, u_t
    
class StochasticInterpolantsObjective(ObjectiveFunction):
    """Stochastic Interpolants objective for generalized transport."""
    
    def __init__(
        self,
        model_fn: Callable,
        alpha_fn: Callable[[TimeArray], TimeArray] = lambda t: 1 - t,
        beta_fn: Callable[[TimeArray], TimeArray] = lambda t: t,
        gamma_fn: Callable[[TimeArray], TimeArray] = lambda t: jnp.sqrt(t * (1 - t))
    ):
        """Initialize SI objective.
        
        Args:
            model_fn: Model function
            alpha_fn: Coefficient for x_0 in interpolant
            beta_fn: Coefficient for x_1 in interpolant  
            gamma_fn: Coefficient for noise in interpolant
        """
        self.model_fn = model_fn
        self.alpha_fn = alpha_fn
        self.beta_fn = beta_fn
        self.gamma_fn = gamma_fn
    
    def compute_interpolant(
        self,
        t: TimeArray,
        x0: SampleArray,
        x1: SampleArray,
        noise: SampleArray
    ) -> Tuple[SampleArray, VelocityArray]:
        """Compute stochastic interpolant and target velocity."""
        t = t.reshape(-1, 1)
        
        alpha_t = self.alpha_fn(t)
        beta_t = self.beta_fn(t)
        gamma_t = self.gamma_fn(t)
        
        # Interpolant: I_t = α(t)x_0 + β(t)x_1 + γ(t)ε
        x_t = alpha_t * x0 + beta_t * x1 + gamma_t * noise
        
        # Target velocity (time derivative)
        # For simple case: u_t = α'(t)x_0 + β'(t)x_1 + γ'(t)ε
        # Using finite differences for derivatives (can be replaced with exact)
        dt = 1e-4
        alpha_prime = (self.alpha_fn(t + dt) - self.alpha_fn(t - dt)) / (2 * dt)
        beta_prime = (self.beta_fn(t + dt) - self.beta_fn(t - dt)) / (2 * dt)
        gamma_prime = (self.gamma_fn(t + dt) - self.gamma_fn(t - dt)) / (2 * dt)
        
        u_t = alpha_prime * x0 + beta_prime * x1 + gamma_prime * noise
        
        return x_t, u_t
    
    def compute_loss(
        self,
        key: PRNGKeyArray,
        data_batch: SampleArray,
        reference_samples: Optional[SampleArray] = None,
        **kwargs
    ) -> Tuple[Float[Array, ""], Dict[str, Any]]:
        """Compute SI loss."""
        batch_size, dim = data_batch.shape
        
        # Sample components
        if reference_samples is None:
            key_prior, key = jax.random.split(key)
            reference_samples = jax.random.normal(key_prior, (batch_size, dim))
        
        key_time, key_noise, key = jax.random.split(key, 3)
        t = jax.random.uniform(key_time, (batch_size,), minval=1e-4, maxval=1.0)
        noise = jax.random.normal(key_noise, (batch_size, dim))
        
        # Compute interpolant and target
        x_t, u_t = self.compute_interpolant(t, reference_samples, data_batch, noise)
        
        # Predict velocity
        v_pred = self.model_fn(params, t, x_t)

        l2_error = jnp.linalg.norm(v_pred-u_t,axis = -1)**2
        
        # Compute loss
        loss = jnp.mean(l2_error)
        
        metrics = {
            "si_loss": loss,
            "interpolant_norm": jnp.mean(jnp.linalg.norm(x_t, axis=1)),
        }
        
        return loss, metrics
    



# Utility functions for common interpolation schemes
def linear_interpolation_coefficients(t: TimeArray) -> Tuple[TimeArray, TimeArray]:
    """Standard linear interpolation coefficients."""
    return 1 - t, t


def cosine_interpolation_coefficients(t: TimeArray) -> Tuple[TimeArray, TimeArray]:
    """Smooth cosine interpolation coefficients."""
    alpha = 0.5 * (1 - jnp.cos(jnp.pi * t))
    return 1 - alpha, alpha


def polynomial_interpolation_coefficients(
    t: TimeArray, 
    power: float = 2.0
) -> Tuple[TimeArray, TimeArray]:
    """Polynomial interpolation with adjustable power."""
    beta = t ** power
    return 1 - beta, beta

