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
from flax.nnx import sigmoid
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
        params: ModelParams,
        key: PRNGKeyArray,
        data_batch: SampleArray,
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
            model_fn:Callable,
            sigma: float = 0.0,
            time_sampling: str = "uniform"
    ):
        '''
        Initialize FM        
        Args:
            model_fn: Velocity field model function
            sigma: Noise level for conditional FM (0.0 for deterministic)
            time_sampling: Time sampling strategy ("uniform", "logit_normal")
        '''

        self.model_fn = model_fn
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
            key: PRNGKeyArray,
            data_batch: SampleArray,
            prior_samples: Optional[SampleArray] = None
    )-> Tuple[Float[Array,""],Dict[str,Any]]:
        '''Compute FM loss'''

        batch_size,dim = data_batch.shape

        # Sample from prior if not provided
        if prior_samples == None:
            
            key_prior,key = jrn.split(key)
            prior_samples = jrn.normal(key = key_prior, shape=(batch_size,dim))

        # Sample time points
        key_time,key = jrn.spit(key)
        t = self.sample_time(key= key_time,batch_size=batch_size)

        # Noise for CFM
        noise = None
        if self.sigma>0:
            key_noise,key = jrn.split(key)
            noise = jrn.normal(key= key_noise,batch_size = (batch_size,dim))
        
        # Compute interpolation and target velocity
        x_t,u_t = self.interpolate(t = t, x0 = prior_samples, x1 = data_batch,noise=noise)

        # Predict velocity
        v_pred = self.model_fn(t,x_t)

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
    





