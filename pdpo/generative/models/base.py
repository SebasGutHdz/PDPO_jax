from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Callable
import jax 
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array,Float,PyTree

from pdpo_jax.core.types import (
    ModelParams,
    ModelState,
    TimeArray,
    SampleArray,
    VelocityArray,
    ScoreArray,
    PRNGKeyArray
)

class GenerativeModel(ABC):
    """
    Abstract base class for generative models.    
    """

    @abstractmethod
    def __init__(
        self,
        dim: int,
        model_fn: Callable[...,nnx.Module],
        model_kwargs: Dict[str,Any],
        **kwars
    ):
        """Initialize the generative model.
        
        Args:
            dim: Dimension of the data space
            model_fn: Constructor for the neural network (e.g., MLP)
            model_kwargs: Arguments for model_fn
            **kwargs: Additional model-specific arguments
        """
        pass

    @abstractmethod
    def compute_velocity(
        self,
        params: ModelParams,
        t: TimeArray,
        x: SampleArray,
        model_state: Optional[ModelState] = None,
    )-> Tuple[VelocityArray,ModelState]:
        """Compute the velocity field at given time and position.
        
        Args:
            params: Model parameters
            t: Time values in [0, 1]
            x: Positions/samples
            model_state: Optional model state (for models with batch norm, etc.)
            
        Returns:
            velocity: Velocity field v(t, x)
            new_model_state: Updated model state
        """
        pass

    @abstractmethod
    def sample_trajectory(
        self,
        params: ModelParams,
        key: PRNGKeyArray,
        n_samples: int,
        n_steps: int = 10,
        model_state: Optional[ModelState] = None
    )-> Tuple[SampleArray,ModelState]:
        """Sample from the learned distribution.
        
        Args:
            params: Model parameters
            key: JAX random key
            n_samples: Number of samples to generate
            n_steps: Number of integration steps
            model_state: Optional model state
            
        Returns:
            samples: Generated samples at t=1
            new_model_state: Updated model state
        """
        pass

    @abstractmethod
    def training_step(
        self,
        params: ModelParams,
        key: PRNGKeyArray,
        data_batch: SampleArray,
        model_state: Optional[ModelState] = None
    ) -> Tuple[Float[Array, ""], Dict[str, Any], ModelState]:
        """Compute loss for a training batch.
        
        Args:
            params: Model parameters
            key: JAX random key
            data_batch: Batch of training data
            model_state: Optional model state
            
        Returns:
            loss: Scalar loss value
            metrics: Dictionary of additional metrics
            new_model_state: Updated model state
        """
        pass
    @property
    @abstractmethod
    def requires_score(self) -> bool:
        """Whether this model requires score function computation."""
        pass
    
    def initialize_parameters(
        self,
        key: PRNGKeyArray,
        dummy_input: SampleArray
    ) -> Tuple[ModelParams, ModelState]:
        """Initialize model parameters given a dummy input.
        
        Args:
            key: JAX random key
            dummy_input: Example input for shape inference
            
        Returns:
            params: Initialized parameters
            model_state: Initial model state
        """
        # This can have a default implementation
        raise NotImplementedError("Subclasses should implement parameter initialization")