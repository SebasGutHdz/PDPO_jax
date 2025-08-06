from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Callable
import jax 
import jax.numpy as jnp
import jax.random as jrn
from flax import nnx
from jaxtyping import Array,Float,PyTree

from pdpo.core.types import (
    ModelParams,
    ModelState,
    TimeArray,
    SampleArray,
    VelocityArray,
    ScoreArray,
    PRNGKeyArray
)
from pdpo.generative.optimization.objectives import FlowMatchingObjective,ConditionalFlowMatching,StochasticInterpolantsObjective
from pdpo.ode.solvers import ODESolver


class MatchingMethod(ABC):
    """
    Abstract base class for generative models used in PDPO boundary initialization.
    
    Subclasses: FlowMatching, ConditionalFlowMatching, StochasticInterpolant
    """
    
    def __init__(
        self,
        method_name: str,
        vf_model: nnx.Module,
        optimizer: nnx.Optimizer,
        ode_solver: ODESolver,
        scheduler: Optional[Callable] = None,
        reference_sampler: Optional[Callable] = None
    ):
        """
        Initialize the matching method.
        
        Args:
            method_name: Method identifier ("fm", "cfm", "si")
            vf_model: Velocity field neural network (nnx.Module)
            optimizer: JAX optimizer for training
            scheduler: Optional learning rate scheduler
        """
        assert method_name in ["fm", "cfm", "si"], f"Invalid method_name: {method_name}"
        
        self.method_name = method_name
        self.vf_model = vf_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ode_solver = ode_solver
         if reference_sampler is None:
            # Create a proper sampler function that takes key and shape
            self.reference_sampler = lambda key, shape: jrn.normal(key=key, shape=shape)
        else:
            self.reference_sampler = reference_sampler

        
    def __repr__(self) -> str:
        """String representation showing method and model architecture."""
        model_cls = type(self.vf_model).__name__
        
        if hasattr(self.vf_model, 'layers'):
            n_layers = len(self.vf_model.layers)
            model_repr = f"{model_cls} with {n_layers} layers"
        else:
            model_repr = model_cls

        return f"{self.__class__.__name__}(method='{self.method_name}', model={model_repr})"
    
    @abstractmethod
    def compute_loss(
        self,
        key: PRNGKeyArray,
        data_batch: SampleArray,
        reference_samples: Optional[SampleArray] = None,
    ) -> Tuple[Float[Array, ""], Dict[str, Any]]:
        """
        Compute training loss using objectives from generative/optimization/objectives.py
        
        Args:
            params: Model parameters
            key: JAX random key
            data_batch: Target samples (ρ₁)
            x1: Optional source samples (ρ₀). If None, use Gaussian
            model_state: Optional model state
            
        Returns:
            loss: Scalar loss value
            metrics: Dictionary of training metrics
            new_model_state: Updated model state
        """
        pass
    
    def eval_model(
        self,
        model: nnx.Module,
        t: TimeArray,
        x: SampleArray,
        batch_size: int
    )-> VelocityArray:
        """
        Evaluate the velocity field model with proper time conditioning.
        
        Args:
            model: The neural network model
            t: Time values, float or jnp.array shape (batch_size,) or (batch_size,1)
            x: Sample positions, shape (batch_size, dim) 
            batch_size: Batch size for verification
            
        Returns:
            Predicted velocities, shape (batch_size, dim)
        """
        if t.ndim ==0:  # element from jnp.array
            t_explanded jnp.full((batch_size, 1), t)
        elif t.ndim == 1: # Batch of times with format (bs,)
            t_expanded = t.reshape(-1, 1)
        elif  t.ndim == 2 : # Batch of times with correct format
            t_expanded = t
        else:
            raise ValueError("t does not have the right shape, valid float of jnp with shapes (bs,) and (bs,1)")
        model_input = jnp.concatenate([t_expanded,x], axis=-1)
        v_pred = model(model_input)
        return v_pred
    
    def sample_trajectory(
        self,
        x0: SampleArray,
        n_steps: int = 10,
    ) -> SampleArray:
        """
        Generate sample trajectories by integrating the velocity field.
        
        Uses fixed step size ODE integration from /ode/solvers.py
        
        Args:
            x0: Batch of initial conditions (bs,d)
            n_steps: Number of integration steps (default: 10)            
        Returns:
            samples: Final samples at t=1, shape (n_samples, dim)
            new_model_state: Updated model state
        """
        t = jnp.linspace(0,1,n_steps)
        x = x0
        dt = 1/(n_steps-1)

        for i in range(n_steps - 1):
            x = self.ode_solver.step(self.vf_model, t_span[i], x, dt)
        return x
    
    def training_step(
        self,
        key: PRNGKeyArray,
        data_batch: SampleArray,
        reference_samples: Optional[SampleArray] = None
    ) -> Tuple[Float[Array, ""], Dict[str, Any], ModelParams, ModelState]:
        """
        Execute one training step.
        
        Args:
            key: JAX random key
            data_batch: Batch of target data
            reference_samples: Optional source data for ρ₀ → ρ₁ training
            model_state: Optional model state
            
        Returns:
            loss: Scalar loss value
            metrics: Dictionary of training metrics
            updated_params: Updated model parameters
            updated_state: Updated model state
        """
        batch_size,dim = data_batch.shape      
        if reference_samples is None:
            key_refernce,subkey = jnr.split(key)
            reference_samples = self.reference_sampler(key_reference,batch_size)
        def loss_fn(model):
            return self.compute_loss(model,key,data_batch, reference_samples)
        # Compute loss and gradients
        (loss, metrics), grads = nnx.grad(loss_fn)(self.vf_model)
        
        # Update parameters
        self.optimizer.update(self.vf_model,grads)
        
        # Update learning rate if scheduler provided
        if self.scheduler is not None:
            new_lr = self.scheduler(self.optimizer.state.step)
            metrics['learning_rate'] = new_lr
            
        return loss, metrics
    
    def train(
        self,
        key: PRNGKeyArray,
        target_data: SampleArray,
        num_epochs: int,
        batch_size: int,
        source_data: Optional[SampleArray] = None,
        eval_frequency: int = 100
    ) -> Tuple[ModelParams, ModelState, Dict[str, Any]]:
        """
        Train the generative model.
        
        Args:
            key: JAX random key
            target_data: Target distribution samples (ρ₁)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            source_data: Optional source distribution samples (ρ₀)
            model_state: Initial model state
            eval_frequency: How often to log training progress
            
        Returns:
            final_params: Trained model parameters
            final_state: Final model state
            training_history: Dictionary of training metrics over time
        """
        history = {'train_loss': [], 'epochs': []}
        current_params = params
        current_state = model_state
        
        num_batches = len(target_data) // batch_size
        
        for epoch in range(num_epochs):
            epoch_key = jax.random.fold_in(key, epoch)
            epoch_loss = 0.0
            
            # Shuffle data
            perm = jax.random.permutation(epoch_key, len(target_data))
            shuffled_target = target_data[perm]
            shuffled_source = source_data[perm] if source_data is not None else None
            
            for batch_idx in range(num_batches):
                batch_key = jax.random.fold_in(epoch_key, batch_idx)
                
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(target_data))
                target_batch = shuffled_target[start_idx:end_idx]
                source_batch = shuffled_source[start_idx:end_idx] if shuffled_source is not None else None
                
                # Training step
                loss, metrics, current_params, current_state = self.training_step(
                    current_params, batch_key, target_batch, source_batch, current_state
                )
                
                epoch_loss += loss
                
            # Record training metrics
            avg_loss = epoch_loss / num_batches
            
            if epoch % eval_frequency == 0:
                history['train_loss'].append(float(avg_loss))
                history['epochs'].append(epoch)
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
                
        return current_params, current_state, history
    





