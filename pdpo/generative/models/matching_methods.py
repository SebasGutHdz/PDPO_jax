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
from pdpo.generative.optimization import objectives
from pdpo.ode import solvers
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
        scheduler: Optional[Callable] = None
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
        params: ModelParams,
        key: PRNGKeyArray,
        data_batch: SampleArray,
        x1: Optional[SampleArray] = None,
        model_state: Optional[ModelState] = None
    ) -> Tuple[Float[Array, ""], Dict[str, Any], ModelState]:
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
    
    
    def sample_trajectory(
        self,
        x0: SampleArray,
        n_steps: int = 10,
        model_state: Optional[ModelState] = None
    ) -> Tuple[SampleArray, ModelState]:
        """
        Generate sample trajectories by integrating the velocity field.
        
        Uses fixed step size ODE integration from /ode/solvers.py
        
        Args:
            params: Model parameters
            key: JAX random key
            n_samples: Number of samples to generate
            n_steps: Number of integration steps (default: 10)
            x0: Optional initial samples. If None, sample from reference distribution
            model_state: Optional model state
            
        Returns:
            samples: Final samples at t=1, shape (n_samples, dim)
            new_model_state: Updated model state
        """
        dt = 1/n_steps
        t = jnp.linspace(0,1,n_steps)
        x1 = self.ode_solver(self.vf_model,t[0],x0,dt)
        for i in range(1,n_steps):
            x1 = self.ode_solver(self.vf_model,t[i],x1,dt)
        return x1
    
    def training_step(
        self,
        params: ModelParams,
        key: PRNGKeyArray,
        data_batch: SampleArray,
        x1: Optional[SampleArray] = None,
        model_state: Optional[ModelState] = None
    ) -> Tuple[Float[Array, ""], Dict[str, Any], ModelParams, ModelState]:
        """
        Execute one training step.
        
        Args:
            params: Current model parameters
            key: JAX random key
            data_batch: Batch of target data
            x1: Optional source data for ρ₀ → ρ₁ training
            model_state: Optional model state
            
        Returns:
            loss: Scalar loss value
            metrics: Dictionary of training metrics
            updated_params: Updated model parameters
            updated_state: Updated model state
        """
        def loss_fn(params):
            return self.compute_loss(params, key, data_batch, x1, model_state)
        
        # Compute loss and gradients
        (loss, metrics, new_model_state), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, self.optimizer.state)
        new_params = nnx.apply_updates(params, updates)
        
        # Update learning rate if scheduler provided
        if self.scheduler is not None:
            new_lr = self.scheduler(self.optimizer.state.step)
            metrics['learning_rate'] = new_lr
            
        return loss, metrics, new_params, new_model_state
    
    def train(
        self,
        params: ModelParams,
        key: PRNGKeyArray,
        target_data: SampleArray,
        num_epochs: int,
        batch_size: int,
        source_data: Optional[SampleArray] = None,
        model_state: Optional[ModelState] = None,
        eval_frequency: int = 100
    ) -> Tuple[ModelParams, ModelState, Dict[str, Any]]:
        """
        Train the generative model.
        
        Args:
            params: Initial model parameters
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
        scheduler: Optional[Callable] = None
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
            scheduler=scheduler
        )
    
    def compute_loss(
        self,
        params: ModelParams,
        key: PRNGKeyArray,
        data_batch: SampleArray,
        x1: Optional[SampleArray] = None,
        model_state: Optional[ModelState] = None
    ) -> Tuple[Float[Array, ""], Dict[str, Any], ModelState]:
        """
        Compute Flow Matching loss using existing objective.
        
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
        # Use existing flow_matching_loss from objectives.py
        # Adapt to the actual API in your objectives module
        loss, metrics, new_model_state = objectives.flow_matching_loss(
            model=self.vf_model,
            params=params,
            key=key,
            x1=data_batch,  # Target samples
            x0=x1,  # Source samples (None for Gaussian)
            model_state=model_state
        )
        
        return loss, metrics, new_model_state
    
    def sample_trajectory(
        self,
        params: ModelParams,
        key: PRNGKeyArray,
        n_samples: int,
        n_steps: int = 10,
        x0: Optional[SampleArray] = None,
        model_state: Optional[ModelState] = None
    ) -> Tuple[SampleArray, ModelState]:
        """
        Generate samples by integrating the velocity field ODE.
        
        Integrates dx/dt = v_θ(t, x) from t=0 to t=1 using fixed step size.
        
        Args:
            params: Model parameters
            key: JAX random key
            n_samples: Number of samples to generate
            n_steps: Number of integration steps
            x0: Optional initial samples. If None, sample from N(0,I)
            model_state: Optional model state
            
        Returns:
            samples: Final samples at t=1, shape (n_samples, dim)
            new_model_state: Updated model state
        """
        # Initialize samples
        if x0 is not None:
            x = x0.copy()
            dim = x0.shape[-1]
        else:
            # Default to 2D for now - in practice, infer from model or pass as parameter
            dim = 2
            x = jax.random.normal(key, (n_samples, dim))
        
        # Time integration parameters
        dt = 1.0 / n_steps
        current_model_state = model_state
        
        # Integrate using fixed step size Euler method
        for step in range(n_steps):
            t = step * dt
            
            # Compute velocity: dx/dt = v_θ(t, x)
            velocity, current_model_state = self._velocity_field(
                params, t, x, current_model_state
            )
            
            # Euler step: x_{t+dt} = x_t + dt * v_θ(t, x_t)
            x = x + dt * velocity
        
        return x, current_model_state
    
    def _velocity_field(
        self,
        params: ModelParams,
        t: float,
        x: SampleArray,
        model_state: Optional[ModelState] = None
    ) -> Tuple[SampleArray, ModelState]:
        """
        Compute velocity field v_θ(t, x) using the neural network.
        
        Args:
            params: Model parameters
            t: Time value (scalar)
            x: Position samples, shape (batch_size, dim)
            model_state: Optional model state
            
        Returns:
            velocity: Velocity field v_θ(t, x), shape (batch_size, dim)
            new_model_state: Updated model state
        """
        batch_size = x.shape[0]
        
        # Time conditioning - broadcast time to match batch dimension
        t_expanded = jnp.full((batch_size, 1), t)
        
        # Concatenate position and time: input = [x, t]
        model_input = jnp.concatenate([x, t_expanded], axis=-1)
        
        # Forward pass through the velocity field network
        velocity = self.vf_model(model_input)

        return velocity,model_state
