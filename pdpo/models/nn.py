"""
Neural network architectures for PDPO using JAX and nnx.

This module provides MLP implementations compatible with Neural ODEs
and the parametric pushforward framework.
"""

from typing import Callable, Optional, Union
import jax
import jax.numpy as jnp
from flax import nnx


def get_activation_fn(name: str) -> Callable:
    """Get activation function by name."""
    activations = {
        'softplus': nnx.softplus,
        'relu': nnx.relu,
        'tanh': nnx.tanh,
        'sin': lambda x: jnp.sin(x),
        'sigmoid': nnx.sigmoid,
        'swish': nnx.swish,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name]


class MLP(nnx.Module):
    """
    Multi-layer perceptron with optional time-varying input.
    
    Compatible with Neural ODE integration and parametric optimization.
    """
    
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        layer_width: int,
        activation: Union[str, Callable] = 'softplus',
        time_varying: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize MLP.
        
        Args:
            input_size: Dimension of input features
            output_size: Dimension of output features  
            num_layers: Total number of layers (must be >= 2)
            layer_width: Number of neurons in hidden layers
            activation: Activation function name or callable
            time_varying: If True, adds 1 to input_size for time dimension
            rngs: Random number generators for initialization
        """
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2 (input + output)")
        
        self.time_varying = time_varying
        output_size = input_size
        self.input_size = input_size + (1 if time_varying else 0)
        self.output_size = input_size # Mapping from R^d->R^d
        self.num_layers = num_layers
        self.layer_width = layer_width

        
        # Get activation function
        if isinstance(activation, str):
            self.activation_fn = get_activation_fn(activation)
        else:
            self.activation_fn = activation
        
        # Build layers
        self.layers = []
        
        # Input layer
        self.layers.append(
            nnx.Linear(self.input_size, layer_width, rngs=rngs)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                nnx.Linear(layer_width, layer_width, rngs=rngs)
            )
        
        # Output layer
        self.layers.append(
            nnx.Linear(layer_width, output_size, rngs=rngs)
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (..., input_size) or (..., input_size + 1) if time_varying
            
        Returns:
            Output tensor of shape (..., output_size)
        """
        # Validate input dimensions
        expected_dim = self.input_size
        if x.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected input dimension {expected_dim}, got {x.shape[-1]}. "
                f"time_varying={self.time_varying}"
            )
        
        # Forward pass
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation_fn(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x


def create_mlp(
    input_size: int,
    num_layers: int,
    layer_width: int,
    activation: Union[str, Callable] = 'softplus',
    time_varying: bool = True,
    key: Optional[jax.Array] = None,
) -> MLP:
    """
    Create an MLP with specified architecture.
    
    Args:
        input_size: Dimension of input features
        output_size: Dimension of output features
        num_layers: Total number of layers (must be >= 2)
        layer_width: Number of neurons in hidden layers
        activation: Activation function name or callable
        time_varying: If True, adds time as additional input dimension
        key: JAX random key for initialization (if None, creates new key)
        
    Returns:
        Initialized MLP model
        
    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> model = create_mlp(2, 2, 4, 128, 'softplus', True, key)
        >>> # For time-varying case, input should be [batch, features + 1]
        >>> x = jnp.ones((100, 3))  # batch_size=100, features=2, time=1
        >>> output = model(x)  # shape: (100, 2)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    rngs = nnx.Rngs(key)
    
    return MLP(
        input_size=input_size,
        num_layers=num_layers,
        layer_width=layer_width,
        activation=activation,
        time_varying=time_varying,
        rngs=rngs,
    )


def count_parameters(model: MLP) -> int:
    """Count total number of parameters in the model."""
    total = 0
    for layer in model.layers:
        # Each Linear layer has weight and bias
        weight_params = layer.kernel.size
        bias_params = layer.bias.size
        total += weight_params + bias_params
    return total


def get_model_info(model: MLP) -> dict:
    """Get detailed information about the model architecture."""
    return {
        'input_size': model.input_size,
        'output_size': model.output_size,
        'num_layers': model.num_layers,
        'layer_width': model.layer_width,
        'time_varying': model.time_varying,
        'total_parameters': count_parameters(model),
        'activation': model.activation_fn.__name__ if hasattr(model.activation_fn, '__name__') else str(model.activation_fn),
    }