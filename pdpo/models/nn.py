"""
Neural network architectures for PDPO using JAX and nnx.

This module provides MLP implementations compatible with Neural ODEs
and the parametric pushforward framework.
"""

from typing import Callable, Optional, Union
import jax
import jax.numpy as jnp
from flax import nnx
import math as math

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


def timestep_embedding(timesteps: jnp.ndarray, dim: int, max_period: float = 10000.0) -> jnp.ndarray:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: a 1-D Array of N indices, one per batch element.
                   These may be fractional.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
        
    Returns:
        an [N x dim] Array of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    )
    args = timesteps[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding




class MLPTimeEmbedding(nnx.Module):
    """
    MLP with sophisticated sinusoidal time embedding.
    
    This implementation separates spatial and temporal processing, combining them
    in the hidden space before producing the final output. The time embedding
    uses sinusoidal positional encoding similar to transformers.
    
    Architecture:
    1. Sinusoidal time embedding -> MLP processing
    2. Spatial input -> MLP processing  
    3. Combine in hidden space (addition)
    4. Final MLP -> output
    """
    
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        layer_width: int,
        activation: Union[str, Callable] = 'relu',
        time_embed_dim: int = 128,
        step_scale: float = 1000.0,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize MLP with time embedding.
        
        Args:
            input_size: Dimension of spatial input features
            num_layers: Total number of layers for spatial processing (must be >= 2)
            layer_width: Number of neurons in hidden layers
            activation: Activation function name or callable
            time_embed_dim: Dimension of sinusoidal time embedding
            step_scale: Scaling factor for timesteps (typically 1000)
            rngs: Random number generators for initialization
        """
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2 (input + output)")
        
        self.input_size = input_size + 1 # Increase by 1 to include time
        self.output_size = input_size  # Vector field: R^d -> R^d
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.time_embed_dim = time_embed_dim
        self.step_scale = step_scale
        
        # Get activation function
        if isinstance(activation, str):
            self.activation_fn = get_activation_fn(activation)
        else:
            self.activation_fn = activation
        
        # Time processing: embedding -> linear layers
        self.t_layers = [
            nnx.Linear(time_embed_dim, layer_width, rngs=rngs),
            nnx.Linear(layer_width, layer_width, rngs=rngs)
        ]
        
        # Spatial processing: input -> hidden representation
        self.x_layers = []
        
        # Input layer (spatial)
        self.x_layers.append(
            nnx.Linear(input_size, layer_width, rngs=rngs)
        )
        
        # Hidden layers (spatial)
        for _ in range(num_layers - 2):
            self.x_layers.append(
                nnx.Linear(layer_width, layer_width, rngs=rngs)
            )
        
        # Output processing: combined hidden -> output
        self.out_layers = [
            nnx.Linear(layer_width, layer_width, rngs=rngs),
            nnx.Linear(layer_width, input_size, rngs=rngs)  # Output same dimension as input
        ]
    
    def __call__(self, xt: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the time-embedded network.
        
        Args:
            xt: Spatial input tensor of shape (batch_size, input_size + 1), time is the first dimension
            
        Returns:
            Output tensor of shape (batch_size, input_size)
        """
        #Extract time variable
        t = xt[:,0]
        #Extract space variable
        x = xt[:,1:]

        # # Ensure proper shapes
        # if t.ndim == 2 and t.shape[-1] == 1:
        #     t = t.squeeze(-1)  # Remove trailing dimension if present
        # elif t.ndim == 0:
        #     t = jnp.array([t])  # Convert scalar to array
        
        # if x.shape[-1] != self.input_size:
        #     raise ValueError(
        #         f"Expected spatial input dimension {self.input_size}, got {x.shape[-1]}"
        #     )
        
        # Ensure batch dimensions match
        batch_size = x.shape[0]
        # if t.shape[0] != batch_size:
        #     if t.shape[0] == 1:
        #         t = jnp.repeat(t, batch_size)
        #     else:
        #         raise ValueError(
        #             f"Batch size mismatch: t has {t.shape[0]} samples, x has {batch_size}"
        #         )
        
        # Process time: sinusoidal embedding -> MLP
        t_scaled = t * self.step_scale
        t_emb = timestep_embedding(t_scaled, self.time_embed_dim)
        
        # Time processing
        t_hidden = t_emb
        for i, layer in enumerate(self.t_layers):
            t_hidden = layer(t_hidden)
            if i < len(self.t_layers) - 1:  # No activation on final time layer
                t_hidden = self.activation_fn(t_hidden)
        
        # Process spatial input: x -> hidden representation
        x_hidden = x
        for layer in self.x_layers:
            x_hidden = layer(x_hidden)
            x_hidden = self.activation_fn(x_hidden)
        
        # Combine time and spatial information in hidden space
        combined_hidden = x_hidden + t_hidden
        
        # Output processing
        output = combined_hidden
        for i, layer in enumerate(self.out_layers):
            output = layer(output)
            if i < len(self.out_layers) - 1:  # Activation on all but final layer
                output = self.activation_fn(output)
        
        return output


def create_mlp_time_embedding(
    input_size: int,
    num_layers: int,
    layer_width: int,
    activation: Union[str, Callable] = 'relu',
    time_embed_dim: int = 128,
    step_scale: float = 1000.0,
    key: Optional[jax.Array] = None,
) -> MLPTimeEmbedding:
    """
    Create an MLP with time embedding.
    
    Args:
        input_size: Dimension of spatial input features
        num_layers: Total number of layers for spatial processing (must be >= 2)
        layer_width: Number of neurons in hidden layers
        activation: Activation function name or callable
        time_embed_dim: Dimension of sinusoidal time embedding
        step_scale: Scaling factor for timesteps (typically 1000)
        key: JAX random key for initialization (if None, creates new key)
        
    Returns:
        Initialized MLPTimeEmbedding model
        
    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> model = create_mlp_time_embedding(
        ...     input_size=2, 
        ...     num_layers=4, 
        ...     layer_width=128,
        ...     activation='relu',
        ...     key=key
        ... )
        >>> t = jnp.array([0.5])  # time
        >>> x = jnp.ones((1, 2))  # spatial input
        >>> output = model(t, x)  # shape: (1, 2)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    rngs = nnx.Rngs(key)
    
    return MLPTimeEmbedding(
        input_size=input_size,
        num_layers=num_layers,
        layer_width=layer_width,
        activation=activation,
        time_embed_dim=time_embed_dim,
        step_scale=step_scale,
        rngs=rngs,
    )


def count_parameters_time_embedding(model: MLPTimeEmbedding) -> int:
    """Count total number of parameters in the time-embedded model."""
    total = 0
    
    # Count time processing parameters
    for layer in model.t_layers:
        total += layer.kernel.size + layer.bias.size
    
    # Count spatial processing parameters
    for layer in model.x_layers:
        total += layer.kernel.size + layer.bias.size
    
    # Count output processing parameters
    for layer in model.out_layers:
        total += layer.kernel.size + layer.bias.size
    
    return total


def get_model_info_time_embedding(model: MLPTimeEmbedding) -> dict:
    """Get detailed information about the time-embedded model architecture."""
    return {
        'input_size': model.input_size,
        'output_size': model.output_size,
        'num_layers': model.num_layers,
        'layer_width': model.layer_width,
        'time_embed_dim': model.time_embed_dim,
        'step_scale': model.step_scale,
        'total_parameters': count_parameters_time_embedding(model),
        'activation': model.activation_fn.__name__ if hasattr(model.activation_fn, '__name__') else str(model.activation_fn),
        'architecture_type': 'MLPTimeEmbedding',
    }

