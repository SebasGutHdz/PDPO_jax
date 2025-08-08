





from typing import Callable, Optional, Union
import jax
import jax.numpy as jnp
from flax import nnx
import math as math
from pdpo.models.nn import MLP, MLPTimeEmbedding



def create_model(
        type: str = 'mlp',
        args_arch: Optional[list] = None,
) -> Union[MLP, MLPTimeEmbedding]:
    """
    Create a model based on the specified type and architecture parameters.
    
    Args:
        type: Type of model to create ('mlp' or 'mlp_time_embedding')
        args_arch: List containing architecture parameters
            - For 'mlp': [input_size, num_layers, layer_width, activation, time_varying]
            - For 'mlp_time_embedding': [input_size, num_layers, layer_width, activation, time_embed_dim, step_scale]
    Returns:
        Initialized model instance
    """
    
    if type == 'mlp':
        mlp = create_mlp(
            input_size=args_arch[0],
            num_layers=args_arch[1],
            layer_width=args_arch[2],
            activation=args_arch[3],
            time_varying=args_arch[4],
            key=args_arch[5] 
        )
        return mlp
    elif type == 'mlp_time_embedding':
        mlp_time_emb = create_mlp_time_embedding(
            input_size=args_arch[0],
            num_layers=args_arch[1],
            layer_width=args_arch[2],
            activation=args_arch[3],
            key=args_arch[4],
            time_embed_dim=args_arch[5] if len(args_arch) > 5 else 128,
            step_scale=args_arch[6] if len(args_arch) > 6 else 1000.0
        )
        return mlp_time_emb
    else:
        raise ValueError(f"Unknown model type: {type}")


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


def create_mlp_time_embedding(
    input_size: int,
    num_layers: int,
    layer_width: int,
    activation: Union[str, Callable] = 'relu',
    key: Optional[jax.Array] = None,
    time_embed_dim: int = 128,
    step_scale: float = 1000.0,
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

