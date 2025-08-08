


from typing import Callable, Optional, Union
import jax
import jax.numpy as jnp
from flax import nnx
import math as math
from pdpo.models.nn import MLP, MLPTimeEmbedding



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

