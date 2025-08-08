from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable
import jax.numpy as jnp
import jax.random as jrn
from jax import lax
from jaxtyping import  PyTree


from pdpo.spline.interpolation import linear_interpolation_states
from pdpo.core.types import (
    TimeStepsArray
)
from pdpo.spline.interpolation import linear_interpolation_states
from pdpo.spline.types_interpolation import SplineState, SplineConfig



def Assemble_spline(
    theta0: PyTree,
    theta1: PyTree,
    type_arch: str,
    arch: List[int],
    data0: str = None,
    data1: str = None,
    number_of_knots: int = 3,
    spline_type: str = 'linear',
    device: str = 'cpu',
    prior_dist: Optional[str] = 'gaussian',
    p: int = 2
) -> Tuple[SplineState, TimeStepsArray]:
    """
    Creates a spline interpolation between two sets of neural network parameters.
    
    Args:
        theta0: Initial parameter vector
        theta1: Final parameter vector
        type_arch: Type of architecture (e.g., 'mlp', 'mlp_time_embedding')
        arch: Network architecture [input_dim, hidden_dim, num_layers, activation]
        data0: Initial distribution identifier
        data1: Final distribution identifier
        number_of_knots: Number of interior interpolation points
        spline_type: Interpolation type ('linear' or 'cubic')
        device: Computation device
        prior_dist: Base distribution parameters
        p: Norm used for kinetic energy
        
    Returns:
        spline_state: Initialized SplineState
        t: Time points for interpolation
    """
    total_knots = number_of_knots + 2  # Include boundaries
    t = jnp.linspace(0, 1, total_knots)
    
    # Initialize interior control points using linear interpolation
    interior_times = t[1:-1]  # Exclude boundaries
    interior_points = linear_interpolation_states(theta0, theta1, interior_times)
    
    # Create configuration
    config = SplineConfig(
        num_interior_points=number_of_knots,
        type_architecture=type_arch,
        spline_type=spline_type,
        architecture=arch,
        data0=data0,
        data1=data1,
        device=device
    )

    # Create spline state
    spline_state = SplineState(
        control_points=interior_points,
        boundary_params=(theta0, theta1),
        time_points=t,
        config=config,
        prior=prior_dist
    )

    return spline_state, t