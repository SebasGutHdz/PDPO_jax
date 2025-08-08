from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Callable, NamedTuple 
from jaxtyping import Array, Float, PyTree
from pdpo.core.types import (TimeStepsArray)
from pdpo.ode.solvers import ODESolver, MidpointSolver
# =============================================================================
# Configuration and State Types
# =============================================================================

@dataclass
class SplineConfig:
    """Configuration for spline interpolation."""
    num_interior_points: int
    """Number of interior control points."""
    spline_type: str  # 'linear' or 'cubic'
    """Type of spline interpolation ('linear' or 'cubic')."""
    type_architecture: str 
    '''Type of architecture for spline interpolation (e.g., 'mlp', 'mlp_time_embedding').'''
    architecture: List[int]  
    """
    List of layer sizes for the neural network.
    For mlp: [input_dim, hidden_dim, num_layers, activation,time_varying]
    For mlp_time_embedding: [input_dim, hidden_dim, num_layers, activation]
    """
    data0: str  
    """Source distribution name"""
    data1: str  
    """Target distribution name"""
    device: str
    """Computation device ('cpu' or 'gpu')"""
    solver: ODESolver = MidpointSolver
    

@dataclass
class SplineState:
    """State container for spline parameters."""
    control_points: List[PyTree]  # Interior control points [θ_t1, ..., θ_tK]
    """Interior control points for spline interpolation."""
    boundary_params: Tuple[PyTree, PyTree]  # (θ_0, θ_1)
    """Boundary parameters for spline interpolation."""
    time_points: TimeStepsArray  # Time discretization
    """Time points for interpolation."""
    config: SplineConfig
    """Configuration for spline interpolation."""
    prior: Optional[str] = 'std_gaussian'  # Prior distribution type
    """Prior distribution type for the spline."""


@dataclass
class ProblemConfig:
    """Configuration for the optimization problem."""
    splinestate: SplineState
    ''' Configuration of spline curve. '''
    ke_modifier: Optional[List[Callable]] = None  
    ''' Kinetic energy modifiers '''
    potential: Optional[List[Callable]] = None
    ''' Potential energy terms '''
    entropy: int = 0
    ''' Entropy coefficient for the optimization '''
    fisher: int = 0
    ''' Fisher information coefficient for the optimization '''
    p: int = 2
    ''' Norm for kinetic energy ''' 
   

@dataclass
class OptimizationHistory:
    """Container for optimization history."""
    lagrangian: List[float]
    """Lagrangian values during optimization."""
    kinetic: List[float]
    """Kinetic energy values during optimization."""
    potential: List[float]
    """Potential energy values during optimization."""
    iterations: List[int]
    """Iteration numbers during optimization."""