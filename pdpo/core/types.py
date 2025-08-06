"""
Core type definitions for PDPO JAX implementation.

This module provides type hints, NamedTuples, and custom types used throughout
the PDPO codebase for improved type safety and code clarity.
"""

from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Protocol, NamedTuple
)
from typing_extensions import TypeAlias
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Bool, PyTree
from flax import nnx
from dataclasses import dataclass
# =============================================================================
# JAX and Random Key Types
# =============================================================================

PRNGKeyArray: TypeAlias = jax.Array
"""JAX PRNG key for random number generation."""

PyTreeParams: TypeAlias = PyTree
"""PyTree structure containing model parameters."""


# =============================================================================
# Basic Array Types
# =============================================================================

# Sample and trajectory arrays
SampleArray: TypeAlias = Float[Array, "batch_size dim"]
"""Array of sample points in R^d."""

TrajectoryArray: TypeAlias = Float[Array, "batch_size time_steps dim"] 
"""Array of sample trajectories over time."""

# Time-related arrays
TimeArray: TypeAlias = Float[Array, "batch_size"]
"""Array of time points in [0,1]."""

TimeStepsArray: TypeAlias = Float[Array, "time_steps"]
"""Array of discrete time steps for integration."""

# Velocity and dynamics arrays
VelocityArray: TypeAlias = Float[Array, "batch_size dim"]
"""Array of velocity vectors."""

VelocityFieldArray: TypeAlias = Float[Array, "batch_size time_steps dim"]
"""Velocity field over trajectories."""

# Score and density arrays
ScoreArray: TypeAlias = Float[Array, "batch_size dim"]
"""Array of score function values (∇ log ρ)."""

DensityArray: TypeAlias = Float[Array, "batch_size"]
"""Array of probability density values."""

LogDensityArray: TypeAlias = Float[Array, "batch_size"]
"""Array of log probability density values."""

# Action and energy arrays
ActionArray: TypeAlias = Float[Array, "time_steps"]
"""Array of action values over time."""

EnergyArray: TypeAlias = Float[Array, "time_steps"]
"""Array of energy values (kinetic/potential) over time."""

ScalarArray: TypeAlias = Float[Array, ""]
"""Scalar array (0-dimensional)."""


# =============================================================================
# Parameter and Model Types
# =============================================================================

ModelParams: TypeAlias = PyTreeParams
"""Parameters for neural network models."""

ModelState: TypeAlias = Optional[nnx.State]
"""Optional state for stateful models (e.g., batch norm)."""

SplineParams: TypeAlias = Float[Array, "num_control_points param_dim"]
"""Parameters for spline control points in parameter space."""

BoundaryParams: TypeAlias = Tuple[Float[Array, "param_dim"], Float[Array, "param_dim"]]
"""Boundary parameters (θ₀, θ₁) for source and target."""


# =============================================================================
# Configuration Types
# =============================================================================

class ArchitectureConfig(NamedTuple):
    """Configuration for neural network architecture."""
    input_dim: int
    hidden_dim: int
    num_layers: int
    activation: str = "softplus"
    time_varying: bool = True


class SplineConfig(NamedTuple):
    """Configuration for spline interpolation."""
    num_control_points: int
    interpolation_type: str = "cubic"  # "cubic" or "linear"
    time_steps: int = 20


class OptimizationConfig(NamedTuple):
    """Configuration for optimization procedure."""
    path_lr: float = 1e-3
    coupling_lr: float = 1e-4
    path_steps: int = 20
    coupling_steps: int = 20
    boundary_weight: float = 1000.0
    geodesic_warmup_steps: int = 100


class PotentialConfig(NamedTuple):
    """Configuration for potential functions."""
    names: List[str]
    coefficients: Dict[str, float]
    sigma: float = 1.0  # For Fisher information


class ExperimentConfig(NamedTuple):
    """Complete experiment configuration."""
    architecture: ArchitectureConfig
    spline: SplineConfig
    optimization: OptimizationConfig
    potential: PotentialConfig
    device: str = "cpu"
    seed: int = 42


# =============================================================================
# Structured Data Types
# =============================================================================

class SplineControlPoints(NamedTuple):
    """Container for spline control points and metadata."""
    control_points: SplineParams
    boundary_params: BoundaryParams
    time_points: TimeStepsArray
    
    @property
    def num_interior_points(self) -> int:
        """Number of interior control points (excluding boundaries)."""
        return self.control_points.shape[0] - 2
    
    @property
    def total_points(self) -> int:
        """Total number of control points including boundaries."""
        return self.control_points.shape[0]


class TrajectoryData(NamedTuple):
    """Container for sample trajectory data."""
    samples: TrajectoryArray
    velocities: VelocityFieldArray
    times: TimeStepsArray
    log_densities: Optional[Float[Array, "batch_size time_steps"]] = None
    scores: Optional[Float[Array, "batch_size time_steps dim"]] = None


class BoundaryConditions(NamedTuple):
    """Container for boundary condition data."""
    source_samples: SampleArray
    target_samples: SampleArray
    source_params: Float[Array, "param_dim"]
    target_params: Float[Array, "param_dim"]


class OptimizationState(NamedTuple):
    """State container for optimization procedures."""
    iteration: int
    loss: float
    gradient_norm: float
    boundary_error: Tuple[float, float]  # (source_error, target_error)
    action_value: float
    kinetic_energy: float
    potential_energy: float


class EnergyComponents(NamedTuple):
    """Decomposition of total energy."""
    kinetic: EnergyArray
    potential: EnergyArray
    entropy: Optional[EnergyArray] = None
    fisher_info: Optional[EnergyArray] = None
    
    @property
    def total(self) -> EnergyArray:
        """Total energy as sum of components."""
        total = self.kinetic + self.potential
        if self.entropy is not None:
            total = total + self.entropy
        if self.fisher_info is not None:
            total = total + self.fisher_info
        return total


# =============================================================================
# Function Type Protocols
# =============================================================================

class PushforwardFunction(Protocol):
    """Protocol for pushforward map T_θ: R^d → R^d."""
    
    def __call__(
        self, 
        params: ModelParams, 
        samples: SampleArray,
        model_state: Optional[ModelState] = None
    ) -> Tuple[SampleArray, Optional[ModelState]]:
        """Apply pushforward transformation."""
        ...


class VelocityFunction(Protocol):
    """Protocol for velocity field v_θ(t, x): [0,1] × R^d → R^d."""
    
    def __call__(
        self,
        params: ModelParams,
        t: TimeArray,
        x: SampleArray,
        model_state: Optional[ModelState] = None
    ) -> Tuple[VelocityArray, Optional[ModelState]]:
        """Evaluate velocity field."""
        ...


class PotentialFunction(Protocol):
    """Protocol for potential energy functions F(ρ): P(R^d) → R."""
    
    def __call__(
        self,
        samples: Union[SampleArray, TrajectoryArray],
        **kwargs: Any
    ) -> Union[ScalarArray, EnergyArray]:
        """Evaluate potential energy."""
        ...


class ObjectiveFunction(Protocol):
    """Protocol for objective functions used in training."""
    
    def __call__(
        self,
        params: ModelParams,
        key: PRNGKeyArray,
        data_batch: SampleArray,
        **kwargs: Any
    ) -> Tuple[ScalarArray, Dict[str, Any]]:
        """Compute loss and metrics."""
        ...


# =============================================================================
# Optimizer Types
# =============================================================================

OptimizerState: TypeAlias = Any
"""State for JAX optimizers (optax)."""

GradientUpdateFunction: TypeAlias = Callable[
    [ModelParams, ModelParams, OptimizerState], 
    Tuple[ModelParams, OptimizerState]
]
"""Function signature for gradient updates."""


# =============================================================================
# Data Types for Datasets
# =============================================================================

class DatasetSample(NamedTuple):
    """Single sample from a dataset."""
    data: SampleArray
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class DataBatch:
    """Batch of data samples."""
    samples: SampleArray
    labels: Optional[List[str]] = None
    batch_size: int = 0
    
    def __post_init__(self):
        """Set batch_size if not provided."""
        if self.batch_size == 0:
            self.batch_size = self.samples.shape[0]


# =============================================================================
# Evaluation and Metrics Types
# =============================================================================

class WassersteinMetrics(NamedTuple):
    """Wasserstein distance metrics."""
    w2_distance: float
    computation_time: float
    num_samples: int


class ActionMetrics(NamedTuple):
    """Action functional metrics."""
    total_action: float
    kinetic_component: float
    potential_component: float
    path_length: float


class BoundaryMetrics(NamedTuple):
    """Boundary approximation quality metrics."""
    source_w2_error: float
    target_w2_error: float
    source_samples_quality: float
    target_samples_quality: float


class ExperimentMetrics(NamedTuple):
    """Complete experiment evaluation metrics."""
    wasserstein: WassersteinMetrics
    action: ActionMetrics
    boundary: BoundaryMetrics
    computational_time: float
    memory_usage: float


# =============================================================================
# Type Guards and Validation
# =============================================================================

def is_valid_time_array(t: Array) -> bool:
    """Check if array represents valid time points in [0,1]."""
    return jnp.all((t >= 0.0) & (t <= 1.0))


def is_valid_sample_array(x: Array, expected_dim: Optional[int] = None) -> bool:
    """Check if array represents valid samples."""
    if x.ndim < 2:
        return False
    if expected_dim is not None and x.shape[-1] != expected_dim:
        return False
    return True


def is_valid_trajectory_array(traj: Array, expected_dim: Optional[int] = None) -> bool:
    """Check if array represents valid trajectories."""
    if traj.ndim != 3:
        return False
    if expected_dim is not None and traj.shape[-1] != expected_dim:
        return False
    return True


# =============================================================================
# Convenience Type Aliases
# =============================================================================

# Common combinations
SampleBatch: TypeAlias = SampleArray
"""Alias for batch of samples."""

TimeBatch: TypeAlias = TimeArray  
"""Alias for batch of time points."""

ParameterVector: TypeAlias = Float[Array, "param_dim"]
"""Single parameter vector in parameter space."""

LossValue: TypeAlias = ScalarArray
"""Scalar loss value."""

MetricsDict: TypeAlias = Dict[str, Union[float, Array]]
"""Dictionary of training/evaluation metrics."""

# Model construction types
ModelConstructor: TypeAlias = Callable[..., nnx.Module]
"""Function that constructs a neural network model."""

InitializationFunction: TypeAlias = Callable[
    [PRNGKeyArray, SampleArray], 
    Tuple[ModelParams, ModelState]
]
"""Function for initializing model parameters."""