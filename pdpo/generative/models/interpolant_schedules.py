
import jax.numpy as jnp

from pdpo.core.types import (
    ModelParams,
    SampleArray,
    TimeArray,
    VelocityArray,
    ScoreArray
)

class InterpolantSchedule:
    """Base class for stochastic interpolant coefficient schedules."""
    
    def __init__(self, schedule_type: str = "linear", **params):
        self.schedule_type = schedule_type
        self.params = params
        
    def alpha(self, t: TimeArray) -> TimeArray:
        """Coefficient for x_0 in interpolant."""
        if self.schedule_type == "linear":
            return 1 - t
        elif self.schedule_type == "trigonometric":
            return jnp.cos(jnp.pi * t / 2)
        elif self.schedule_type == "polynomial":
            power = self.params.get("power", 2.0)
            return 1 - t**power
        elif self.schedule_type == "vp":  # Variance preserving
            beta_t = t
            return jnp.sqrt(1 - beta_t**2)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def beta(self, t: TimeArray) -> TimeArray:
        """Coefficient for x_1 in interpolant."""
        if self.schedule_type == "linear":
            return t
        elif self.schedule_type == "trigonometric":
            return jnp.sin(jnp.pi * t / 2)
        elif self.schedule_type == "polynomial":
            power = self.params.get("power", 2.0)
            return t**power
        elif self.schedule_type == "vp":  # Variance preserving
            return t
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def gamma(self, t: TimeArray) -> TimeArray:
        """Coefficient for noise in interpolant."""
        sigma = self.params.get("sigma", 0.1)
        
        if self.schedule_type == "linear":
            return sigma * jnp.sqrt(t * (1 - t))
        elif self.schedule_type == "trigonometric":
            return sigma * jnp.sqrt(t * (1 - t))
        elif self.schedule_type == "polynomial":
            return sigma * jnp.sqrt(t * (1 - t))
        elif self.schedule_type == "vp":  # Variance preserving
            beta_t = t
            return beta_t
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def get_coefficient_functions(self):
        """Return coefficient functions for use with StochasticInterpolantsObjective."""
        return self.alpha, self.beta, self.gamma

# =============================================================================
# Utility Functions for Common Schedules
# =============================================================================

def create_linear_schedule(**params) -> InterpolantSchedule:
    """Create linear interpolant schedule (equivalent to Flow Matching)."""
    return InterpolantSchedule("linear", **params)


def create_trigonometric_schedule(**params) -> InterpolantSchedule:
    """Create smooth trigonometric interpolant schedule."""
    return InterpolantSchedule("trigonometric", **params)


def create_polynomial_schedule(power: float = 2.0, **params) -> InterpolantSchedule:
    """Create polynomial interpolant schedule."""
    return InterpolantSchedule("polynomial", power=power, **params)


def create_vp_schedule(**params) -> InterpolantSchedule:
    """Create variance-preserving interpolant schedule."""
    return InterpolantSchedule("vp", **params)