"""
Basic ODE solver for flow based generative models
"""

from abc import ABC, abstractmethod
from typing import Any, Callable,Optional,Tuple
import jax.numpy as jnp
from jaxtyping import Array,Float
from flax import nnx


from pdpo.core.types import (
    TimeArray,
    SampleArray,
    VelocityArray,
)


class ODESolver(ABC):
    ''' Abstract base class for ODE solvers '''
    def __init__(self, name: str, order: int):
        self.name = name
        self.order = order

    def __repr__(self):
        return f"{self.name}(order={self.order})"
    
    @abstractmethod
    def step(
        self,
        f: Callable,
        t: float,
        x: SampleArray,
        dt: float,
        args: Optional[Tuple[Any,...]] = None        
    ) -> SampleArray:
        pass


class EulerSolver(ODESolver):
    '''Euler method'''
    def __init__(self):
        super().__init__(name='euler', order=1)

    def step(
        self,
        f: Callable,
        t: float,
        x: SampleArray,
        dt: float,
        args: Optional[Tuple[Any,...]] = None        
    ) -> SampleArray:
        args = args or ()
        k1 = f(t, x, *args)
        return x + dt * k1


class MidpointSolver(ODESolver):
    '''Midpoint method'''
    def __init__(self):
        super().__init__(name='midpoint', order=2)

    def step(
        f: Callable,
        t: float,
        x: SampleArray,
        dt: float,
        args: Optional[Tuple[Any,...]] = None        
    ) -> SampleArray:
        args = args or ()
        k1 = f(t, x, *args)
        x_mid = x + 0.5 * dt * k1
        t_mid = t + 0.5 * dt
        k2 = f(t_mid, x_mid, *args)
        return x + dt * k2



def eval_model(
        model: nnx.Module,
        t: TimeArray,
        x: SampleArray
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
        t_expanded = jnp.full((x.shape[0], 1), t)
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
        vf: nnx.Module,
        x0: SampleArray,
        ode_solver:ODESolver = MidpointSolver,
        n_steps: int = 10,
        backward: bool = False,
):
    """
    Sample a trajectory using the ODE solver with the velocity field model.
    """
    if backward:
        # Reverse the order of time steps for backward integration
        t = jnp.linspace(1.0, 0.0, n_steps)
    else:
        # Forward time steps    
        t = jnp.linspace(0, 1, n_steps)
    x = x0
    dt = (t[1]-t[0])
    eval_model_ = lambda t,x: eval_model(vf,t,x)
    for i in range(n_steps - 1):
        x = ode_solver.step(
            f = eval_model_,
            t = t[i], 
            x = x, 
            dt = dt)
    return x




# Usage example 
# def solve_ode(
#     solver: ODESolver,
#     f: Callable,
#     x0: SampleArray,
#     t_span: Float[Array, "n_steps"],
#     args: Optional[Tuple[Any, ...]] = None
# ) -> SampleArray:
#     n_steps = len(t_span)
#     trajectory = jnp.zeros((n_steps,) + x0.shape)
#     trajectory = trajectory.at[0].set(x0)

#     x = x0
#     for i in range(n_steps - 1):
#         t = t_span[i]
#         dt = t_span[i + 1] - t
#         x = solver.step(f, x, t, dt, args=args)
#         trajectory = trajectory.at[i + 1].set(x)

#     return trajectory