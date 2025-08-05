"""
Basic ODE solver for flow based generative models
"""

from abc import ABC, abstractmethod
from typing import Any, Callable,Optional,Tuple
import jax.numpy as jnp
from jaxtyping import Array,Float


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
        self,
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