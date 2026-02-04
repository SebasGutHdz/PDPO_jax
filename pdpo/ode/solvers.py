"""
Basic ODE solver for flow based generative models
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple
import jax.numpy as jnp
from jaxtyping import Array, Float
from flax import nnx
from functools import partial
import jax


from pdpo.core.types import (
    TimeArray,
    SampleArray,
    VelocityArray,
)


def string_2_solver(solver_str: str) -> 'ODESolver':
    """
    Converts a string to an ODE solver instance.

    Args:
        solver_str: Name of the solver ('euler', 'heun', 'midpoint')

    Returns:
        solver: The corresponding ODE solver instance
    """
    if solver_str == 'euler':
        return EulerSolver()
    elif solver_str == 'heun':
        return HeunSolver()
    elif solver_str == 'midpoint':
        return MidpointSolver()
    else:
        raise ValueError(f'Solver {solver_str} not recognized. Available solvers: euler, heun, midpoint.')


class ODESolver(nnx.Module):
    """
    Base class for ODE solvers compatible with Flax NNX.
    Provides unified interface for solving ODEs with history tracking.
    """

    def __init__(self):
        pass

    def step(self, f: Callable, t_list: Array, step_index: int, solution_history: list) -> Array:
        """
        Perform one integration step. Must be implemented by subclasses.

        Args:
            f: ODE function dy/dt = f(t, y)
            t_list: Array of time points
            step_index: Current time step index
            solution_history: List of previous solutions

        Returns:
            y_new: Solution at next time point
        """
        raise NotImplementedError("Subclasses must implement the step method")

    def __call__(self, f: Callable, t_list: Array, y0: Array, history: bool = False) -> Array:
        """
        Solve the ODE dy/dt = f(t, y) from t_list[0] to t_list[-1].

        Args:
            f: ODE function dy/dt = f(t, y), where t is float and y is shape [bs, d]
            t_list: Array of time points
            y0: Initial value at t0, shape [bs, d]
            history: If True, return solution at all time points

        Returns:
            y: Solution at all time points (if history=True) or final value only
        """
        solution_history = [y0]

        for i in range(len(t_list) - 1):
            y_new = self.step(f, t_list, i, solution_history)
            solution_history.append(y_new)

        if history:
            return jnp.array(solution_history).transpose((1, 0, 2))  # Shape (batch_size, time_steps, dim)
        else:
            return solution_history[-1]


class EulerSolver(ODESolver):
    """Euler's method for solving ODEs."""

    def __init__(self):
        super().__init__()

    def step(self, f: Callable, t_list: Array, step_index: int, solution_history: list) -> Array:
        """
        Perform one Euler integration step.

        Args:
            f: ODE function dy/dt = f(t, y)
            t_list: Array of time points
            step_index: Current time step index
            solution_history: List of previous solutions

        Returns:
            y_new: Solution at next time point
        """
        dt = t_list[step_index + 1] - t_list[step_index]
        t_current = t_list[step_index]
        y_current = solution_history[-1]
        y_new = y_current + dt * f(t_current, y_current)
        return y_new


class HeunSolver(ODESolver):
    """Heun's method (improved Euler) for solving ODEs."""

    def __init__(self):
        super().__init__()

    def step(self, f: Callable, t_list: Array, step_index: int, solution_history: list) -> Array:
        """
        Perform one Heun integration step.

        Args:
            f: ODE function dy/dt = f(t, y)
            t_list: Array of time points
            step_index: Current time step index
            solution_history: List of previous solutions

        Returns:
            y_new: Solution at next time point
        """
        dt = t_list[step_index + 1] - t_list[step_index]
        t_current = t_list[step_index]
        y_current = solution_history[-1]

        # Predictor step (Euler)
        y_predictor = y_current + dt * f(t_current, y_current)

        # Corrector step
        y_new = y_current + (dt / 2) * (f(t_current, y_current) + f(t_list[step_index + 1], y_predictor))

        return y_new


class MidpointSolver(ODESolver):
    """Midpoint method for solving ODEs."""

    def __init__(self):
        super().__init__()

    def step(self, f: Callable, t_list: Array, step_index: int, solution_history: list) -> Array:
        """
        Perform one midpoint integration step.

        Args:
            f: ODE function dy/dt = f(t, y)
            t_list: Array of time points
            step_index: Current time step index
            solution_history: List of previous solutions

        Returns:
            y_new: Solution at next time point
        """
        dt = t_list[step_index + 1] - t_list[step_index]
        t_current = t_list[step_index]
        y_current = solution_history[-1]

        # Midpoint evaluation
        y_mid = y_current + 0.5 * dt * f(t_current, y_current)
        t_mid = t_current + 0.5 * dt

        # Full step using midpoint derivative
        y_new = y_current + dt * f(t_mid, y_mid)

        return y_new



def eval_model(
        model: nnx.Module,
        t: TimeArray,
        x: SampleArray,
        time_dependent: bool = False
    ) -> VelocityArray:
    """
    Evaluate the velocity field model with proper time conditioning.

    Args:
        model: The neural network model
        t: Time values, float or jnp.array shape (batch_size,) or (batch_size,1)
        x: Sample positions, shape (batch_size, dim)
        time_dependent: Whether the model is time-dependent

    Returns:
        Predicted velocities, shape (batch_size, dim)
    """
    if not time_dependent:
        model_input = x
    else:
        if t.ndim == 0:  # element from jnp.array
            t_expanded = jnp.full((x.shape[0], 1), t)
        elif t.ndim == 1:  # Batch of times with format (bs,)
            t_expanded = t.reshape(-1, 1)
        elif t.ndim == 2:  # Batch of times with correct format
            t_expanded = t
        else:
            raise ValueError("t does not have the right shape, valid float of jnp with shapes (bs,) and (bs,1)")

        model_input = jnp.concatenate([t_expanded, x], axis=-1)

    v_pred = model(model_input)
    return v_pred

def sample_trajectory(
        vf: nnx.Module,
        x0: SampleArray,
        ode_solver: Optional[ODESolver] = None,
        n_steps: int = 10,
        backward: bool = False
):
    """
    Sample a trajectory using the ODE solver with the velocity field model.

    Args:
        vf: Velocity field model (nnx.Module)
        x0: Initial state, shape (batch_size, dim)
        ode_solver: ODE solver instance (if None, uses EulerSolver())
        n_steps: Number of time steps
        backward: If True, integrate backward from t=1 to t=0
        time_dependent: Whether the velocity field is time-dependent

    Returns:
        x: Final state, shape (batch_size, dim)
    """
    if ode_solver is None:
        ode_solver = EulerSolver()

    if backward:
        # Reverse the order of time steps for backward integration
        t = jnp.linspace(1.0, 0.0, n_steps)
    else:
        # Forward time steps
        t = jnp.linspace(0, 1, n_steps)

    # Define the vector field function
    def vector_field(t_val, x):
        return eval_model(vf, t_val, x, time_dependent=vf.time_varying)

    # Use the solver
    x = ode_solver(
        f=vector_field,
        t_list=t,
        y0=x0,
        history=False
    )
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