"""
Adjoint method for Neural ODEs - Efficient backpropagation through ODE solvers.

This module implements the adjoint sensitivity method from:
"Neural Ordinary Differential Equations" (Chen et al., 2018)
https://arxiv.org/abs/1806.07366

The adjoint method allows efficient gradient computation by solving a backward ODE
instead of backpropagating through the entire forward ODE solve.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional, Any
from jaxtyping import Array, PyTree
from flax import nnx
from functools import partial

from pdpo.core.types import TimeArray, SampleArray, VelocityArray
from pdpo.ode.solvers import ODESolver, string_2_solver


def flatten_pytree(pytree: PyTree) -> Tuple[Array, Callable]:
    """
    Flatten a PyTree into a 1D array and return unflattening function.

    Args:
        pytree: PyTree to flatten

    Returns:
        flat_array: 1D array of flattened parameters
        unflatten_fn: Function to reconstruct PyTree from flat array
    """
    flat, tree_def = jax.tree_util.tree_flatten(pytree)
    flat_array = jnp.concatenate([x.ravel() for x in flat])

    def unflatten_fn(flat_array):
        sizes = [x.size for x in flat]
        split_arrays = []
        start = 0
        for size, original in zip(sizes, flat):
            split_arrays.append(flat_array[start:start + size].reshape(original.shape))
            start += size
        return jax.tree_util.tree_unflatten(tree_def, split_arrays)

    return flat_array, unflatten_fn


class AdjointODESolver:
    """
    Adjoint ODE solver for efficient gradient computation through Neural ODEs.

    The adjoint method solves an augmented backward ODE to compute gradients
    without storing intermediate forward states.

    Forward ODE:  dx/dt = f(t, x, θ)
    Adjoint ODE:  da/dt = -a^T ∂f/∂x
                  dL/dθ = -∫ a^T ∂f/∂θ dt

    where a is the adjoint state (same shape as x).
    """

    def __init__(self,
                 dynamics_fn: Callable,
                 solver: str = 'euler',
                 dt0: float = 0.01):
        """
        Initialize adjoint ODE solver.

        Args:
            dynamics_fn: Function f(t, x, params) -> dx/dt
            solver: ODE solver name ('euler', 'heun', 'midpoint')
            dt0: Time step size
        """
        self.dynamics_fn = dynamics_fn
        self.solver = string_2_solver(solver)
        self.dt0 = dt0

    def forward_ode(self,
                    t_span: Array,
                    y0: SampleArray,
                    params: PyTree,
                    history: bool = False) -> Tuple[Array, Array]:
        """
        Solve forward ODE and return solution.

        Args:
            t_span: Time points for integration
            y0: Initial state, shape (batch_size, dim)
            params: Model parameters
            history: If True, return full trajectory

        Returns:
            y_final: Final state or trajectory
            t_span: Time points used
        """
        def ode_fn(t, y, args=None):
            return self.dynamics_fn(t, y, params)

        y_final = self.solver(ode_fn, t_span, y0, history=history)
        return y_final, t_span

    def augmented_dynamics(self,
                          t: float,
                          augmented_state: Tuple[Array, Array, Array],
                          params: PyTree,
                          reverse_time: bool = True) -> Tuple[Array, Array, Array]:
        """
        Augmented dynamics for the adjoint method.

        The augmented state consists of:
        - y: Forward state (batch_size, dim)
        - a: Adjoint state (batch_size, dim)
        - vjp_params: Accumulated parameter gradients (flat)

        Args:
            t: Current time
            augmented_state: (y, a, vjp_params)
            params: Model parameters
            reverse_time: If True, integrate backwards in time

        Returns:
            (dy/dt, da/dt, d(vjp_params)/dt)
        """
        y, a, vjp_params_flat = augmented_state

        # Time direction
        t_actual = -t if reverse_time else t

        # Define dynamics that takes full batch
        def dynamics_batched(y_batch, params_):
            return self.dynamics_fn(t_actual, y_batch, params_)

        # Compute forward dynamics and VJPs for state and params simultaneously
        # dy/dt = f(t, y, θ)
        dy_dt, vjp_fn = jax.vjp(lambda y_: dynamics_batched(y_, params), y)

        # Adjoint equation: da/dt = -a^T ∂f/∂y
        # vjp_fn(a) gives us a^T ∂f/∂y
        da_dt = -vjp_fn(a)[0]

        # Parameter gradients: d(∂L/∂θ)/dt = -a^T ∂f/∂θ
        # We need VJP w.r.t. params with cotangent = a
        _, vjp_params_fn = jax.vjp(lambda p: dynamics_batched(y, p), params)
        dL_dparams = vjp_params_fn(a)[0]

        # Flatten parameter gradients
        dL_dparams_flat, _ = flatten_pytree(dL_dparams)

        if reverse_time:
            return -dy_dt, -da_dt, -dL_dparams_flat
        else:
            return dy_dt, da_dt, dL_dparams_flat

    def solve_adjoint(self,
                     t_span: Array,
                     y_final: SampleArray,
                     adjoint_final: SampleArray,
                     params: PyTree) -> Tuple[Array, PyTree]:
        """
        Solve the adjoint ODE backward in time.

        Args:
            t_span: Forward time points
            y_final: Final state from forward pass
            adjoint_final: Initial adjoint state (typically ∂L/∂y_final)
            params: Model parameters

        Returns:
            y0: Initial state (reconstructed)
            grad_params: Gradients w.r.t. parameters
        """
        # Reverse time for backward integration
        t_reversed = jnp.flip(t_span)

        # Initialize augmented state
        params_flat, unflatten_fn = flatten_pytree(params)
        vjp_params_init = jnp.zeros_like(params_flat)

        augmented_init = (y_final, adjoint_final, vjp_params_init)

        # Solve augmented ODE backward
        def aug_ode_fn(t, aug_state, args=None):
            return self.augmented_dynamics(t, aug_state, params, reverse_time=True)

        # Integrate
        solution_history = [augmented_init]
        for i in range(len(t_reversed) - 1):
            aug_new = self.augmented_step(aug_ode_fn, t_reversed, i, solution_history)
            solution_history.append(aug_new)

        # Extract final augmented state
        y0, a0, vjp_params_flat = solution_history[-1]

        # Unflatten parameter gradients
        grad_params = unflatten_fn(vjp_params_flat)

        return y0, grad_params

    def augmented_step(self,
                      aug_ode_fn: Callable,
                      t_list: Array,
                      step_index: int,
                      solution_history: list) -> Tuple[Array, Array, Array]:
        """
        Single integration step for augmented dynamics.

        Args:
            aug_ode_fn: Augmented ODE function
            t_list: Time points
            step_index: Current step index
            solution_history: Previous solutions

        Returns:
            Updated augmented state (y, a, vjp_params)
        """
        dt = t_list[step_index + 1] - t_list[step_index]
        t_current = t_list[step_index]
        aug_current = solution_history[-1]

        # Euler step for augmented system
        y, a, vjp = aug_current
        dy, da, dvjp = aug_ode_fn(t_current, aug_current)

        y_new = y + dt * dy
        a_new = a + dt * da
        vjp_new = vjp + dt * dvjp

        return (y_new, a_new, vjp_new)


def odeint_adjoint(dynamics_fn: Callable,
                   y0: SampleArray,
                   t_span: Array,
                   params: PyTree,
                   solver: str = 'euler',
                   dt0: float = 0.01) -> Array:
    """
    Solve ODE with adjoint method for automatic differentiation.

    This function is differentiable and uses the adjoint method for
    efficient gradient computation.

    Args:
        dynamics_fn: ODE function f(t, y, params)
        y0: Initial state
        t_span: Time points
        params: Model parameters
        solver: ODE solver name
        dt0: Time step size

    Returns:
        y_final: Solution at final time

    Example:
        >>> def f(t, y, params):
        ...     return -params['k'] * y
        >>> y0 = jnp.array([[1.0]])
        >>> t_span = jnp.linspace(0, 1, 11)
        >>> params = {'k': 1.0}
        >>> y_final = odeint_adjoint(f, y0, t_span, params)
        >>> grad_fn = jax.grad(lambda p: odeint_adjoint(f, y0, t_span, p).sum())
        >>> grads = grad_fn(params)
    """
    adjoint_solver = AdjointODESolver(dynamics_fn, solver, dt0)

    @jax.custom_vjp
    def solve_ode(params_):
        y_final, _ = adjoint_solver.forward_ode(t_span, y0, params_, history=False)
        return y_final

    def solve_ode_fwd(params_):
        y_final = solve_ode(params_)
        # Store what we need for backward pass
        return y_final, (y_final, params_)

    def solve_ode_bwd(res, g):
        y_final, params_ = res
        # g is the gradient of loss w.r.t. y_final
        # Solve adjoint ODE to get gradients
        y0_reconstructed, grad_params = adjoint_solver.solve_adjoint(
            t_span, y_final, g, params_
        )
        return (grad_params,)

    solve_ode.defvjp(solve_ode_fwd, solve_ode_bwd)

    return solve_ode(params)


# =============================================================================
# Simplified Adjoint for NNX Models
# =============================================================================

def adjoint_neural_ode(node_model,
                      y0: SampleArray,
                      t_span: Tuple[float, float] = (0.0, 1.0),
                      n_steps: int = 100,
                      method: str = 'discrete') -> Array:
    """
    Neural ODE with adjoint method for NNX models.

    This provides a simpler interface specifically for NeuralODE models
    using Flax NNX.

    Args:
        node_model: NeuralODE model instance
        y0: Initial state
        t_span: (t_start, t_end) tuple
        n_steps: Number of integration steps
        method: 'discrete' or 'continuous' adjoint

    Returns:
        y_final: Final state

    Note:
        For 'discrete' adjoint, we use JAX's automatic differentiation
        through the ODE solver steps (simpler but more memory).

        For 'continuous' adjoint, we implement the adjoint ODE method
        (more complex but memory-efficient for long integrations).
    """
    if method == 'discrete':
        # Use JAX autodiff through solver (simple, works well for short integrations)
        return node_model(y0, t_span=t_span, history=False)

    elif method == 'continuous':
        # Use continuous adjoint method (memory efficient for long integrations)
        t_list = jnp.linspace(t_span[0], t_span[1], n_steps)

        def dynamics(t, y, params):
            graphdef, _ = nnx.split(node_model.dynamics)
            model = nnx.merge(graphdef, params)
            return model(y)

        _, params = nnx.split(node_model.dynamics)

        return odeint_adjoint(dynamics, y0, t_list, params,
                            solver='euler', dt0=(t_span[1] - t_span[0]) / n_steps)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'discrete' or 'continuous'.")


# =============================================================================
# Utility Functions
# =============================================================================

def checkpointed_odeint(dynamics_fn: Callable,
                       y0: SampleArray,
                       t_span: Array,
                       params: PyTree,
                       checkpoint_steps: int = 10) -> Array:
    """
    ODE integration with gradient checkpointing.

    Saves memory by only storing checkpoints and recomputing
    intermediate states during backpropagation.

    Args:
        dynamics_fn: ODE function
        y0: Initial state
        t_span: Time points
        params: Parameters
        checkpoint_steps: Number of steps between checkpoints

    Returns:
        y_final: Final state
    """
    # Split time span into segments
    n_segments = len(t_span) // checkpoint_steps
    segments = jnp.array_split(t_span, n_segments)

    y = y0
    for segment in segments:
        # Use checkpoint for each segment
        @jax.checkpoint
        def solve_segment(y_in):
            return odeint_adjoint(dynamics_fn, y_in, segment, params)

        y = solve_segment(y)

    return y
