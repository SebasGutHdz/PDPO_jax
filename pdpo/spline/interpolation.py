"""
JAX implementation of interpolation functions for PDPO splines.
Supports PyTree structures for control points.
"""

import jax
import jax.numpy as jnp
from typing import List
from jaxtyping import Array, Float, PyTree
from typing import Tuple

from pdpo.core.types import TimeStepsArray




def linear_interpolation_states(state0,state1,t_points):
    '''
    Linear interpolation of two parameters
    '''
    param_series = []

    for t in t_points:
        
        param = jax.tree.map(
            lambda a,b: (1-t)*a + t*b,
            state0,state1
            )
        
        param_series.append(param)
    

    return param_series

def cubic_interp(
    t: Float[Array, "T"],        # Control point times
    xt: List[PyTree],                  # Control point parameters (PyTree) 
    s: Float[Array, "S"]         # Query times
) -> PyTree:                     # Interpolated parameters (same PyTree structure)
    """
    Cubic Hermite interpolation of PyTree parameters over time.
    
    Args:
        t: Time points at control points, shape (T,)
        xt: PyTree with leaves representing parameters at control points
        s: Query time points, shape (S,)
        
    Returns:
        PyTree with same structure as xt, interpolated at query times s
    """
    def compute_derivatives(x, t):
        dt = jnp.diff(t)

        def get_dx(i):
            if i == 0:
                return jax.tree.map(lambda a, b: (b - a) / dt[0], x[0], x[1])
            elif i == len(x) - 1:
                return jax.tree.map(lambda a, b: (a - b) / dt[-1], x[-1], x[-2])
            else:
                h = t[i + 1] - t[i - 1]
                return jax.tree.map(lambda a, b: (b - a) / h, x[i + 1], x[i - 1])

        return [ (x[i], get_dx(i)) for i in range(len(x)) ]

    # Step 1: Compute derivatives
    xt = compute_derivatives(xt, t)

    # Step 2: Convert list of tuples to tuple of PyTrees with leading time axis
    x_vals, dx_vals = zip(*xt)  # unzip
    x_vals = jax.tree.map(lambda *args: jnp.stack(args), *x_vals)    # [T, ...]
    dx_vals = jax.tree.map(lambda *args: jnp.stack(args), *dx_vals)  # [T, ...]

    # Step 3: Interpolation function (for one s)
    def interpolate_single_time(sj):
        idx = jnp.searchsorted(t, sj, side="right") - 1
        idx = jnp.clip(idx, 0, t.shape[0] - 2)

        t0, t1 = t[idx], t[idx + 1]
        dt = t1 - t0
        tau = (sj - t0) / dt

        h00 = 2 * tau**3 - 3 * tau**2 + 1
        h10 = tau**3 - 2 * tau**2 + tau
        h01 = -2 * tau**3 + 3 * tau**2
        h11 = tau**3 - tau**2

        def index(pytree, i):
            return jax.tree.map(lambda arr: jax.lax.dynamic_index_in_dim(arr, i, axis=0, keepdims=False), pytree)

        x0 = index(x_vals, idx)
        x1 = index(x_vals, idx + 1)
        dx0 = index(dx_vals, idx)
        dx1 = index(dx_vals, idx + 1)

        return jax.tree.map(
            lambda a, b, c, d: h00 * a + h10 * dt * b + h01 * c + h11 * dt * d,
            x0, dx0, x1, dx1
        )

    # Step 4: Vectorize over s
    return jax.vmap(interpolate_single_time)(s)

def unstack_pytree(batched_tree):
    """
    Convert a batched PyTree (with leading time axis T) into a list of T individual PyTrees.
    """
    T = jax.tree_util.tree_leaves(batched_tree)[0].shape[0]
    return [jax.tree_util.tree_map(lambda x: x[i], batched_tree) for i in range(T)]
