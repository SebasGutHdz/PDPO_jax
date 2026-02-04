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
from pdpo.spline.types_interpolation import SplineState 



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



def linear_interp(
    t: Float[Array, "T"],
    xt: List[PyTree],
    s_query: Float[Array, "S"]
) -> List[PyTree]:
    
    x_vals = jax.tree.map(lambda *args: jnp.stack(args), *xt)
    def interp_one_time(s):
        # Find index of left point
        idx = jnp.searchsorted(t, s, side="right") - 1
        idx = jnp.clip(idx, 0, len(t) - 2)
        # idx_item = idx.item()
        t0, t1 = t[idx], t[idx + 1]
        # xt0 = xt[idx_item]#jax.tree.map(lambda arr: arr[idx], xt)
        # xt1 = xt[idx_item+1]#jax.tree.map(lambda arr: arr[idx + 1], xt)
        def index(pytree, i):
            return jax.tree.map(lambda arr: jax.lax.dynamic_index_in_dim(arr, i, axis=0, keepdims=False), pytree)

        xt0 = index(x_vals, idx)
        xt1 = index(x_vals, idx + 1)
        alpha = (s - t0) / (t1 - t0)
        return jax.tree.map(lambda a, b: (1 - alpha) * a + alpha * b, xt0, xt1)

    return jax.vmap(interp_one_time)(s_query)


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
    T = len(xt)

    # Stack PyTree list into batched PyTree with leading time axis
    x_vals = jax.tree.map(lambda *args: jnp.stack(args), *xt)  # [T, ...]

    # Compute finite differences: fd[i] = (x[i+1] - x[i]) / (t[i+1] - t[i])
    dt = jnp.diff(t)  # [T-1]

    def compute_fd(x_arr):
        # x_arr has shape [T, ...]
        return (x_arr[1:] - x_arr[:-1]) / (dt.reshape((-1,) + (1,) * (x_arr.ndim - 1)) + 1e-10)

    fd = jax.tree.map(compute_fd, x_vals)  # [T-1, ...]

    # Compute tangents m using average of adjacent finite differences
    def compute_tangents(fd_arr):
        # fd_arr has shape [T-1, ...]
        # Interior points: average of adjacent finite differences
        m_interior = (fd_arr[1:] + fd_arr[:-1]) / 2  # [T-2, ...]
        # Left endpoint: use first finite difference
        m_left = fd_arr[0:1]  # [1, ...]
        # Right endpoint: use last finite difference
        m_right = fd_arr[-1:]  # [1, ...]
        return jnp.concatenate([m_left, m_interior, m_right], axis=0)  # [T, ...]

    m_vals = jax.tree.map(compute_tangents, fd)  # [T, ...]

    # Interpolation function (for one s)
    def interpolate_single_time(sj):
        idx = jnp.searchsorted(t[1:], sj, side="left")
        idx = jnp.clip(idx, 0, T - 2)
        right_idx = (idx + 1) % T

        t0, t1 = t[idx], t[right_idx]
        dx = t1 - t0 + 1e-10
        tau = (sj - t0) / dx

        def index(pytree, i):
            return jax.tree.map(lambda arr: jax.lax.dynamic_index_in_dim(arr, i, axis=0, keepdims=False), pytree)

        p0 = index(x_vals, idx)
        p1 = index(x_vals, right_idx)
        m0 = index(m_vals, idx)
        m1 = index(m_vals, right_idx)

        # Standard Hermite basis functions
        h00 = 2 * tau**3 - 3 * tau**2 + 1
        h10 = tau**3 - 2 * tau**2 + tau
        h01 = -2 * tau**3 + 3 * tau**2
        h11 = tau**3 - tau**2

        # Hermite interpolation: p(τ) = h00*p0 + h10*dt*m0 + h01*p1 + h11*dt*m1
        # m0, m1 are velocities (dp/dt), so we scale by dt to get tangent vectors
        return jax.tree.map(
            lambda p0_, p1_, m0_, m1_: (
                h00 * p0_ + h10 * dx * m0_ + h01 * p1_ + h11 * dx * m1_
            ),
            p0, p1, m0, m1
        )

    # Vectorize over s
    return jax.vmap(interpolate_single_time)(s)

def unstack_pytree(batched_tree):
    """
    Convert a batched PyTree (with leading time axis T) into a list of T individual PyTrees.
    """
    T = jax.tree_util.tree_leaves(batched_tree)[0].shape[0]
    return [jax.tree_util.tree_map(lambda x: x[i], batched_tree) for i in range(T)]


def interp(spline_state: SplineState, query_t: TimeStepsArray) -> List[PyTree]:
    """
    Interpolate spline at query times.
    
    Args:
        spline_state: Spline state containing control points
        query_t: Query time points
        
    Returns:
        Interpolated parameters at query times
    """
    # Combine boundary and interior points
    theta0, theta1 = spline_state.boundary_params
    all_points = [theta0] + spline_state.control_points + [theta1]
    
    if spline_state.config.spline_type == 'cubic':
        return unstack_pytree(cubic_interp(spline_state.time_points, all_points, query_t))
    else:  # linear
        return unstack_pytree(linear_interp(spline_state.time_points, all_points, query_t))