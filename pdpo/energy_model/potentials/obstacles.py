"""
JAX implementation of obstacle potential functions for PDPO.

This module provides various obstacle cost functions that create potential barriers
in the sample space, preventing particle trajectories from passing through
forbidden regions.
"""

import jax
import jax.numpy as jnp
from jax import nn
from jaxtyping import Array, Float
from typing import Tuple, Dict, Any
import math
from jax import lax

from pdpo.core.types import TrajectoryArray, EnergyArray


# =============================================================================
# Obstacle Configuration Functions
# =============================================================================

def obstacle_cfg_stunnel() -> Tuple[float, float, float, list, float]:
    """Configuration for S-tunnel obstacles."""
    a, b, c = 20, 1, 90
    centers = [[5, 6], [-5, -6]]
    
    return a, b, c, centers


def obstacle_cfg_gmm() -> Tuple[list, float]:
    """Configuration for Gaussian mixture model obstacles."""
    centers = [[6, 6], [6, -6], [-6, -6]]
    radius = 1.5
    
    return centers, radius


def obstacle_cfg_vneck() -> Tuple[float, float]:
    """Configuration for V-neck obstacle."""
    c_sq = 0.36
    coef = 2
    return c_sq, coef


# =============================================================================
# Obstacle Cost Functions
# =============================================================================

def obstacle_cost_gmm(xt: TrajectoryArray, key = None) -> EnergyArray:
    """
    Gaussian mixture model obstacles.
    
    Args:
        xt: Sample trajectories, shape (batch_size, time_steps, 2)
        
    Returns:
        Cost array, shape (batch_size, time_steps)
    """
    # Handle input shape
    original_shape = xt.shape[:-1]
    xt_flat = xt.reshape(-1, xt.shape[-1])
    batch_size = xt_flat.shape[0]
    
    centers, radius = obstacle_cfg_gmm()
    
    # Convert centers to JAX arrays
    obs1 = jnp.array(centers[0]).reshape(1, -1).repeat(batch_size, axis=0)
    obs2 = jnp.array(centers[1]).reshape(1, -1).repeat(batch_size, axis=0)
    obs3 = jnp.array(centers[2]).reshape(1, -1).repeat(batch_size, axis=0)
    
    # Compute distances
    dist1 = jnp.linalg.norm(xt_flat - obs1, axis=-1)
    dist2 = jnp.linalg.norm(xt_flat - obs2, axis=-1)
    dist3 = jnp.linalg.norm(xt_flat - obs3, axis=-1)
    
    # Compute costs using softplus
    cost1 = nn.softplus(100 * (radius - dist1))
    cost2 = nn.softplus(100 * (radius - dist2))
    cost3 = nn.softplus(100 * (radius - dist3))
    
    total_cost = (cost1 + cost2 + cost3) 
    return total_cost.reshape(original_shape)


def obstacle_cost_vneck(xt: TrajectoryArray ,key = None) -> EnergyArray:
    """
    V-neck shaped obstacle.
    
    Args:
        xt: Sample trajectories, shape (batch_size, time_steps, 2)
        
    Returns:
        Cost array, shape (batch_size, time_steps)
    """
    assert xt.shape[-1] == 2, "V-neck obstacle requires 2D input"
    
    c_sq, coef = obstacle_cfg_vneck()
    
    xt_sq = jnp.square(xt)
    d = coef * xt_sq[..., 0] - xt_sq[..., 1]
    
    cost = nn.softplus(-c_sq - d) 
    return cost


# def obstacle_cost_stunnel(xt: TrajectoryArray,key = None) -> EnergyArray:
#     """
#     S-tunnel obstacles with rotated elliptical barriers.
    
#     Args:
#         xt: Sample trajectories, shape (batch_size, time_steps, d)
#         scale: Scaling factor for obstacle cost
        
#     Returns:
#         Cost array, shape (batch_size, time_steps)
#     """
#     # Extract spatial coordinates (first 2 dimensions)
#     xx = xt[..., :2]
#     original_shape = xx.shape[:-1]
#     xx_flat = xx.reshape(-1, 2)
#     batch_size = xx_flat.shape[0]
#     _,_,_,_,scale = obstacle_cfg_stunnel()
#     # Rotation matrix
#     theta = jnp.pi / 5
#     cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
#     rot_mat = jnp.array([[cos_theta, -sin_theta],
#                          [sin_theta, cos_theta]])
    
#     # Bottom/Left obstacle
#     center1 = jnp.array([-2.0, 0.5])
#     xxcent1 = xx_flat - center1
#     xxcent1_rot = xxcent1 @ rot_mat.T
    
#     covar_mat1 = jnp.array([[5.0, 0.0], [0.0, 0.0]])
#     bb_vec1 = jnp.array([0.0, 2.0])
    
#     xxcov1 = xxcent1_rot @ covar_mat1
#     quad1 = jnp.sum(xxcov1 * xxcent1_rot, axis=1)
#     lin1 = jnp.sum(xxcent1_rot * bb_vec1, axis=1)
#     out1 = jnp.maximum(0.0, -((quad1 + lin1) + 1)) * scale
    
#     # Top/Right obstacle
#     center2 = jnp.array([2.0, -0.5])
#     xxcent2 = xx_flat - center2
#     xxcent2_rot = xxcent2 @ rot_mat.T
    
#     covar_mat2 = jnp.array([[5.0, 0.0], [0.0, 0.0]])
#     bb_vec2 = jnp.array([0.0, -2.0])
    
#     xxcov2 = xxcent2_rot @ covar_mat2
#     quad2 = jnp.sum(xxcov2 * xxcent2_rot, axis=1)
#     lin2 = jnp.sum(xxcent2_rot * bb_vec2, axis=1)
#     out2 = jnp.maximum(0.0, -((quad2 + lin2) + 1)) * scale
    
#     total_cost = (out1 + out2) * 100
#     return total_cost.reshape(original_shape)

def obstacle_cost_stunnel(xt: TrajectoryArray,key = None) -> EnergyArray:
    """
    S-tunnel obstacles with rotated elliptical barriers.
    
    Args:
        xt: Sample trajectories, shape (batch_size, time_steps, d)
        scale: Scaling factor for obstacle cost
        
    Returns:
        Cost array, shape (batch_size, time_steps)
    """

    a,b,c,centers = obstacle_cfg_stunnel()

    # Extract spatial coordinates (first 2 dimensions)

    assert xt.shape[-1] == 2, "s-tunnel obstacle requires 2D input"
    
    x,y = xt[...,0], xt[...,1]
    # print(x.shape,y.shape)
    d = a*(x-centers[0][0])**2 + b*(y-centers[0][1])**2
    c1 = nn.softplus(c-d)
    d = a*(x-centers[1][0])**2 + b*(y-centers[1][1])**2
    c2 = nn.softplus(c-d)
    cost = (c1 + c2) 
    return cost.reshape(xt.shape[:-1])

def congestion_cost(xt: TrajectoryArray, key: jax.Array) -> EnergyArray:
    """
    Congestion cost to prevent particle crowding.
    
    Args:
        xt: Sample trajectories, shape (batch_size, time_steps, dim)
        key: JAX random key for permutation
        
    Returns:
        Cost array, shape (batch_size, time_steps)
    """
    batch_size, time_steps, dim = xt.shape
    
    # Create random permutation for pairing
    perm = jax.random.permutation(key, batch_size)
    yt = xt[perm]
    
    # Compute pairwise distances
    dd = xt - yt
    dist = jnp.sum(dd**2, axis=-1)
    
    # Congestion potential
    congestion = 2.0 / (dist + 1e-6)
    
    return congestion * 5.0


def quadratic_well(xt: TrajectoryArray, key = None) -> EnergyArray:
    """
    Attractive quadratic potential well.
    
    Args:
        xt: Sample trajectories, shape (batch_size, time_steps, dim)
        
    Returns:
        Cost array, shape (batch_size, time_steps)
    """
    dim = xt.shape[-1]
    
    # Increased coefficient matrix
    A = jnp.eye(dim) * 2.5
    
    # Base quadratic potential
    base_potential = -0.5 * jnp.sum((xt @ A) * xt, axis=-1)
    
    # Distance from center for sharpening
    center_dist = jnp.sum(xt**2, axis=-1)
    
    # Exponential sharpening near center
    sharpening_factor = jnp.exp(-center_dist * 0.5) + 1.0
    
    # Final potential with enhanced center sharpness
    sharp_potential = base_potential * sharpening_factor
    
    return sharp_potential


def geodesic(xt: TrajectoryArray,key = None) -> EnergyArray:
    """
    Zero potential for geodesic problems.
    
    Args:
        xt: Sample trajectories, shape (batch_size, time_steps, dim)
        
    Returns:
        Zero cost array, shape (batch_size, time_steps)
    """
    batch_size, time_steps = xt.shape[:2]
    return jnp.zeros((batch_size, time_steps))


# =============================================================================
# Obstacle Function Registry
# =============================================================================

OBSTACLE_FUNCTIONS = {
    'obstacle_cost_stunnel': obstacle_cost_stunnel,
    'obstacle_cost_vneck': obstacle_cost_vneck,
    'obstacle_cost_gmm': obstacle_cost_gmm,
    'congestion_cost': congestion_cost,
    'quadratic_well': quadratic_well,
    'geodesic': geodesic,
}


def get_obstacle_function(name: str):
    """Get obstacle function by name."""
    if name not in OBSTACLE_FUNCTIONS:
        raise ValueError(f"Unknown obstacle function: {name}. Available: {list(OBSTACLE_FUNCTIONS.keys())}")
    return OBSTACLE_FUNCTIONS[name]


def list_obstacle_functions() -> list[str]:
    """List all available obstacle functions."""
    return list(OBSTACLE_FUNCTIONS.keys())



# Deterministic ordering of potential functions
POTENTIAL_NAMES = tuple(sorted(OBSTACLE_FUNCTIONS.keys()))
POTENTIALS_JAX = tuple(OBSTACLE_FUNCTIONS[name] for name in POTENTIAL_NAMES)

# Mapping name → index
POTENTIAL_NAME_TO_IDX = {name: i for i, name in enumerate(POTENTIAL_NAMES)}

def potential_dispatch(samples_path, fn_idx, key=None):
    """
    Dispatch to a potential function given by its integer index.
    Uses lax.switch for JIT compatibility.
    """
    return lax.switch(fn_idx, POTENTIALS_JAX, samples_path,key)