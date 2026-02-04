"""
Common potential functions for density path optimization.

This module provides pre-defined potential functions and a factory for creating
LinearPotential instances with these functions.
"""

from typing import Callable, Optional, Any
from jaxtyping import PyTree, Array
import jax
import jax.numpy as jnp
from flax import nnx

from pdpo.energy_model.functionals.linear_functional import LinearPotential


# =============================================================================
# Potential Function Definitions
# =============================================================================

@jax.jit
def quadratic_potential_fn(x: Array) -> Array:
    """
    Quadratic potential U(x) = |x|²/2

    Args:
        x: Positions array of shape (batch_size, d)

    Returns:
        Potential values of shape (batch_size,)
    """
    return jnp.sum(x**2, axis=-1) / 2.0


@jax.jit
def double_well_potential_fn(x: Array, alpha: float = 1.0) -> Array:
    """
    Double-well potential U(x,y) = (y² - 2)² + α*x²

    Creates two wells along the y-axis with harmonic confinement in x.

    Args:
        x: Positions array of shape (batch_size, 2)
        alpha: Confinement strength in x-direction

    Returns:
        Potential values of shape (batch_size,)
    """
    x_coord = x[:, 0]  # x-coordinates
    y_coord = x[:, 1]  # y-coordinates

    # Double-well in y: (y² - 2)²
    y_term = (y_coord**2 - 2.0)**2

    # Harmonic confinement in x: αx²
    x_term = alpha * x_coord**2

    return y_term + x_term


@jax.jit
def four_well_potential_fn(x: Array, alpha: float = 1.0) -> Array:
    """
    Four-well potential U(x,y) = α*(y² - 2)² + α*(x² - 2)²

    Creates four wells at approximately (±√2, ±√2).

    Args:
        x: Positions array of shape (batch_size, 2)
        alpha: Strength of the quartic terms

    Returns:
        Potential values of shape (batch_size,)
    """
    x_coord = x[:, 0]  # x-coordinates
    y_coord = x[:, 1]  # y-coordinates

    # Four-well in y: α*(y² - 2)²
    y_term = alpha * (y_coord**2 - 2.0)**2

    # Four-well in x: α*(x² - 2)²
    x_term = alpha * (x_coord**2 - 2.0)**2

    return y_term + x_term


@jax.jit
def quartic_potential_fn(x: Array, strength: float = 0.1) -> Array:
    """
    Quartic potential U(x) = strength * |x|⁴ + 0.5 * |x|²

    Provides softer confinement than quadratic potential.

    Args:
        x: Positions array of shape (batch_size, d)
        strength: Strength of quartic term

    Returns:
        Potential values of shape (batch_size,)
    """
    r_squared = jnp.sum(x**2, axis=-1)
    return strength * r_squared**2 + 0.5 * r_squared


@jax.jit
def styblinski_tang_potential_fn(x: Array, d: int = 2) -> Array:
    """
    Styblinski-Tang potential U(x) = 0.5 * Σ (x_i^4 - 16*x_i^2 + 5*x_i)

    Global minimum at x_i = -2.903534 for all i.

    Args:
        x: Positions array of shape (batch_size, d)
        d: Dimension (for documentation, not used)

    Returns:
        Potential values of shape (batch_size,)
    """
    return 0.5 * jnp.sum(x**4 - 16*x**2 + 5*x, axis=-1)


@jax.jit
def aggregation_potential_fn(x: Array, a: int = 4, b: int = 2) -> Array:
    """
    Aggregation potential U(x) = |x|^a/a - |x|^b/b

    With a > b > 0. Creates aggregation behavior.

    Args:
        x: Positions array of shape (batch_size, d)
        a: Power of repulsion term (a > b)
        b: Power of attraction term (b > 0)

    Returns:
        Potential values of shape (batch_size,)
    """
    return jnp.linalg.norm(x, axis=-1)**a / a - jnp.linalg.norm(x, axis=-1)**b / b


@jax.jit
def zero_potential_fn(x: Array) -> Array:
    """
    Zero potential U(x) = 0

    Useful for testing or when no external potential is needed.

    Args:
        x: Positions array of shape (batch_size, d)

    Returns:
        Zero array of shape (batch_size,)
    """
    return jnp.zeros(x.shape[0])


@jax.jit
def harmonic_potential_fn(x: Array, omega: float = 1.0) -> Array:
    """
    Anisotropic harmonic potential U(x) = 0.5 * ((1/2)*x₀² + (1/3)*x₁²)

    Args:
        x: Positions array of shape (batch_size, 2)
        omega: Frequency parameter (currently not used, for API compatibility)

    Returns:
        Potential values of shape (batch_size,)
    """
    return 0.5 * ((1/2) * x[:, 0]**2 + (1/3) * x[:, 1]**2)


# =============================================================================
# Obstacle and Cost Functions (migrated from Torch Density_Path_Opt)
# =============================================================================

@jax.jit
def obstacle_cost_gmm_fn(x: Array, centers: Array = None, radius: float = 1.5) -> Array:
    """
    GMM obstacle cost with three circular obstacles.

    Args:
        x: Positions array of shape (batch_size, 2)
        centers: Obstacle centers, shape (3, 2). Default: [[6,6], [6,-6], [-6,-6]]
        radius: Radius of obstacles. Default: 1.5

    Returns:
        Cost values of shape (batch_size,)

    Reference: GSBM state_cost.py
    """
    if centers is None:
        centers = jnp.array([[6.0, 6.0], [6.0, -6.0], [-6.0, -6.0]])

    # Compute distances to each obstacle
    dist1 = jnp.linalg.norm(x - centers[0], axis=-1)
    dist2 = jnp.linalg.norm(x - centers[1], axis=-1)
    dist3 = jnp.linalg.norm(x - centers[2], axis=-1)

    # Softplus approximation of max(0, radius - dist)
    cost1 = jax.nn.softplus(100 * (radius - dist1))
    cost2 = jax.nn.softplus(100 * (radius - dist2))
    cost3 = jax.nn.softplus(100 * (radius - dist3))

    return (cost1 + cost2 + cost3) * 50


@jax.jit
def obstacle_cost_vneck_fn(x: Array, c_sq: float = 0.36, coef: float = 2.0) -> Array:
    """
    V-neck obstacle cost creating a V-shaped forbidden region.

    Args:
        x: Positions array of shape (batch_size, 2)
        c_sq: Threshold parameter. Default: 0.36
        coef: Coefficient for x-direction. Default: 2.0

    Returns:
        Cost values of shape (batch_size,)

    Reference: GSBM state_cost.py
    """
    assert x.shape[-1] == 2, "V-neck obstacle requires 2D positions"

    x_sq = x**2
    d = coef * x_sq[:, 0] - x_sq[:, 1]

    return jax.nn.softplus(-c_sq - d) * 3000


@jax.jit
def obstacle_cost_stunnel_fn(x: Array, scale: float = 1.0) -> Array:
    """
    S-tunnel obstacle cost with two diagonal elliptical obstacles.

    Creates a challenging S-shaped passage between two diagonal obstacles.

    Args:
        x: Positions array of shape (batch_size, 2) or (batch_size, d) where d >= 2
        scale: Scaling factor for obstacle strength. Default: 1.0

    Returns:
        Cost values of shape (batch_size,)

    Reference: APAC-Net twodiag_cylinder.py
    """
    # Extract first 2 dimensions
    xx = x[:, :2]

    # Rotation matrix (pi/5 radians)
    theta = jnp.pi / 5
    rot_mat = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                         [jnp.sin(theta), jnp.cos(theta)]])

    # Bottom/Left obstacle
    center1 = jnp.array([-2.0, 0.5])
    xxcent1 = xx - center1
    xxcent1_rot = xxcent1 @ rot_mat.T

    covar_mat1 = jnp.array([[5.0, 0.0], [0.0, 0.0]])
    bb_vec1 = jnp.array([0.0, 2.0])

    quad1 = jnp.sum((xxcent1_rot @ covar_mat1) * xxcent1_rot, axis=-1)
    lin1 = jnp.sum(xxcent1_rot * bb_vec1, axis=-1)
    out1 = (-1) * (quad1 + lin1 + 1)
    out1 = scale * jnp.maximum(out1, 0)

    # Top/Right obstacle
    center2 = jnp.array([2.0, -0.5])
    xxcent2 = xx - center2
    xxcent2_rot = xxcent2 @ rot_mat.T

    covar_mat2 = jnp.array([[5.0, 0.0], [0.0, 0.0]])
    bb_vec2 = jnp.array([0.0, -2.0])

    quad2 = jnp.sum((xxcent2_rot @ covar_mat2) * xxcent2_rot, axis=-1)
    lin2 = jnp.sum(xxcent2_rot * bb_vec2, axis=-1)
    out2 = (-1) * (quad2 + lin2 + 1)
    out2 = scale * jnp.maximum(out2, 0)

    return (out1 + out2) * 100


@jax.jit
def gaussian_well_potential_fn(x: Array,
                                center1: Array = None,
                                center2: Array = None,
                                center3: Array = None,
                                sigma: float = 0.5,
                                depth: float = 1000.0) -> Array:
    """
    Multiple Gaussian wells potential.

    Creates attractive potential wells using Gaussian functions. Default creates
    three wells suitable for 2D problems.

    Args:
        x: Positions array of shape (batch_size, d)
        center1: Center of first well. Default: [0, 1.5] (padded with zeros for d>2)
        center2: Center of second well. Default: [0, -1.5] (padded with zeros for d>2)
        center3: Center of third well. Default: [0, 0] (padded with zeros for d>2)
        sigma: Width of the wells. Default: 0.5
        depth: Depth (magnitude) of the wells. Default: 1000.0

    Returns:
        Potential values of shape (batch_size,)
    """
    d = x.shape[-1]

    # Set default centers based on dimension
    if center1 is None:
        if d >= 2:
            center1 = jnp.array([0.0, 1.5])
        else:
            center1 = jnp.array([3.0])

    if center2 is None:
        if d >= 2:
            center2 = jnp.array([0.0, -1.5])
        else:
            center2 = jnp.array([-3.0])

    if center3 is None:
        center3 = jnp.zeros(min(d, 2))

    # Pad centers to match dimension
    if d > len(center1):
        center1 = jnp.concatenate([center1, jnp.zeros(d - len(center1))])
    if d > len(center2):
        center2 = jnp.concatenate([center2, jnp.zeros(d - len(center2))])
    if d > len(center3):
        center3 = jnp.concatenate([center3, jnp.zeros(d - len(center3))])

    # Calculate squared distances
    dist1_sq = jnp.sum((x - center1)**2, axis=-1)
    dist2_sq = jnp.sum((x - center2)**2, axis=-1)
    dist3_sq = jnp.sum((x - center3)**2, axis=-1)

    # Gaussian wells (negative for attraction)
    well1 = -depth * jnp.exp(-dist1_sq / (2 * sigma**2))
    well2 = -depth * jnp.exp(-dist2_sq / (2 * sigma**2))
    well3 = -depth * jnp.exp(-dist3_sq / (2 * (sigma / 4)**2))

    return well1 + well2 + well3


# =============================================================================
# Interaction Kernel Functions
# =============================================================================

@jax.jit
def congestion_kernel_fn(r: Array, strength: float = 2.0, eps: float = 1.0) -> Array:
    """
    Congestion/repulsion interaction kernel based on inverse distance.

    This kernel creates repulsive forces between particles, useful for
    modeling congestion or particle repulsion in density path problems.

    Args:
        r: Relative positions (x - y), shape (batch_size, d)
        strength: Strength of repulsion. Default: 2.0
        eps: Regularization to avoid singularity. Default: 1.0

    Returns:
        Interaction values of shape (batch_size,)

    Reference: Density_Path_Opt obstacles.py congestion_cost
    """
    dist_sq = jnp.sum(r**2, axis=-1)
    return strength / (dist_sq + eps)


@jax.jit
def gaussian_kernel_fn(r: Array, sigma: float = 1.0) -> Array:
    """
    Gaussian interaction kernel W(r) = exp(-|r|²/(2σ²))

    Commonly used for soft repulsive/attractive interactions.

    Args:
        r: Relative positions (x - y), shape (batch_size, d)
        sigma: Width of the Gaussian. Default: 1.0

    Returns:
        Interaction values of shape (batch_size,)
    """
    dist_sq = jnp.sum(r**2, axis=-1)
    return jnp.exp(-dist_sq / (2 * sigma**2))


@jax.jit
def coulomb_kernel_fn(r: Array, eps: float = 0.1) -> Array:
    """
    Coulomb/electrostatic interaction kernel W(r) = 1/|r|

    Models long-range repulsive interactions.

    Args:
        r: Relative positions (x - y), shape (batch_size, d)
        eps: Regularization to avoid singularity. Default: 0.1

    Returns:
        Interaction values of shape (batch_size,)
    """
    dist = jnp.linalg.norm(r, axis=-1)
    return 1.0 / (dist + eps)


@jax.jit
def power_law_kernel_fn(r: Array, exponent: float = 2.0, eps: float = 0.1) -> Array:
    """
    Power-law interaction kernel W(r) = 1/|r|^α

    Args:
        r: Relative positions (x - y), shape (batch_size, d)
        exponent: Power law exponent. Default: 2.0
        eps: Regularization to avoid singularity. Default: 0.1

    Returns:
        Interaction values of shape (batch_size,)
    """
    dist = jnp.linalg.norm(r, axis=-1)
    return 1.0 / ((dist + eps)**exponent)


# =============================================================================
# Factory Function
# =============================================================================

def create_potentials():
    """
    Factory function to create common LinearPotential instances.

    Returns:
        Dictionary mapping potential names to LinearPotential instances
    """
    # Basic potentials
    quadratic_potential = LinearPotential(quadratic_potential_fn)
    double_well_potential = LinearPotential(double_well_potential_fn, alpha=1.0)
    strong_double_well = LinearPotential(double_well_potential_fn, alpha=5.0)
    four_well_potential = LinearPotential(four_well_potential_fn, alpha=1.0)
    strong_four_well = LinearPotential(four_well_potential_fn, alpha=5.0)
    quartic_potential = LinearPotential(quartic_potential_fn, strength=0.1)
    styblinski_tang_potential = LinearPotential(styblinski_tang_potential_fn, d=2)
    aggregation_potential = LinearPotential(aggregation_potential_fn, a=4, b=2)
    zero_potential = LinearPotential(zero_potential_fn)
    harmonic_potential = LinearPotential(harmonic_potential_fn, omega=1.0)

    # Obstacle potentials (from Density_Path_Opt)
    obstacle_gmm = LinearPotential(obstacle_cost_gmm_fn, radius=1.5)
    obstacle_vneck = LinearPotential(obstacle_cost_vneck_fn, c_sq=0.36, coef=2.0)
    obstacle_stunnel = LinearPotential(obstacle_cost_stunnel_fn, scale=1.0)
    gaussian_wells = LinearPotential(gaussian_well_potential_fn, sigma=0.5, depth=1000.0)

    return {
        # Basic potentials
        'quadratic': quadratic_potential,
        'double_well': double_well_potential,
        'strong_double_well': strong_double_well,
        'four_well': four_well_potential,
        'strong_four_well': strong_four_well,
        'quartic': quartic_potential,
        'styblinski_tang': styblinski_tang_potential,
        'aggregation': aggregation_potential,
        'zero': zero_potential,
        'harmonic': harmonic_potential,
        # Obstacle potentials
        'obstacle_gmm': obstacle_gmm,
        'obstacle_vneck': obstacle_vneck,
        'obstacle_stunnel': obstacle_stunnel,
        'gaussian_wells': gaussian_wells,
    }


def create_interaction_kernels():
    """
    Factory function to create common interaction kernel functions.

    These kernels are designed to work with InteractionPotential class.
    They take relative positions r = x - y and return interaction strength.

    Returns:
        Dictionary mapping kernel names to kernel functions

    Usage:
        >>> from pdpo.energy_model import create_interaction_kernels, InteractionPotential
        >>> kernels = create_interaction_kernels()
        >>> congestion_potential = InteractionPotential(
        ...     interaction_fn=kernels['congestion'],
        ...     coeff=1.0
        ... )
    """
    return {
        'congestion': congestion_kernel_fn,
        'gaussian': gaussian_kernel_fn,
        'coulomb': coulomb_kernel_fn,
        'power_law': power_law_kernel_fn,
    }
