"""
Utility functions for computing divergence, jacobian, and related quantities
for neural ODEs. These are used for log-likelihood and score function computations.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from pdpo.core.types import TimeArray, SampleArray, PRNGKeyArray
from pdpo.ode.solvers import eval_model


def divergence_vf(
        vf_model: nnx.Module,
        t: TimeArray,
        x: SampleArray,
        time_dependent: bool = False
) -> Array:
    """
    Compute the divergence of the vector field with respect to x.

    Args:
        vf_model: The vector field model (nnx.Module)
        t: Time array of shape (batch_size,)
        x: Sample array of shape (batch_size, dim)
        time_dependent: Whether the vector field is time-dependent

    Returns:
        div: Divergence of the vector field at (t,x), shape (batch_size,)
    """

    def velocity_field(t, x):
        return eval_model(vf_model, t, x, time_dependent=time_dependent)

    # Compute the divergence using jax.jacfwd
    def compute_divergence(t_single, x_single):
        jacobian = jax.jacfwd(velocity_field, argnums=1)(t_single, x_single[None, :])
        return jnp.trace(jacobian[0].squeeze())

    # Vectorize over the batch dimension
    div = jax.vmap(compute_divergence)(t, x)
    return div


def divergence_vf_hutch(
    vf_model: nnx.Module,
    t: TimeArray,
    x: SampleArray,
    time_dependent: bool = False,
    num_samples: int = 1000
) -> Array:
    """
    Compute the divergence of the vector field using Hutchinson's estimator.

    Args:
        vf_model: The vector field model
        t: Time array of shape (batch_size,)
        x: Sample array of shape (batch_size, dim)
        time_dependent: Whether the vector field is time-dependent
        num_samples: Number of random samples for Hutchinson estimator

    Returns:
        div: Divergence of the vector field at (t,x), shape (batch_size,)
    """
    batch_size, dim = x.shape

    def compute_divergence_hutchinson_single(t_single: Array, x_single: Array, key: PRNGKeyArray) -> Array:
        """
        Compute divergence using Hutchinson's estimator for a single sample.
        """
        # Define velocity field with fixed time parameter
        def velocity_field_fixed_t(x: Array) -> Array:
            return eval_model(vf_model, t_single, x, time_dependent=time_dependent)

        def jvp_fn(x_single: Array, v: Array) -> Array:
            """Compute JVP of velocity field w.r.t. x at x_single in direction v"""
            _, jvp = jax.jvp(velocity_field_fixed_t, (x_single[None, :],), (v[None, :],))
            return jvp[0]

        # Generate keys for random vectors
        keys = jax.random.split(key, num_samples)

        # Per key, generate the rademacher vector and compute the jvp
        def single_estimate(key):
            # Generate random Rademacher vector
            v = jax.random.rademacher(key, shape=(dim,), dtype=jnp.float32)
            jvp_result = jvp_fn(x_single, v)
            return jnp.dot(jvp_result, v)

        # Average over multiple random vectors
        estimates = jax.vmap(single_estimate)(keys)
        return jnp.mean(estimates)

    # Generate keys for each batch element
    key = jax.random.PRNGKey(0)
    batch_keys = jax.random.split(key, batch_size)

    # Vectorize over the batch dimension
    div = jax.vmap(compute_divergence_hutchinson_single)(t, x, batch_keys)
    return div


def jacobian_vf(
        vf_model: nnx.Module,
        t: TimeArray,
        x: SampleArray,
        time_dependent: bool = False
) -> Array:
    """
    Compute the Jacobian of the vector field with respect to x.

    Args:
        vf_model: The vector field model
        t: Time array of shape (batch_size,)
        x: Sample array of shape (batch_size, dim)
        time_dependent: Whether the vector field is time-dependent

    Returns:
        jac: Jacobian of the vector field at (t,x), shape (batch_size, dim, dim)
    """

    def velocity_field(t, x):
        return eval_model(vf_model, t, x, time_dependent=time_dependent)

    # Compute the Jacobian using jax.jacfwd
    def compute_jacobian(t_single, x_single):
        jacobian = jax.jacfwd(velocity_field, argnums=1)(t_single, x_single[None, :])
        return jacobian[0].squeeze()

    # Vectorize over the batch dimension
    jac = jax.vmap(compute_jacobian)(t, x)
    return jac


def compute_jacobian_and_grad_div(
    vf_model: nnx.Module,
    t: TimeArray,
    x: SampleArray,
    time_dependent: bool = False
) -> tuple[Array, Array]:
    """
    Compute both jacobian and gradient of divergence in single pass.

    Args:
        vf_model: The vector field model
        t: Time array of shape (batch_size,)
        x: Sample array of shape (batch_size, dim)
        time_dependent: Whether the vector field is time-dependent

    Returns:
        jacobian: Jacobian matrix, shape (batch_size, dim, dim)
        grad_div: Gradient of divergence, shape (batch_size, dim)
    """

    def forward_fn(x_pos):
        jac_batch = jacobian_vf(vf_model, t, x_pos, time_dependent=time_dependent)  # (batch_size, dim, dim)
        div_sum = jnp.sum(jnp.trace(jac_batch, axis1=1, axis2=2))
        return div_sum, jac_batch  # (value, auxiliary)

    # Use jax.value_and_grad with has_aux=True
    (div_sum, jacobian), grad_div = jax.value_and_grad(
        forward_fn,
        has_aux=True
    )(x)

    return jacobian, grad_div
