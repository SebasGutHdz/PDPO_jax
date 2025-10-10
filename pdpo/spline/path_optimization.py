
from typing import  Optional, Tuple
from dataclasses import replace
import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import lax
import optax


from pdpo.spline.types_interpolation import  ProblemConfig
from pdpo.data.toy_datasets import inf_train_gen
# from pdpo.spline.energy import kinetic_energy, potential_energy, lagrangian
from pdpo.energy_model.lagrangian import lagrangian
from pdpo.spline.dynamics import gen_sample_trajectory
from pdpo.core.types import (
  PRNGKeyArray, SampleArray
)
from pdpo.spline.types_interpolation import SplineState,OptimizationHistory,ProblemConfig
from pdpo.models.builder import create_model

# =============================================================================
# Optimization Functions
# =============================================================================





def optimize_path(
    problem_config: ProblemConfig,
    key: PRNGKeyArray,
    epochs: int,
    learning_rate: float = 0,
    t_node: int = 10,
    batch_size: int = 1000,
    x0: Optional[SampleArray] = None
) -> Tuple[SplineState, OptimizationHistory]:
    """
    Optimizes the interior points of the spline path while keeping endpoints fixed.
    This version uses jax.jit + jax.lax.scan to run the training loop inside XLA.
    """

    spline_state = problem_config.splinestate

    # Precompute constants
    t_partition = problem_config.discretization_integral
    t_traj = jnp.linspace(0.0, 1.0, t_partition)

    # Build model once (assumed pure / functional in evaluation)
    arch = spline_state.config.architecture + [key]
    key, subkey = jrn.split(key)
    vf = create_model(
        type=spline_state.config.type_architecture,
        args_arch=arch
    )

    # Sample initial x0 once (if user didn't provide one). If you want fresh samples each epoch,
    # we can sample inside the loss using the per-step RNG key.
    if x0 is None:
        x0 = inf_train_gen(
            data_type=spline_state.prior,
            key=key,
            batch_size=batch_size,
            dim=spline_state.config.architecture[0]
        )
        key, subkey = jrn.split(key)
    control_points = spline_state.control_points
    control_points = jax.device_put(control_points, device=jax.devices("gpu")[0])

    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(control_points)
    init_control_points = control_points

    # Pre-split keys for each epoch to avoid splitting inside scan (stable trace)
    keys = jrn.split(key, epochs + 1)[1:]  # length == epochs

    
    def loss_fn(control_points, step_key):
        """Given control points and a PRNG key, produce scalar loss and aux (ke, pe)."""
        # Replace control points in a copy of the spline state
        temp_state = replace(spline_state, control_points=control_points)

        # Generate sample trajectory 
        samples_path = gen_sample_trajectory(
            temp_state,
            vf,
            key=step_key,
            x0=x0,
            num_samples=batch_size,
            t_traj=t_traj,
            time_steps_node=t_node,
            solver=spline_state.config.solver
        )

        total_cost, ke, pe = lagrangian(samples_path, t_traj, problem_config,key = subkey)
         
        return total_cost, (ke, pe)

    # JIT compile loss+grad to avoid repeated retracing
    # loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    # Per-step function for lax.scan
    def train_step(carry, step_key):
        """
        carry: (control_points, opt_state)
        step_key: PRNGKey for this step
        returns: (new_carry, metrics)
        """
        control_points, opt_state = carry

        # compute loss and grads
        (loss_val, (ke_val, pe_val)), grads = loss_and_grad(control_points, step_key)

        # optimizer update
        updates, opt_state = optimizer.update(grads, opt_state, params=control_points)
        control_points = optax.apply_updates(control_points, updates)

        new_carry = (control_points, opt_state)
        metrics = jnp.array([loss_val, ke_val, pe_val])
        return new_carry, metrics

    # Run scan over pre-split keys
    init_carry = (init_control_points, opt_state)
    # This line of code is effectively doing the for loop. 
    (final_control_points, final_opt_state), metrics = lax.scan(
        f = train_step,
        init = init_carry,
        xs = keys,
        length=epochs
    )

    # metrics shape: (epochs, 3)
    metrics = jnp.asarray(metrics)  # ensure array
    loss_hist = metrics[:, 0].tolist()
    ke_hist = metrics[:, 1].tolist()
    pe_hist = metrics[:, 2].tolist()
    iter_hist = list(range(epochs))

    optimized_state = replace(spline_state, control_points=final_control_points)

    history = OptimizationHistory(
        lagrangian=loss_hist,
        kinetic=ke_hist,
        potential=pe_hist,
        iterations=iter_hist
    )

    return optimized_state,final_opt_state, history















# =============================================================================
# Additional Helper Functions
# =============================================================================

def geodesic_warmup(
    problem_config : ProblemConfig,
    key: PRNGKeyArray,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 1000,
    x0: Optional[SampleArray] = None
) -> SplineState:
    """
    Initializes control points by optimizing for geodesic path in Wasserstein space.
    
    Args:
        spline_state: Initial spline state
        key: JAX random key
        num_epochs: Number of warmup iterations
        learning_rate: Learning rate for warmup
        batch_size: Batch size for sampling
        
    Returns:
        Warmed-up spline state
    """
    # Call of optimize path with spline_state being the zero potential 
    # Replace problem_configsL potential function, entropy, fisher and ke_modifier
    if problem_config.potential is not None:    
        problem_config = replace(problem_config, potential=None)
    if problem_config.entropy > 0:
        problem_config = replace(problem_config, entropy=0.0)
    if problem_config.fisher > 0:
        problem_config = replace(problem_config, fisher=0.0)
    if problem_config.ke_modifier is not None:
        problem_config = replace(problem_config, ke_modifier=None)

    # Now problem_config is fully updated, we can use it for optimization
    optimized_state, final_opt_state, history = optimize_path(
        problem_config=problem_config,
        key=key,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        x0=x0
    )

    return optimized_state, history