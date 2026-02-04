"""
Neural ODE implementation with support for log-likelihood and score function computation.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from jaxtyping import PyTree, Array
from flax import nnx
import jax.scipy.stats as stats

from pdpo.core.types import SampleArray, TimeArray, VelocityArray, TrajectoryArray
from pdpo.ode.solvers import string_2_solver, eval_model
from pdpo.ode.log_ode_utils import (
    divergence_vf,
    divergence_vf_hutch,
    jacobian_vf,
    compute_jacobian_and_grad_div
)


class NeuralODE(nnx.Module):
    """
    Neural ODE class with support for:
    - Forward and backward integration
    - Log-likelihood computation via continuous normalizing flows
    - Score function computation via adjoint methods
    - Push-forward and pull-back operations
    """

    def __init__(self,
                 dynamics_model: nnx.Module,
                 time_dependent: bool = False,
                 solver: str = "euler",
                 dt0: float = 0.1,
                 rtol: float = 1e-4,
                 atol: float = 1e-6):
        """
        Initialize Neural ODE.

        Args:
            dynamics_model: Neural network defining the dynamics
            time_dependent: Whether the dynamics depend on time
            solver: ODE solver to use ('euler', 'heun', 'midpoint')
            dt0: Initial time step size
            rtol: Relative tolerance (for adaptive solvers)
            atol: Absolute tolerance (for adaptive solvers)
        """
        self.dynamics = dynamics_model
        self.solver = string_2_solver(solver)
        self.dt0 = dt0
        self.rtol = rtol
        self.atol = atol
        self.time_dependent = time_dependent

    def vector_field(self, t: TimeArray, y: SampleArray, args: Optional[dict] = None) -> SampleArray:
        """
        Define the vector field function.

        Args:
            t: Time values
            y: State values
            args: Optional arguments

        Returns:
            Velocity at (t, y)
        """
        if self.time_dependent:
            return eval_model(self.dynamics, t, y, time_dependent=True)
        return self.dynamics(y)

    def log_likelihood(self,
                      t: TimeArray,
                      xt: TrajectoryArray,
                      log_prob_init: Optional[Array] = None,
                      method: str = 'exact',
                      params: Optional[PyTree] = None,
                      log_trajectory: bool = False) -> Array:
        """
        Solve ODE for log-likelihood using continuous normalizing flows.

        Args:
            t: Time array of shape (time_steps,)
            xt: Sample array of shape (batch_size, time_steps, dim)
            log_prob_init: Initial log-likelihood at t=0, shape (batch_size,)
            method: Method to compute log-likelihood ('exact' or 'hutchinson')
            params: Optional PyTree of parameters for the dynamics model
            log_trajectory: If True, return full log-likelihood trajectory

        Returns:
            log_likelihood: Log-likelihood at (t,x), shape (batch_size,) or (time_steps, batch_size)
        """
        if log_prob_init is None:
            log_prob_init = stats.multivariate_normal.logpdf(
                xt[:, 0, :],
                mean=jnp.zeros(xt.shape[-1]),
                cov=jnp.eye(xt.shape[-1])
            )

        solution_history = [log_prob_init]

        for i in range(len(t) - 1):
            # Expand t to match batch size
            t_reshape = t[i + 1] * jnp.ones(xt[:, i + 1, :].shape[0])  # Shape (batch_size,)

            # ODE RHS is negative divergence of the vector field
            log_rhs = lambda time, logp: -self.divergence(
                t=t_reshape,
                x=xt[:, i + 1, :],
                method=method,
                params=params
            )

            log_new = self.solver.step(log_rhs, t, i, solution_history)
            solution_history.append(log_new)

        if log_trajectory:
            return jnp.array(solution_history)
        else:
            return solution_history[-1]

    def score_function(self,
                      t: TimeArray,
                      xt: TrajectoryArray,
                      score_init: Optional[Array] = None,
                      method: str = 'exact',
                      params: Optional[PyTree] = None,
                      score_trajectory: bool = False):
        """
        Solve ODE for score function ∇log ρ.

        Args:
            t: Time array of shape (time_steps,)
            xt: Sample array of shape (batch_size, time_steps, dim)
            score_init: Initial score at t=0, shape (batch_size, dim)
            method: 'exact' or 'autodiff'
            params: Optional PyTree of parameters for the dynamics model
            score_trajectory: If True, return full score trajectory

        Returns:
            score: Score function at (t,x), shape (batch_size, dim)
        """
        if method not in ['exact', 'autodiff']:
            raise ValueError(f'Method {method} not recognized. Available: exact, autodiff.')

        if method == "exact":
            if score_init is None:
                score_init = -xt[:, 0, :]  # Score of standard normal

            solution_history = [score_init]

            for i in range(len(t) - 1):
                # Expand t to match batch size
                t_reshape = t[i] * jnp.ones(xt[:, i, :].shape[0])

                # Compute jacobian and gradient of divergence
                jacobian, grad_div = self.jacobian_grad_and_div(
                    t=t_reshape,
                    x=xt[:, i, :],
                    method=method,
                    params=params
                )

                # Score ODE: ds/dt = -J^T s - ∇(div v)
                score_rhs = lambda time, score: -jnp.einsum('bji,bj->bi', jacobian, score) - grad_div
                score_new = self.solver.step(score_rhs, t, i, solution_history)
                solution_history.append(score_new)

            if score_trajectory:
                return jnp.array(solution_history)
            else:
                return solution_history[-1]

        elif method == "autodiff":
            # Autodiff method: differentiate through the entire flow
            def log_push_pull(x):
                if len(x.shape) == 1:
                    x = x[jnp.newaxis, :]

                # Pull back to reference
                backwards_traj, _ = self.pull_back(x, params=params, history=True)

                # Reverse for forward pass
                fwd_traj = backwards_traj[:, ::-1, :]

                # Compute log-likelihood
                log_px = self.log_likelihood(
                    t, fwd_traj,
                    log_prob_init=None,
                    method='exact',
                    params=params,
                    log_trajectory=False
                )

                return log_px[0]

            # Compute score via gradient of log-likelihood
            score = jax.vmap(jax.grad(log_push_pull))(xt[:, -1, :].copy())

            if score_trajectory:
                raise NotImplementedError("Score trajectory not implemented for autodiff method.")
            else:
                return score

    def __call__(self,
                 y0: Array,
                 t_span: Optional[Tuple[float, float]] = (0.0, 1.0),
                 params: Optional[PyTree] = None,
                 history: bool = False) -> Array:
        """
        Solve the ODE from t_span[0] to t_span[1] with initial condition y0.

        Args:
            y0: Initial condition, shape (batch_size, dim) or (dim,)
            t_span: Tuple of (t0, t1) for integration bounds
            params: Optional parameters for the dynamics model
            history: If True, return full trajectory

        Returns:
            Final state at t1 (or full trajectory if history=True)
        """
        if params is None:
            model = self.dynamics
        else:
            graphdef, _ = nnx.split(self.dynamics)
            model = nnx.merge(graphdef, params)

        def vector_field(t: float, y: Array, args: Optional[dict] = None):
            return eval_model(model, t, y, time_dependent=self.time_dependent)

        if t_span[0] < t_span[1]:
            t_list = jnp.arange(t_span[0], t_span[1] + self.dt0, self.dt0)
        else:
            t_list = jnp.arange(t_span[0], t_span[1] - self.dt0, -self.dt0)

        y = self.solver(vector_field, t_list, y0, history=history)

        if history:
            return y, t_list
        return y.reshape(-1, y0.shape[-1])

    def divergence(self,
                  t: TimeArray,
                  x: SampleArray,
                  method: str = "exact",
                  params: Optional[PyTree] = None) -> Array:
        """
        Compute divergence of the vector field.

        Args:
            t: Time values
            x: Sample positions
            method: 'exact' or 'hutchinson'
            params: Optional parameters

        Returns:
            Divergence values
        """
        if params is None:
            model = self.dynamics
        else:
            graphdef, _ = nnx.split(self.dynamics)
            model = nnx.merge(graphdef, params)

        if method == "exact":
            return divergence_vf(model, t, x, self.time_dependent)
        elif method == "hutchinson":
            return divergence_vf_hutch(model, t, x, self.time_dependent, num_samples=50)

    def jacobian_grad_and_div(self,
                             t: TimeArray,
                             x: SampleArray,
                             method: str = "exact",
                             params: Optional[PyTree] = None) -> Array:
        """
        Compute jacobian and gradient of divergence.

        Args:
            t: Time values
            x: Sample positions
            method: Computation method
            params: Optional parameters

        Returns:
            (jacobian, grad_div) tuple
        """
        if params is None:
            model = self.dynamics
        else:
            graphdef, _ = nnx.split(self.dynamics)
            model = nnx.merge(graphdef, params)

        return compute_jacobian_and_grad_div(model, t, x, self.time_dependent)

    def push_forward(self,
                    z: SampleArray,
                    params: Optional[PyTree] = None,
                    history: bool = False) -> SampleArray:
        """
        Push forward samples z through the Neural ODE to obtain x.

        Args:
            z: Reference samples, shape (batch_size, dim)
            params: Parameters for the dynamics model
            history: If True, return full trajectory

        Returns:
            x: Transformed samples
        """
        if params is None:
            _, params = nnx.split(self.dynamics)
        return self.__call__(z, params=params, history=history)

    def pull_back(self,
                 x: SampleArray,
                 params: Optional[PyTree] = None,
                 history: bool = False) -> SampleArray:
        """
        Pull back samples x through the Neural ODE to obtain z.

        Args:
            x: Target samples, shape (batch_size, dim)
            params: Parameters for the dynamics model
            history: If True, return full trajectory

        Returns:
            z: Reference samples
        """
        if params is None:
            _, params = nnx.split(self.dynamics)

        return self.__call__(x, t_span=(1.0, 0.0), params=params, history=history)
