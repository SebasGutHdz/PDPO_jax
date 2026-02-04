"""
ODE integration and neural ODE implementations.

This module provides:
- ODE solvers (Euler, Heun, Midpoint)
- Neural ODE class with log-likelihood and score computation
- Adjoint method for efficient gradient computation
- Utility functions for divergence and jacobian calculations
"""

from pdpo.ode.solvers import (
    ODESolver,
    EulerSolver,
    HeunSolver,
    MidpointSolver,
    string_2_solver,
    eval_model,
    sample_trajectory
)

from pdpo.ode.neural_ode import NeuralODE

from pdpo.ode.log_ode_utils import (
    divergence_vf,
    divergence_vf_hutch,
    jacobian_vf,
    compute_jacobian_and_grad_div
)

from pdpo.ode.adjoint import (
    AdjointODESolver,
    odeint_adjoint,
    adjoint_neural_ode,
    checkpointed_odeint,
    flatten_pytree
)

__all__ = [
    # Solvers
    'ODESolver',
    'EulerSolver',
    'HeunSolver',
    'MidpointSolver',
    'string_2_solver',
    'eval_model',
    'sample_trajectory',
    # Neural ODE
    'NeuralODE',
    # Adjoint methods
    'AdjointODESolver',
    'odeint_adjoint',
    'adjoint_neural_ode',
    'checkpointed_odeint',
    'flatten_pytree',
    # Utilities
    'divergence_vf',
    'divergence_vf_hutch',
    'jacobian_vf',
    'compute_jacobian_and_grad_div',
]
