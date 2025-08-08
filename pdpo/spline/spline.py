from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Callable, NamedTuple
import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import lax
from jaxtyping import Array, Float, PyTree
import optax
from flax import nnx

from pdpo.spline.interpolation import linear_interpolation_states, cubic_interp, linear_interp
from pdpo.models.nn import create_mlp
from pdpo.ode.solvers import ODESolver, EulerSolver, MidpointSolver
from pdpo.core.types import (
    SampleArray, TrajectoryArray, TimeStepsArray, EnergyArray,
    ScalarArray, PRNGKeyArray
)


from pdpo.spline.interpolation import linear_interpolation_states, cubic_interp, linear_interp
from pdpo.spline.types_interpolation import SplineState, SplineConfig,ProblemConfig
from pdpo.data.toy_datasets import inf_train_gen


