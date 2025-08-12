
from typing import List, Optional, Tuple, Callable
import jax
from functools import partial
import jax.numpy as jnp
from jax import lax


from jaxtyping import Array, PyTree


from pdpo.spline.interpolation import linear_interpolation_states, cubic_interp, linear_interp
from pdpo.spline.types_interpolation import  ProblemConfig
from pdpo.core.types import (
 TrajectoryArray, TimeStepsArray, EnergyArray,
    ScalarArray
)
from pdpo.energy_model.potentials.obstacles import potential_dispatch


def potential_energy(samples_path: TrajectoryArray
                     ,fn_ids: Array, coeffs: Array, key = None) -> EnergyArray:
    """
    Args:
        samples_path: array (batch, time_steps, dim)
        fn_ids: integer array (num_potentials,)
        coeffs: float array (num_potentials,)
    
    Returns:
        potential_energy: array (batch, time_steps)
    
    """

    def _single_term(idx, coeff):
        return potential_dispatch(samples_path, idx, key=key) * coeff

    pes = jax.vmap(_single_term)(fn_ids, coeffs)

    # print(pes.shape)

    return pes.sum(axis=0).mean(axis=0)