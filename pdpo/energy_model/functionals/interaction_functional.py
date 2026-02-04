"""
Interaction potential functional class for particle interactions.
"""

from typing import Optional, Union, Callable
from jaxtyping import PyTree, Array
import jax
import jax.numpy as jnp
from flax import nnx
import jax.scipy.stats as stats


class InteractionPotential:
    """
    A class for handling interaction energy functionals:
    F(ρ) = 0.5 ∫∫ W(x-y)ρ(x)ρ(y)dxdy

    where W is a user-defined interaction function.
    """

    def __init__(self,
                 interaction_fn: Union[str, Callable[[Array, Array], Array]],
                 coeff: float = 1.0,
                 **interaction_kwargs):
        """
        Initialize InteractionPotential with an interaction function.

        Args:
            interaction_fn: Function that takes positions (batch_size, d) and (batch_size, d)
                           and returns interaction values (batch_size,).
                           Can also be a string like 'gaussian' or 'coulomb'.
            coeff: Coefficient for the functional
            **interaction_kwargs: Additional keyword arguments for the interaction function
        """
        self.interaction_fn = interaction_fn
        self.interaction_kwargs = interaction_kwargs
        self.coeff = coeff

    def __call__(self, x: Array, y: Array) -> Array:
        """
        Evaluate interaction function at given positions.

        The functionals are assumed to be shift-invariant: W(x,y) = W(x-y).

        Args:
            x: Positions array of shape (batch_size, d)
            y: Positions array of shape (batch_size, d)

        Returns:
            Interaction values of shape (batch_size,)
        """
        z = x - y  # (batch_size, d)
        return self.interaction_fn(z, **self.interaction_kwargs)

    def compute_energy_gradient(self,
                                parametric_model,
                                z_samples: Array,
                                params: Optional[PyTree] = None) -> PyTree:
        """
        Compute gradient of energy functional w.r.t. model parameters.

        Args:
            parametric_model: Model instance (NeuralODE or similar)
            z_samples: Reference samples (batch_size, d)
            params: Parameters to evaluate gradient at

        Returns:
            (grad, energy_value) tuple
        """
        if params is None:
            _, params = nnx.split(parametric_model)

        def energy_functional(p: PyTree) -> Array:
            # Split z-samples in half for x and y
            batch_size = z_samples.shape[0]
            mid_point = batch_size // 2
            x_samples = parametric_model(z_samples[:mid_point, :], params=p)
            y_samples = parametric_model(z_samples[mid_point:, :], params=p)

            # Ensure equal lengths
            if len(x_samples) < len(y_samples):
                y_samples = y_samples[:len(x_samples), :]
            elif len(y_samples) < len(x_samples):
                x_samples = x_samples[:len(y_samples), :]

            # Evaluate interaction potential
            potential_vals = self(x_samples, y_samples)
            energy = 0.5 * jnp.mean(potential_vals)
            return energy

        values, grad = jax.value_and_grad(energy_functional)(params)

        return grad, values

    def evaluate_energy(self,
                       parametric_model,
                       z_samples: Array,
                       x_samples: Optional[Array] = None,
                       y_samples: Optional[Array] = None,
                       params: Optional[PyTree] = None) -> tuple[Array, Array]:
        """
        Evaluate current energy F(ρ_θ).

        Args:
            parametric_model: Model instance
            z_samples: Reference samples
            x_samples: Pre-computed pushforward samples (optional)
            y_samples: Pre-computed pushforward samples (optional)
            params: Model parameters

        Returns:
            (energy, x_samples) tuple
        """
        if params is None:
            _, params = nnx.split(parametric_model)

        if x_samples is None or y_samples is None:
            # Split z-samples in half for x and y
            batch_size = z_samples.shape[0]
            mid_point = batch_size // 2
            x_samples = parametric_model(z_samples[:mid_point, :], params=params)
            y_samples = parametric_model(z_samples[mid_point:, :], params=params)

            # Ensure equal lengths
            if len(x_samples) < len(y_samples):
                y_samples = y_samples[:len(x_samples), :]
            elif len(y_samples) < len(x_samples):
                x_samples = x_samples[:len(y_samples), :]

        potential_vals = self(x_samples, y_samples)
        energy = 0.5 * jnp.mean(potential_vals)

        return energy, x_samples
