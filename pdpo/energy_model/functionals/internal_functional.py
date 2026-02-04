"""
Internal potential functional class for entropy and Fisher information.
"""

from typing import Optional, Union
from jaxtyping import PyTree, Array
import jax
import jax.numpy as jnp
from flax import nnx
import jax.scipy.stats as stats


class InternalPotential:
    """
    A class for handling internal energy functionals F(ρ) = ∫ f(ρ(x))dx.

    Currently supports:
    - Entropy: -∫ ρ(x) log ρ(x) dx
    - Fisher information: ∫ ||∇ρ(x)||²/ρ(x) dx

    These quantities are computed by solving ODEs along the flow.
    """

    def __init__(self,
                 functional: Union[str, Array] = 'entropy',
                 coeff: float = 1.0,
                 sigma: Optional[float] = 1.0,
                 method: str = 'exact',
                 prob_dim=None):
        """
        Initialize InternalPotential.

        Args:
            functional: Type of functional ('entropy', 'fisher', or list of both)
            coeff: Coefficient for the functional
            sigma: Standard deviation for Fisher information
            method: Computation method ('exact' or 'hutchinson' for entropy,
                   'exact' or 'autodiff' for fisher)
            prob_dim: Dimension of the probability space
        """
        if type(functional) == str:
            functional = [functional]

        for func in functional:
            if func not in ['entropy', 'fisher']:
                raise ValueError("Unsupported functional type. Choose 'entropy' or 'fisher'.")
            if func == 'entropy' and method not in ['exact', 'hutchinson']:
                raise ValueError("Unsupported method for entropy. Choose 'exact' or 'hutchinson'.")
            if func == 'fisher' and method not in ['exact', 'autodiff']:
                raise ValueError("Unsupported method for fisher. Choose 'exact' or 'autodiff'.")

        if coeff <= 0:
            raise ValueError("Coefficient must be positive.")

        self.functional = functional
        self.coeff = coeff
        self.sigma = sigma
        self.prob_dim = prob_dim
        self.method = method

    def __call__(self,
                 parametric_model,
                 z_samples: Array,
                 z_trajectory: Optional[Array] = None,
                 time_steps: Optional[Array] = None,
                 params: Optional[PyTree] = None) -> Array:
        """
        Evaluate the internal potential.

        Args:
            parametric_model: Model instance (NeuralODE or similar)
            z_samples: Reference samples (batch_size, d)
            z_trajectory: Pre-computed trajectory (if available)
            time_steps: Pre-computed time steps (if available)
            params: Model parameters

        Returns:
            internal_energy: Scalar energy value
        """
        if params is not None:
            graphdef, _ = nnx.split(parametric_model)
            parametric_model = nnx.merge(graphdef, params)

        # Obtain trajectory for computation of internal energy
        if z_trajectory is None or time_steps is None:
            z_trajectory, time_steps = parametric_model(z_samples, history=True)
            # Set prob_dim on first call
            if self.prob_dim is None:
                self.prob_dim = z_trajectory.shape[-1]

        internal_energy = 0.0

        # Entropy computation
        if 'entropy' in self.functional:
            log_prob_init = stats.multivariate_normal.logpdf(
                z_samples,
                mean=jnp.zeros(self.prob_dim),
                cov=jnp.eye(self.prob_dim)
            )

            entropy = parametric_model.log_likelihood(
                t=time_steps,
                xt=z_trajectory,
                log_prob_init=log_prob_init,
                method=self.method,
                params=params,
                log_trajectory=False
            )

            internal_energy += jnp.mean(entropy) * self.coeff

        # Fisher information computation
        elif 'fisher' in self.functional:
            # Initialize score
            score_init = -z_samples  # Score of standard normal

            score = parametric_model.score_function(
                t=time_steps,
                xt=z_trajectory,
                score_init=score_init,
                method=self.method,
                params=params,
                score_trajectory=False
            )

            # Compute Fisher information
            fisher_info = self.sigma**4 / 8 * jnp.mean(jnp.linalg.norm(score, axis=-1)**2, axis=0)
            internal_energy += fisher_info

        return internal_energy
