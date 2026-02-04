"""
Combined potential functional class managing linear, internal, and interaction potentials.
"""

from typing import Optional, Union
from jaxtyping import PyTree, Array
import jax.numpy as jnp
from flax import nnx
import jax

from pdpo.energy_model.functionals.linear_functional import LinearPotential
from pdpo.energy_model.functionals.internal_functional import InternalPotential
from pdpo.energy_model.functionals.interaction_functional import InteractionPotential


class Potential:
    """
    A class to manage three types of potentials:

    1. Linear potential: F(ρ) = ∫ U(x)ρ(x)dx
    2. Internal potential: F(ρ) = ∫ f(ρ(x))dx (entropy, Fisher info)
    3. Interaction potential: F(ρ) = 0.5 ∫∫ W(x-y)ρ(x)ρ(y)dxdy

    The total energy is F(ρ) = F_linear(ρ) + F_internal(ρ) + F_interaction(ρ).
    """

    def __init__(self,
                 linear: Optional[LinearPotential] = None,
                 internal: Optional[InternalPotential] = None,
                 interaction: Optional[InteractionPotential] = None):
        """
        Initialize combined potential.

        Args:
            linear: LinearPotential instance
            internal: InternalPotential instance
            interaction: InteractionPotential instance
        """
        self.linear = linear
        self.internal = internal
        self.interaction = interaction

    def evaluate_energy(self,
                       parametric_model,
                       z_samples: Array,
                       params: Optional[PyTree] = None) -> tuple[float, Array, float, float, float]:
        """
        Evaluate the total energy functional:
        F(ρ) = F_linear(ρ) + F_internal(ρ) + F_interaction(ρ)

        Args:
            parametric_model: Model instance (NeuralODE or similar)
            z_samples: Reference samples (batch_size, d)
            params: Optional PyTree of parameters for the dynamics model

        Returns:
            energy: Total energy functional
            x_samples: Transformed samples (batch_size, d)
            linear_energy: Linear potential energy
            internal_energy: Internal potential energy
            interaction_energy: Interaction potential energy
        """
        if params is None:
            _, params = nnx.split(parametric_model)

        # Transform reference samples
        if self.internal is not None:
            # Need trajectory for internal potential
            z_trajectory, time_steps = parametric_model(z_samples, history=True, params=params)
            x_samples = z_trajectory[:, -1, :]
        else:
            x_samples = parametric_model(z_samples, params=params)

        energy = 0.0
        linear_energy = 0.0
        internal_energy = 0.0
        interaction_energy = 0.0

        # Linear potential
        if self.linear is not None:
            linear_energy, _ = self.linear.evaluate_energy(
                parametric_model,
                z_samples,
                x_samples=x_samples
            )
            linear_energy = linear_energy * self.linear.coeff
            energy += linear_energy

        # Internal potential
        if self.internal is not None:
            internal_energy = self.internal(
                parametric_model=parametric_model,
                z_samples=z_samples,
                z_trajectory=z_trajectory,
                time_steps=time_steps,
                params=params
            )
            energy += internal_energy

        # Interaction potential
        if self.interaction is not None:
            batch_size = z_samples.shape[0]
            # Split samples for interaction energy computation
            part1_samples = x_samples[:batch_size // 2, :]
            part2_samples = x_samples[batch_size // 2:, :]

            # Ensure equal lengths
            if part1_samples.shape[0] < part2_samples.shape[0]:
                part2_samples = part2_samples[:part1_samples.shape[0], :]
            elif part2_samples.shape[0] < part1_samples.shape[0]:
                part1_samples = part1_samples[:part2_samples.shape[0], :]

            interaction_energy, _ = self.interaction.evaluate_energy(
                parametric_model,
                z_samples,
                x_samples=part1_samples,
                y_samples=part2_samples
            )
            energy += interaction_energy * self.interaction.coeff

        return energy, x_samples, linear_energy, internal_energy, interaction_energy

    def compute_energy_gradient(self,
                               parametric_model,
                               z_samples: Array,
                               params: PyTree) -> PyTree:
        """
        Compute the gradient of the total energy functional:
        ∇_θ F(ρ_θ) = ∇_θ[F_linear(ρ_θ) + F_internal(ρ_θ) + F_interaction(ρ_θ)]

        Args:
            parametric_model: Model instance
            z_samples: Reference samples (batch_size, d)
            params: PyTree of parameters for the dynamics model

        Returns:
            energy_gradient: Gradient of the total energy functional
            energy: Current energy value
            energy_breakdown: Dictionary with individual energy components
        """
        if params is None:
            _, params = nnx.split(parametric_model)

        def energy_evaluation(p: PyTree) -> tuple[Array, dict]:
            # Transform reference samples
            if self.internal is not None:
                z_trajectory, time_steps = parametric_model(z_samples, history=True, params=p)
                x_samples = z_trajectory[:, -1, :]
            else:
                x_samples = parametric_model(z_samples, params=p)

            energy = 0.0
            linear_energy = 0.0
            internal_energy = 0.0
            interaction_energy = 0.0

            # Linear potential
            if self.linear is not None:
                linear_energy, _ = self.linear.evaluate_energy(
                    parametric_model,
                    z_samples,
                    x_samples=x_samples
                )
                linear_energy = linear_energy * self.linear.coeff
                energy += linear_energy

            # Internal potential
            if self.internal is not None:
                internal_energy = self.internal(
                    parametric_model=parametric_model,
                    z_samples=z_samples,
                    z_trajectory=z_trajectory,
                    time_steps=time_steps,
                    params=p
                )
                energy += internal_energy

            # Interaction potential
            if self.interaction is not None:
                batch_size = z_samples.shape[0]
                part1_samples = x_samples[:batch_size // 2, :]
                part2_samples = x_samples[batch_size // 2:, :]

                if part1_samples.shape[0] < part2_samples.shape[0]:
                    part2_samples = part2_samples[:part1_samples.shape[0], :]
                elif part2_samples.shape[0] < part1_samples.shape[0]:
                    part1_samples = part1_samples[:part2_samples.shape[0], :]

                interaction_energy, _ = self.interaction.evaluate_energy(
                    parametric_model,
                    z_samples,
                    x_samples=part1_samples,
                    y_samples=part2_samples,
                    params=p
                )
                energy += interaction_energy * self.interaction.coeff

            energy_breakdown = {
                'internal_energy': internal_energy,
                'linear_energy': linear_energy,
                'interaction_energy': interaction_energy
            }

            return energy, energy_breakdown

        (energy, energy_breakdown), energy_grad = jax.value_and_grad(
            energy_evaluation,
            has_aux=True
        )(params)

        return energy_grad, energy, energy_breakdown
