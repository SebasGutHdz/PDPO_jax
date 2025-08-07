#!/usr/bin/env python3
"""
Pre-train boundary parameters for PDPO using generative models.
"""

import os
# Disable XLA preallocation to avoid memory issues
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
from pathlib import Path
import argparse
import pickle
from typing import Dict, Any
import tqdm

import jax
import jax.random as jrn
from flax import nnx
import optax

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pdpo.core.config import GenerativeConfig, validate_config,setup_device
from pdpo.models.nn import create_mlp,create_mlp_time_embedding
from pdpo.data.toy_datasets import inf_train_gen
from pdpo.generative.models.matching_methods import FlowMatching, StochasticInterpolant
from pdpo.ode.solvers import EulerSolver
from pdpo.generative.models.interpolant_schedules import InterpolantSchedule



def create_method(config: GenerativeConfig, model: nnx.Module, key: jax.Array):
    """Create the appropriate matching method."""
    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(config.training.learning_rate),wrt = nnx.Param)
    
    # Create ODE solver
    ode_solver = EulerSolver()
    
    if config.training.method == "fm":
        return FlowMatching(
            vf_model=model,
            optimizer=optimizer,
            ode_solver=ode_solver,
            sigma=config.method.params.get("sigma", 0.0),
            time_sampling=config.method.params.get("time_sampling", "uniform")
        )
    elif config.training.method == "si":
        return StochasticInterpolant(
            vf_model=model,
            optimizer=optimizer,
            ode_solver=ode_solver,
            interpolant_schedule=config.method.params.get("interpolant_schedule", "linear"),
            schedule_params=config.method.params.get("schedule_params", {"sigma": 0.1}),
            time_sampling=config.method.params.get("time_sampling", "uniform")
        )
    else:
        raise ValueError(f"Unknown method: {config.training.method}")


def save_checkpoint(
    model: nnx.Module, 
    config: GenerativeConfig, 
    epoch: int, 
    loss: float,
    save_path: Path
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'model_state': nnx.state(model),
        'model_config': {
            'input_dim': config.model.input_dim,
            'hidden_dim': config.model.hidden_dim,
            'num_layers': config.model.num_layers,
            'activation': config.model.activation,
            'time_varying': config.model.time_varying
        },
        'epoch': epoch,
        'loss': loss,
        'data_config': {
            'source_type': config.data.source_type,
            'target_type': config.data.target_type,
            'dim': config.data.dim
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def train_generative_model(config_path: str) -> None:
    """Main training function."""
    # Load and validate config
    config = GenerativeConfig.from_yaml(config_path)
    validate_config(config)
    setup_device(config.training.device)
    # Setup JAX
    key = jrn.PRNGKey(config.training.seed)
    model_key, train_key = jrn.split(key)
    
    # Create model
    if config.model.type == 'mlp':
        model = create_mlp(
            input_size=config.model.input_dim,
            num_layers=config.model.num_layers,
            layer_width=config.model.hidden_dim,
            activation=config.model.activation,
        time_varying=config.model.time_varying,
        key=model_key
    )
    elif config.model.type == 'mlp_time_embedding':
        model = create_mlp_time_embedding(
            input_size=config.model.input_dim,
            num_layers=config.model.num_layers,
            layer_width=config.model.hidden_dim,
            activation=config.model.activation,
            key=model_key
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    # Create method
    method = create_method(config, model, train_key)
    
    # Setup output directory
    output_dir = Path(config.output.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training {config.training.method.upper()} model:")
    print(f"Source: {config.data.source_type} -> Target: {config.data.target_type}")
    print(f"Model: {config.model.num_layers} layers, {config.model.hidden_dim} hidden dim")
    
    pbar = tqdm.tqdm(total=config.training.n_epochs, desc="Training Progress")

    # Training loop
    for epoch in range(config.training.n_epochs):
        epoch_key = jrn.fold_in(train_key, epoch)
        data_key, ref_key = jrn.split(epoch_key)
        
        # Generate data
        target_data = inf_train_gen(
            config.data.target_type, 
            data_key, 
            config.training.batch_size, 
            config.data.dim
        )
        
        source_data = inf_train_gen(
            config.data.source_type,
            ref_key,
            config.training.batch_size,
            config.data.dim
        )
        
        # Training step
        loss, metrics = method.training_step(
            epoch_key, target_data, source_data
        )
        
        # Logging
        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch:5d}: Loss = {loss:.6f}")
        pbar.set_postfix({
            'epoch': epoch,
            'loss': f"{loss:.6f}"
        })
        
        # Save checkpoint
        if epoch % config.training.save_interval == 0 or epoch == config.training.n_epochs - 1:
            save_path = output_dir / f"{config.output.model_name}_epoch_{epoch}.pkl"
            save_checkpoint(model, config, epoch, loss, save_path)
        
        # Update progress bar
        pbar.update(1)
    
    pbar.close()
    
    # Save final model
    final_path = output_dir / f"{config.output.model_name}_final.pkl"
    save_checkpoint(model, config, config.training.n_epochs, loss, final_path)
    print(f"Training completed. Final model saved to: {final_path}")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Pre-train boundary parameters for PDPO")
    parser.add_argument(
        "config", 
        type=str, 
        help="Path to YAML configuration file"
    )
    
    args = parser.parse_args()
    train_generative_model(args.config)


if __name__ == "__main__":
    main()