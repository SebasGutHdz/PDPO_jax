"""
Configuration management for PDPO JAX implementation.
"""
import jax
from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class TrainingConfig:
    """Training configuration."""
    method: str
    n_epochs: int
    batch_size: int
    learning_rate: float
    save_interval: int
    device: str
    seed: int


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    type: str  
    input_dim: int
    hidden_dim: int
    num_layers: int
    activation: str
    time_varying: bool


@dataclass
class MethodConfig:
    """Method-specific configuration."""
    params: Dict[str, Any]


@dataclass
class DataConfig:
    """Data configuration."""
    source_type: str
    target_type: str
    dim: int


@dataclass
class OutputConfig:
    """Output configuration."""
    checkpoint_dir: str
    model_name: str


@dataclass
class GenerativeConfig:
    """Complete generative model configuration."""
    training: TrainingConfig
    model: ModelConfig
    method: MethodConfig
    data: DataConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'GenerativeConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            training=TrainingConfig(**config_dict['training']),
            model=ModelConfig(**config_dict['model']),
            method=MethodConfig(params=config_dict['method']),
            data=DataConfig(**config_dict['data']),
            output=OutputConfig(**config_dict['output'])
        )


def validate_config(config: GenerativeConfig) -> None:
    """Validate configuration parameters."""
    assert config.training.method in ["fm", "si"], f"Unknown method: {config.training.method}"
    assert config.model.num_layers >= 2, "num_layers must be >= 2"
    assert config.training.n_epochs > 0, "n_epochs must be positive"
    assert config.training.batch_size > 0, "batch_size must be positive"

def setup_device(device_str: str):
    """Setup JAX device from config string."""
    if device_str == "cpu":
        device = jax.devices("cpu")[0]
    elif device_str == "gpu":
        if jax.devices("gpu"):
            device = jax.devices("gpu")[0]
        else:
            print("Warning: No GPU available, falling back to CPU")
            device = jax.devices("cpu")[0]
    elif device_str.startswith("gpu:"):
        gpu_id = int(device_str.split(":")[1])
        gpu_devices = jax.devices("gpu")
        if gpu_id < len(gpu_devices):
            device = gpu_devices[gpu_id]
        else:
            raise ValueError(f"GPU {gpu_id} not available. Available GPUs: {len(gpu_devices)}")
    else:
        raise ValueError(f"Invalid device specification: {device_str}")
    
    jax.default_device(device)
    print(f"Using device: {device}")
    return