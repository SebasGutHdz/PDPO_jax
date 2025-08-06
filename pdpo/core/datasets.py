"""
JAX implementation of toy datasets for PDPO.

This module provides various synthetic datasets commonly used in optimal transport
and density path optimization problems. All datasets return JAX arrays instead of
NumPy arrays for compatibility with the JAX ecosystem.

Key changes from the original NumPy implementation:
1. Uses JAX random number generation with explicit keys
2. Returns jnp.Array instead of np.ndarray
3. Maintains functional programming style
4. Provides both batch generation and streaming interfaces
"""

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple, Union, Optional
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


def inf_train_gen(
    data_type: str,
    key: PRNGKeyArray,
    batch_size: int = 200,
    dim: int = 2
) -> Float[Array, "batch_size dim"]:
    """
    Generate a batch of samples from the specified dataset.
    
    Args:
        data_type: Type of dataset to generate
        key: JAX random key
        batch_size: Number of samples to generate
        dim: Dimension of the data (for high-dimensional datasets)
        
    Returns:
        Array of shape (batch_size, dim) containing the generated samples
    """
    
    if data_type == "swissroll":
        # Use sklearn for swiss roll generation, then convert to JAX
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return jnp.array(data)

    elif data_type == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return jnp.array(data)

    elif data_type == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # Generate linspace points for each ring
        linspace4 = jnp.linspace(0, 2 * jnp.pi, n_samples4, endpoint=False)
        linspace3 = jnp.linspace(0, 2 * jnp.pi, n_samples3, endpoint=False)
        linspace2 = jnp.linspace(0, 2 * jnp.pi, n_samples2, endpoint=False)
        linspace1 = jnp.linspace(0, 2 * jnp.pi, n_samples1, endpoint=False)

        # Generate circles with different radii
        circ4_x = jnp.cos(linspace4)
        circ4_y = jnp.sin(linspace4)
        circ3_x = jnp.cos(linspace3) * 0.75
        circ3_y = jnp.sin(linspace3) * 0.75
        circ2_x = jnp.cos(linspace2) * 0.5
        circ2_y = jnp.sin(linspace2) * 0.5
        circ1_x = jnp.cos(linspace1) * 0.25
        circ1_y = jnp.sin(linspace1) * 0.25

        X = jnp.stack([
            jnp.concatenate([circ4_x, circ3_x, circ2_x, circ1_x]),
            jnp.concatenate([circ4_y, circ3_y, circ2_y, circ1_y])
        ], axis=1) * 3.0
        
        # Add noise
        noise = random.normal(key, shape=X.shape) * 0.08
        X = X + noise
        
        # Shuffle
        shuffle_key = random.fold_in(key, 1)
        perm = random.permutation(shuffle_key, X.shape[0])
        return X[perm]

    elif data_type == "moons":
        # For moons, use sklearn then convert
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + jnp.array([-1, -0.2])
        return jnp.array(data)
    
    elif data_type in ["8gaussians", "8gaussiansv2", "8gaussiansv3"]:
        return _generate_8gaussians(key, batch_size, data_type)
    
    elif data_type == "4gaussians":
        return _generate_4gaussians(key, batch_size)
    
    elif data_type in ["gaussian0", "gaussian1"]:
        return _generate_single_gaussian(key, batch_size, data_type)
    
    elif data_type in ["gaussian0_d", "gaussian1_d"]:
        return _generate_gaussian_d(key, batch_size, dim, data_type)
    
    elif data_type in ["gauss0_opinion_2d", "gauss1_opinion_2d"]:
        return _generate_opinion_gaussian_2d(key, batch_size, data_type)
    
    elif data_type in ["gauss0_opinion_1000d", "gauss1_opinion_1000d"]:
        return _generate_opinion_gaussian_1000d(key, batch_size, data_type)
    
    elif data_type in ["half_std_gaussian", "std_gaussian"]:
        return _generate_std_gaussian(key, batch_size, data_type)
    
    elif data_type == "pinwheel":
        return _generate_pinwheel(key, batch_size)
    
    elif data_type == "2spirals":
        return _generate_2spirals(key, batch_size)
    
    elif data_type == "checkerboard":
        return _generate_checkerboard(key, batch_size)
    
    elif data_type == "line":
        return _generate_line(key, batch_size)
    
    elif data_type == "cos":
        return _generate_cos(key, batch_size)
    
    else:
        # Default to 8gaussians
        return _generate_8gaussians(key, batch_size, "8gaussians")


def _generate_8gaussians(
    key: PRNGKeyArray, 
    batch_size: int, 
    variant: str = "8gaussians"
) -> Float[Array, "batch_size 2"]:
    """Generate 8 Gaussians mixture."""
    if variant == "8gaussians":
        scale = 4.0
    elif variant == "8gaussiansv2":
        scale = 8.0
    elif variant == "8gaussiansv3":
        scale = 16.0
    else:
        scale = 4.0
    
    centers = jnp.array([
        [1, 0], [-1, 0], [0, 1], [0, -1], 
        [1. / jnp.sqrt(2), 1. / jnp.sqrt(2)],
        [1. / jnp.sqrt(2), -1. / jnp.sqrt(2)], 
        [-1. / jnp.sqrt(2), 1. / jnp.sqrt(2)], 
        [-1. / jnp.sqrt(2), -1. / jnp.sqrt(2)]
    ]) * scale
    
    # Generate noise
    noise_key = random.fold_in(key, 0)
    if variant == "8gaussiansv3":
        cov_scale = 1.0
    else:
        cov_scale = 0.1
    noise = random.multivariate_normal(
        noise_key, 
        mean=jnp.zeros(2), 
        cov=jnp.eye(2) * cov_scale, 
        shape=(batch_size,)
    )
    
    # Assign samples to centers
    elements_per_center = batch_size // 8
    center_indices = jnp.repeat(jnp.arange(8), elements_per_center)
    
    # Handle remainder
    remainder = batch_size - len(center_indices)
    if remainder > 0:
        extra_indices = jnp.full(remainder, 7)
        center_indices = jnp.concatenate([center_indices, extra_indices])
    
    dataset = centers[center_indices] + noise
    return dataset


def _generate_4gaussians(
    key: PRNGKeyArray, 
    batch_size: int
) -> Float[Array, "batch_size 2"]:
    """Generate 4 Gaussians mixture."""
    scale = 2.0
    centers = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) * scale
    
    noise = random.multivariate_normal(
        key, 
        mean=jnp.zeros(2), 
        cov=jnp.eye(2) * 0.1, 
        shape=(batch_size,)
    )
    
    elements_per_center = batch_size // 4
    center_indices = jnp.repeat(jnp.arange(4), elements_per_center)
    
    # Handle remainder
    remainder = batch_size - len(center_indices)
    if remainder > 0:
        extra_indices = jnp.full(remainder, 3)
        center_indices = jnp.concatenate([center_indices, extra_indices])
    
    dataset = centers[center_indices] + noise
    return dataset


def _generate_single_gaussian(
    key: PRNGKeyArray, 
    batch_size: int, 
    data_type: str
) -> Float[Array, "batch_size 2"]:
    """Generate single 2D Gaussian."""
    cov_matrix = jnp.eye(2) * 0.5
    
    if data_type == "gaussian0":
        mean = jnp.array([-11.0, -1.0])
    else:  # gaussian1
        mean = jnp.array([11.0, 1.0])
    
    return random.multivariate_normal(key, mean=mean, cov=cov_matrix, shape=(batch_size,))


def _generate_gaussian_d(
    key: PRNGKeyArray, 
    batch_size: int, 
    dim: int, 
    data_type: str
) -> Float[Array, "batch_size dim"]:
    """Generate high-dimensional Gaussian."""
    cov_matrix = jnp.eye(dim) * 0.5
    
    if data_type == "gaussian0_d":
        mean = jnp.ones(dim)
    else:  # gaussian1_d
        mean = -jnp.ones(dim)
    
    return random.multivariate_normal(key, mean=mean, cov=cov_matrix, shape=(batch_size,))


def _generate_opinion_gaussian_2d(
    key: PRNGKeyArray, 
    batch_size: int, 
    data_type: str
) -> Float[Array, "batch_size 2"]:
    """Generate 2D Gaussians for opinion dynamics."""
    if data_type == "gauss0_opinion_2d":
        cov = jnp.array([[0.5, 0.0], [0.0, 0.25]])
        mean = jnp.array([0.0, 0.0])
    else:  # gauss1_opinion_2d
        cov = 3.0 * jnp.eye(2)
        mean = jnp.array([0.0, 0.0])
    
    return random.multivariate_normal(key, mean=mean, cov=cov, shape=(batch_size,))


def _generate_opinion_gaussian_1000d(
    key: PRNGKeyArray, 
    batch_size: int, 
    data_type: str
) -> Float[Array, "batch_size 1000"]:
    """Generate 1000D Gaussians for opinion dynamics."""
    if data_type == "gauss0_opinion_1000d":
        cov = jnp.eye(1000) * 0.25
        cov = cov.at[0, 0].set(4.0)
        mean = jnp.zeros(1000)
    else:  # gauss1_opinion_1000d
        cov = jnp.eye(1000) * 3.0
        mean = jnp.zeros(1000)
    
    return random.multivariate_normal(key, mean=mean, cov=cov, shape=(batch_size,))


def _generate_std_gaussian(
    key: PRNGKeyArray, 
    batch_size: int, 
    data_type: str
) -> Float[Array, "batch_size 2"]:
    """Generate standard or half-standard Gaussian."""
    if data_type == "half_std_gaussian":
        scale = 0.5
    else:  # std_gaussian
        scale = 1.0
    
    return random.normal(key, shape=(batch_size, 2)) * scale


def _generate_pinwheel(
    key: PRNGKeyArray, 
    batch_size: int
) -> Float[Array, "batch_size 2"]:
    """Generate pinwheel dataset."""
    radial_std = 0.3
    tangential_std = 0.1
    num_classes = 5
    num_per_class = batch_size // 5
    rate = 0.25
    
    rads = jnp.linspace(0, 2 * jnp.pi, num_classes, endpoint=False)
    
    features = random.normal(key, shape=(num_classes * num_per_class, 2)) * jnp.array([radial_std, tangential_std])
    features = features.at[:, 0].add(1.0)
    
    labels = jnp.repeat(jnp.arange(num_classes), num_per_class)
    
    angles = rads[labels] + rate * jnp.exp(features[:, 0])
    
    rotations = jnp.stack([
        jnp.cos(angles), -jnp.sin(angles), 
        jnp.sin(angles), jnp.cos(angles)
    ]).T.reshape(-1, 2, 2)
    
    result = jnp.einsum("ti,tij->tj", features, rotations)
    
    # Permute
    perm_key = random.fold_in(key, 1)
    perm = random.permutation(perm_key, result.shape[0])
    
    return 2 * result[perm]


def _generate_2spirals(
    key: PRNGKeyArray, 
    batch_size: int
) -> Float[Array, "batch_size 2"]:
    """Generate 2 spirals dataset."""
    key1, key2, key3 = random.split(key, 3)
    
    n = jnp.sqrt(random.uniform(key1, shape=(batch_size // 2, 1))) * 540 * (2 * jnp.pi) / 360
    noise1 = random.uniform(key2, shape=(batch_size // 2, 1)) * 0.5
    noise2 = random.uniform(key3, shape=(batch_size // 2, 1)) * 0.5
    
    d1x = -jnp.cos(n) * n + noise1
    d1y = jnp.sin(n) * n + noise2
    
    x = jnp.vstack([
        jnp.hstack([d1x, d1y]), 
        jnp.hstack([-d1x, -d1y])
    ]) / 3
    
    # Add noise
    noise_key = random.fold_in(key, 4)
    noise = random.normal(noise_key, shape=x.shape) * 0.1
    
    return x + noise


def _generate_checkerboard(
    key: PRNGKeyArray, 
    batch_size: int
) -> Float[Array, "batch_size 2"]:
    """Generate checkerboard pattern."""
    grid_size = 4
    total_squares = (grid_size * grid_size) // 2
    points_per_square = batch_size // total_squares
    
    points_list = []
    square_count = 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0:
                # Generate points for this square
                square_key = random.fold_in(key, square_count)
                key_x, key_y = random.split(square_key, 2)
                
                x1 = random.uniform(key_x, shape=(points_per_square,)) * 2 + (i * 2 - 4)
                x2 = random.uniform(key_y, shape=(points_per_square,)) * 2 + (j * 2 - 4)
                
                square_points = jnp.stack([x1, x2], axis=1)
                points_list.append(square_points)
                square_count += 1
    
    return jnp.concatenate(points_list, axis=0)


def _generate_line(
    key: PRNGKeyArray, 
    batch_size: int
) -> Float[Array, "batch_size 2"]:
    """Generate line dataset."""
    x = random.uniform(key, shape=(batch_size,)) * 5 - 2.5
    y = x
    return jnp.stack([x, y], axis=1)


def _generate_cos(
    key: PRNGKeyArray, 
    batch_size: int
) -> Float[Array, "batch_size 2"]:
    """Generate cosine dataset."""
    x = random.uniform(key, shape=(batch_size,)) * 5 - 2.5
    y = jnp.sin(x) * 2.5
    return jnp.stack([x, y], axis=1)


# Streaming dataset class for continuous generation
class InfiniteDatasetGenerator:
    """
    Infinite dataset generator that produces batches on demand.
    Maintains internal random state for reproducible sequences.
    """
    
    def __init__(self, data_type: str, key: PRNGKeyArray, dim: int = 2):
        self.data_type = data_type
        self.dim = dim
        self.key = key
        self.counter = 0
    
    def generate_batch(self, batch_size: int) -> Float[Array, "batch_size dim"]:
        """Generate a single batch of data."""
        batch_key = random.fold_in(self.key, self.counter)
        self.counter += 1
        return inf_train_gen(self.data_type, batch_key, batch_size, self.dim)
    
    def __call__(self, batch_size: int) -> Float[Array, "batch_size dim"]:
        """Make the generator callable."""
        return self.generate_batch(batch_size)


# Utility functions for dataset information
def get_dataset_info(data_type: str) -> dict:
    """Get information about a dataset type."""
    info = {
        "swissroll": {"default_dim": 2, "description": "3D Swiss roll projected to 2D"},
        "circles": {"default_dim": 2, "description": "Concentric circles"},
        "rings": {"default_dim": 2, "description": "Four concentric rings"},
        "moons": {"default_dim": 2, "description": "Two interleaving half circles"},
        "8gaussians": {"default_dim": 2, "description": "8 Gaussians in a circle"},
        "8gaussiansv2": {"default_dim": 2, "description": "8 Gaussians in a circle (larger scale)"},
        "8gaussiansv3": {"default_dim": 2, "description": "8 Gaussians in a circle (largest scale)"},
        "4gaussians": {"default_dim": 2, "description": "4 Gaussians in a cross"},
        "gaussian0": {"default_dim": 2, "description": "Single Gaussian at (-11, -1)"},
        "gaussian1": {"default_dim": 2, "description": "Single Gaussian at (11, 1)"},
        "gaussian0_d": {"default_dim": None, "description": "High-dim Gaussian with positive mean"},
        "gaussian1_d": {"default_dim": None, "description": "High-dim Gaussian with negative mean"},
        "gauss0_opinion_2d": {"default_dim": 2, "description": "2D Gaussian for opinion dynamics (source)"},
        "gauss1_opinion_2d": {"default_dim": 2, "description": "2D Gaussian for opinion dynamics (target)"},
        "gauss0_opinion_1000d": {"default_dim": 1000, "description": "1000D Gaussian for opinion dynamics (source)"},
        "gauss1_opinion_1000d": {"default_dim": 1000, "description": "1000D Gaussian for opinion dynamics (target)"},
        "pinwheel": {"default_dim": 2, "description": "Pinwheel pattern"},
        "2spirals": {"default_dim": 2, "description": "Two intertwining spirals"},
        "checkerboard": {"default_dim": 2, "description": "Checkerboard pattern"},
        "line": {"default_dim": 2, "description": "Diagonal line"},
        "cos": {"default_dim": 2, "description": "Cosine curve"}
    }
    
    return info.get(data_type, {"default_dim": 2, "description": "Unknown dataset"})


def list_available_datasets() -> list[str]:
    """List all available dataset types."""
    return list(get_dataset_info("dummy").keys())[:-1]  # Exclude the dummy key