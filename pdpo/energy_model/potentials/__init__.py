"""
Potential functions module for PDPO.

This module provides various potential energy functions including obstacles,
opinion dynamics, and other energy terms used in density path optimization.
"""

from .obstacles import (
    obstacle_cost_stunnel,
    obstacle_cost_vneck,
    obstacle_cost_gmm,
    congestion_cost,
    quadratic_well,
    geodesic,
    get_obstacle_function,
    list_obstacle_functions,
    OBSTACLE_FUNCTIONS
)

__all__ = [
    'obstacle_cost_stunnel',
    'obstacle_cost_vneck', 
    'obstacle_cost_gmm',
    'congestion_cost',
    'quadratic_well',
    'geodesic',
    'get_obstacle_function',
    'list_obstacle_functions',
    'OBSTACLE_FUNCTIONS'
]
