"""
PropDAG: Bound propagation for DAG-structured neural networks.

Provides abstract templates and example implementations for bound propagation
algorithms in neural network verification. Supports both forward and backward
propagation through directed acyclic graphs with residual/skip connections.

Main components:
- template: Abstract base classes (TNode, TModel, TCache, TArgument)
- toy: Example implementations with verbose logging
- utils: PropMode enum (FORWARD/BACKWARD)
"""

from propdag.propdag.template import (
    TArgument,
    TCache,
    TModel,
    TNode,
    topo_sort_backward,
    topo_sort_forward_bfs,
    topo_sort_forward_dfs,
)
from propdag.propdag.utils import PropMode

__all__ = [
    # Template exports
    "TArgument",
    "TCache",
    "TModel",
    "TNode",
    "topo_sort_backward",
    "topo_sort_forward_bfs",
    "topo_sort_forward_dfs",
    # Utils exports
    "PropMode",
]
