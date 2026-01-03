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

__version__ = "2026.1.0"

from propdag.template import (
    TArgument,
    TCache,
    TModel,
    TNode,
    clear_bwd_cache,
    clear_fwd_cache,
    topo_sort_backward,
    topo_sort_forward_bfs,
    topo_sort_forward_dfs,
)
from propdag.toy import (
    BackwardToyNode,
    ForwardToyNode,
    ToyArgument,
    ToyCache,
    ToyModel,
)
from propdag.utils import PropMode

__all__ = [
    "PropMode",
    "TArgument",
    "TCache",
    "TModel",
    "TNode",
    "__version__",
    "clear_bwd_cache",
    "clear_fwd_cache",
]
