"""
PropDAG: Bound propagation for DAG-structured neural networks.

Provides abstract templates and example implementations for bound propagation
algorithms in neural network verification. Supports both forward and backward
propagation through directed acyclic graphs with residual/skip connections.

Main components:
- template: Abstract base classes (TNode, TModel, TCache, TArgument) for forward graph
- template2: Abstract base classes (T2Node, T2Model, T2Cache, T2Argument) for reversed graph
- toy: Example implementations with verbose logging (uses template/)
- toy2: Example implementations for reversed graph semantics (uses template2/)
- utils: PropMode enum (FORWARD/BACKWARD)

**New in v2026.1.1**: Added template2/ and toy2/ modules for cleaner backward bound propagation semantics.
"""

__version__ = "2026.2.0"

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

# NEW in v2026.2.0: Template2 for reversed graph semantics
from propdag.template2 import (
    T2Argument,
    T2ArgumentType,
    T2Cache,
    T2CacheType,
    T2Model,
    T2Node,
    clear_bwd_cache_t2,
    reverse_dag,
    topo_sort_forward_bfs_t2,
    topo_sort_forward_dfs_t2,
)
from propdag.toy import (
    BackwardToyNode,
    ForwardToyNode,
    ToyArgument,
    ToyCache,
    ToyModel,
)

# NEW in v2026.2.0: Toy2 example implementations for reversed graph
from propdag.toy2 import (
    Toy2Argument,
    Toy2Cache,
    Toy2Model,
    Toy2Node,
)
from propdag.utils import PropMode

__all__ = [
    # Core utilities
    "PropMode",
    # Template2 (new - reversed graph)
    "T2Argument",
    "T2ArgumentType",
    "T2Cache",
    "T2CacheType",
    "T2Model",
    "T2Node",
    # Template (original - forward graph)
    "TArgument",
    "TCache",
    "TModel",
    "TNode",
    # Toy2 (new - example for reversed graph)
    "Toy2Argument",
    "Toy2Cache",
    "Toy2Model",
    "Toy2Node",
    "__version__",
    "clear_bwd_cache",
    "clear_bwd_cache_t2",
    "clear_fwd_cache",
    "reverse_dag",
    "topo_sort_forward_bfs_t2",
    "topo_sort_forward_dfs_t2",
]
