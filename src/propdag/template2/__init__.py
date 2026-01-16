"""
Template2: Reversed graph semantics for backward bound propagation.

This module provides abstract base classes with reversed graph semantics,
eliminating semantic confusion in backward bound propagation algorithms.

**Key Innovation**:
Graph edges are reversed internally, so "forward" propagation through the
reversed graph achieves backward bound propagation naturally.

**Key Differences from template/**:
- template/: Graph Input → Output, backward_bound() goes Output → Input (confusing!)
- template2/: Graph reversed Output → Input, forward() goes "forward" (clear!)

**Single Mode**:
Unlike template/ with multiple modes (FORWARD/BACKWARD),
template2/ is single-purpose: backward bound propagation only.

**Graph Reversal**:
- User builds: Input → Hidden → Output (normal construction)
- T2Model reverses to: Output → Hidden → Input (automatic)
- Propagation flows "forward": Output → Hidden → Input

Example Usage::

    from propdag.template2 import T2Model, T2Node, T2Cache, T2Argument

    # User builds graph normally (Input → Output)
    cache = Toy2Cache()
    args = Toy2Argument()

    input_node = Toy2Node("Input", cache, args)
    output_node = Toy2Node("Output", cache, args)

    input_node.next_nodes = [output_node]
    output_node.pre_nodes = [input_node]

    # T2Model automatically reverses edges internally
    model = T2Model([input_node, output_node])
    model.run()  # forward() propagates Output → Input (backward propagation!)

Components:
- T2Argument: Arguments for reversed graph models (no prop_mode needed)
- T2Cache: Cache for reversed graph (simplified naming)
- T2Model: Model with automatic graph reversal
- T2Node: Abstract node base class for reversed graph
- clear_bwd_cache_t2: Cache clearing utility for template2
- reverse_dag: Helper function to reverse graph edges
- topo_sort_forward_bfs_t2/topo_sort_forward_dfs_t2: Topological sorting for reversed graphs
"""

from propdag.template2._arguments import T2Argument
from propdag.template2._cache import T2Cache
from propdag.template2._model import T2Model, clear_bwd_cache_t2, reverse_dag
from propdag.template2._node import T2ArgumentType, T2CacheType, T2Node
from propdag.template2._sort import topo_sort_forward_bfs_t2, topo_sort_forward_dfs_t2

__all__ = [
    # Core template2 classes
    "T2Argument",
    "T2ArgumentType",
    "T2Cache",
    "T2CacheType",
    "T2Model",
    "T2Node",
    # Cache management utility
    "clear_bwd_cache_t2",
    # Graph reversal helper
    "reverse_dag",
    # Topological sorting (not in public API but importable)
    "topo_sort_forward_bfs_t2",
    "topo_sort_forward_dfs_t2",
]
