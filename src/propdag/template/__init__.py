"""
Template module for propagation of directed acyclic graph.

This module provides abstract base classes and utilities for building computational
graph structures that support both forward and backward propagation.
"""

from propdag.template._arguments import TArgument
from propdag.template._cache import TCache
from propdag.template._model import TModel, clear_bwd_cache, clear_fwd_cache
from propdag.template._node import TNode
from propdag.template._sort import (
    topo_sort_backward,
    topo_sort_forward_bfs,
    topo_sort_forward_dfs,
)

__all__ = [
    # Core template classes
    "TArgument",
    "TCache",
    "TModel",
    "TNode",
    # Cache management utilities
    "clear_bwd_cache",
    "clear_fwd_cache",
    # Note: Topological sorting functions remain importable but not in public API
]
