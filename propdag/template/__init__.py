"""
Template module for propagation of directed acyclic graph.

This module provides abstract base classes and utilities for building computational
graph structures that support both forward and backward propagation.
"""

from propdag.propdag.template._arguments import TArgument
from propdag.propdag.template._cache import TCache
from propdag.propdag.template._model import TModel
from propdag.propdag.template._node import TNode
from propdag.propdag.template._sort import (
    topo_sort_backward,
    topo_sort_forward_bfs,
    topo_sort_forward_dfs,
)

__all__ = [
    "TArgument",
    "TCache",
    "TModel",
    "TNode",
    "topo_sort_backward",
    "topo_sort_forward_bfs",
    "topo_sort_forward_dfs",
]
