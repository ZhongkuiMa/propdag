__docformat__ = "restructuredtext"
__all__ = [
    "TModel",
    "clear_fwd_cache",
    "clear_bwd_cache",
]


from abc import ABC
from collections.abc import Sequence
from typing import Literal

from propdag.propdag.custom_types import NodeType
from propdag.propdag.template._arguments import TArgument
from propdag.propdag.template._cache import TCache
from propdag.propdag.template._sort import (
    topo_sort_backward,
    topo_sort_forward_bfs,
    topo_sort_forward_dfs,
)
from propdag.propdag.utils import PropMode


def clear_fwd_cache(cache_counter: dict[NodeType, int], nodes: Sequence[NodeType]):
    """
    Clear forward caches for nodes when they are no longer needed.

    Decrements cache counter for specified nodes and clears caches when
    counter reaches zero.

    :param cache_counter: Dictionary tracking how many next nodes still need each node's cache
    :param nodes: List of nodes whose cache counters to decrement
    """
    for node in nodes:
        cache_counter[node] -= 1
        if cache_counter[node] <= 0:  # The output node will be -1
            node.clear_fwd_cache()
            del cache_counter[node]


def clear_bwd_cache(cache_counter: dict[NodeType, int], nodes: Sequence[NodeType]):
    """
    Clear backward caches for nodes when they are no longer needed.

    Decrements cache counter for specified nodes and clears caches when
    counter reaches zero.

    :param cache_counter: Dictionary tracking how many previous nodes still need each node's cache
    :param nodes: List of nodes whose cache counters to decrement
    """
    for node in nodes:
        if node in cache_counter:
            # Some next nodes may not be involved in the backward pass.
            cache_counter[node] -= 1
            if cache_counter[node] <= 0:  # The input node will be -1
                node.clear_bwd_cache()
                del cache_counter[node]


class TModel(ABC):
    """
    Template for computational graph model.

    This is a computation graph template. The overall logic is the whole graph and all
    nodes shares the same cache, their methods will operate on this cache. We do not
    consider subgraph, so there is no nested graphs.
    """

    _nodes: list[NodeType]
    _sort_strategy: Literal["dfs", "bfs"]
    _cache: TCache
    _arguments: TArgument
    _all_backward_sorts: dict[NodeType, list[NodeType]]

    verbose: bool

    def __init__(
        self,
        nodes: Sequence[NodeType],
        sort_strategy: Literal["dfs", "bfs"] = "bfs",
        verbose: bool = False,
    ):
        """
        Initialize a computational graph model.

        :param nodes: Sequence of nodes to include in the model
        :param verbose: Enable verbose output during execution
        :raises AssertionError: If not all nodes share the same cache and arguments
        """
        self.verbose = verbose
        self._sort_strategy = sort_strategy
        if sort_strategy == "dfs":
            self._nodes = topo_sort_forward_dfs(nodes, self.verbose)
        elif sort_strategy == "bfs":
            self._nodes = topo_sort_forward_bfs(nodes, self.verbose)
        else:
            raise ValueError(f"Unknown sort strategy: {sort_strategy}")
        self._cache = nodes[0].cache
        self._arguments = nodes[0].argument
        for node in nodes[1:]:
            assert node.cache == self._cache
            assert node.argument == self._arguments

        if self.arguments.prop_mode == PropMode.BACKWARD:
            self._all_backward_sorts = topo_sort_backward(self.nodes, self.verbose)

    def run(self, *args, **kwargs):
        """
        Execute the computational graph.

        Performs forward pass through all nodes in topological order and
        optionally performs back-substitution based on propagation mode.
        Intelligently clears caches to optimize memory usage.

        :param args: Positional arguments to pass to the model
        :param kwargs: Keyword arguments to pass to the model
        """
        cache_counter = {node: len(node.next_nodes) for node in self._nodes}

        node = self.nodes[0]
        if self.verbose:
            print(f"Forward pass {node.name}")
        node.forward()
        # No need to backward for the input node.

        for i in range(1, len(self.nodes)):
            node = self.nodes[i]
            if self.verbose:
                print(f"Forward pass {node.name}")
            node.forward()

            if self.arguments.prop_mode == PropMode.BACKWARD:
                self.backsub(node)

            clear_fwd_cache(cache_counter, node.pre_nodes)

        clear_fwd_cache(cache_counter, [node])

    def backsub(self, node: NodeType):
        """
        Perform back-substitution from a specified node.

        Executes backward passes for all nodes in the backward topological sort
        starting from the given node.

        :param node: Node to start back-substitution from
        """
        backward_sort = self._all_backward_sorts[node]

        if len(backward_sort) == 1:
            # No need to do backward pass for the input node.
            return

        cache_counter = {node: len(node.pre_nodes) for node in backward_sort}

        if self.verbose:
            print(f"\tBack-substitute {node.name}")
        node.backward()

        for j in range(1, len(backward_sort)):
            node = backward_sort[j]
            if self.verbose:
                print(f"\tBack-substitute {node.name}")
            node.backward()
            clear_bwd_cache(cache_counter, node.next_nodes)

        clear_bwd_cache(cache_counter, [node])

    @property
    def sort_strategy(self):
        """
        Get the sorting strategy used in the model.

        :returns: Sorting strategy (either 'dfs' or 'bfs')
        """
        return self._sort_strategy

    @property
    def nodes(self) -> list[NodeType]:
        """
        Get the nodes in the model.

        :returns: Topologically sorted list of nodes
        """
        return self._nodes

    @property
    def cache(self) -> TCache:
        """
        Get the shared cache for the model.

        :returns: Cache instance shared by all nodes
        """
        return self._cache

    @property
    def arguments(self) -> TArgument:
        """
        Get the shared arguments for the model.

        :returns: Arguments instance shared by all nodes
        """
        return self._arguments
