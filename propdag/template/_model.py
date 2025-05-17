__docformat__ = "restructuredtext"
__all__ = [
    "TModel",
    "clear_fwd_cache",
    "clear_bwd_cache",
]


from abc import ABC

from ._arguments import TArgument
from ._cache import TCache
from ._node import TNode
from ..utils import *


def _check_input_output_number(nodes: list[TNode]):
    """
    Verify that the graph has exactly one input node and one output node.

    :param nodes: List of nodes in the computational graph
    :type nodes: list[TNode]
    :raises ValueError: When number of input or output nodes is not equal to 1
    """
    n_inputs = 0
    n_outputs = 0
    for node in nodes:
        if len(node.pre_nodes) == 0:
            n_inputs += 1
        if len(node.next_nodes) == 0:
            n_outputs += 1

    # TODO: The future version should support multiple input and output nodes. The
    #  current version may have errors in the cache handling for multiple inputs and
    #  outputs
    if n_inputs != 1:
        raise ValueError(
            f"Only one input node is allowed, but {n_inputs} input nodes are found."
        )

    if n_outputs != 1:
        raise ValueError(
            f"Only one output node is allowed, but {n_outputs} output nodes are found."
        )


def _topo_sort_forward(nodes: list[TNode]) -> list[TNode]:
    """
    Perform a breadth-first topological sort of nodes.

    This is a breadth-first search algorithm to sort the nodes in topological order.
    We need breadth-first search because we do not want to cache the nodes close to
    the input node. In neural networks, a layer close to the input has more dimensions.

    :param nodes: List of nodes to sort
    :type nodes: list[TNode]
    :returns: Topologically sorted list of nodes
    :rtype: list[TNode]
    :raises ValueError: If the graph contains a cycle
    """
    _check_input_output_number(nodes)

    in_degrees = {node: len(node.pre_nodes) for node in nodes}

    queue = [node for node in nodes if in_degrees[node] == 0]
    sorted_nodes = []
    while queue:
        node = queue.pop(0)
        sorted_nodes.append(node)
        for next_node in node.next_nodes:
            in_degrees[next_node] -= 1
            if in_degrees[next_node] == 0:
                queue.append(next_node)

    if len(sorted_nodes) != len(nodes):
        raise ValueError("Graph has a cycle, cannot perform topological sort")

    return sorted_nodes


def _topo_sort_backward(nodes: list[TNode]) -> dict[TNode, list[TNode]]:
    """
    Generate backward topological sorts for each node.

    For each node, computes a topological sort of all nodes required
    for back-substitution from that node.

    :param nodes: List of nodes in the computational graph
    :type nodes: list[TNode]
    :returns: Dictionary mapping each node to its backward topological sort
    :rtype: dict[TNode, list[TNode]]
    """

    def dfs(node, visited, result):
        if node in visited:
            return
        visited.add(node)
        for pre_node in node.pre_nodes:
            dfs(pre_node, visited, result)
        result.append(node)  # post-order, ensures pre_node comes before node

    backward_sorts = {}
    for node in nodes:
        visited = set()
        result = []
        dfs(node, visited, result)
        backward_sorts[node] = result  # already in correct topological order

    for node, backward_sort in backward_sorts.items():
        backward_sorts[node] = backward_sort[::-1]  # reverse to get the correct order

    # THE FOLLOWING IS FOR DEBUGGING PURPOSES
    # for node, backward_sort in backward_sorts.items():
    #     print(node)
    #     print([n.name for n in backward_sort])

    return backward_sorts


def clear_fwd_cache(cache_counter: dict[TNode, int], nodes: list[TNode]):
    """
    Clear forward caches for nodes when they are no longer needed.

    Decrements cache counter for specified nodes and clears caches when
    counter reaches zero.

    :param cache_counter: Dictionary tracking how many next nodes still need each node's cache
    :type cache_counter: dict[TNode, int]
    :param nodes: List of nodes whose cache counters to decrement
    :type nodes: list[TNode]
    """
    for node in nodes:
        cache_counter[node] -= 1
        if cache_counter[node] <= 0:  # The output node will be -1
            node.clear_fwd_cache()
            del cache_counter[node]


def clear_bwd_cache(cache_counter: dict[TNode, int], nodes: list[TNode]):
    """
    Clear backward caches for nodes when they are no longer needed.

    Decrements cache counter for specified nodes and clears caches when
    counter reaches zero.

    :param cache_counter: Dictionary tracking how many previous nodes still need each node's cache
    :type cache_counter: dict[TNode, int]
    :param nodes: List of nodes whose cache counters to decrement
    :type nodes: list[TNode]
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

    :ivar _nodes: Topologically sorted list of nodes in the graph
    :ivar _cache: Shared cache instance for all nodes
    :ivar _arguments: Shared arguments instance for all nodes
    :ivar _all_backward_sorts: Mapping from nodes to their backward topological sorts
    """

    _nodes: list[TNode]
    _cache: TCache
    _arguments: TArgument
    _all_backward_sorts: dict[TNode, list[TNode]]

    def __init__(self, nodes: list[TNode], verbose: bool = False):
        """
        Initialize a computational graph model.

        :param nodes: List of nodes to include in the model
        :type nodes: list[TNode]
        :param verbose: Enable verbose output during execution
        :type verbose: bool, optional
        :raises AssertionError: If not all nodes share the same cache and arguments
        """
        self.verbose = verbose
        self._nodes = _topo_sort_forward(nodes)
        self._cache = nodes[0].cache
        self._arguments = nodes[0].argument
        for node in nodes[1:]:
            assert node.cache == self._cache
            assert node.argument == self._arguments

        if self.arguments.prop_mode == PropMode.BACKWARD:
            self._all_backward_sorts = _topo_sort_backward(self.nodes)

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

    def backsub(self, node: TNode):
        """
        Perform back-substitution from a specified node.

        Executes backward passes for all nodes in the backward topological sort
        starting from the given node.

        :param node: Node to start back-substitution from
        :type node: TNode
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
    def nodes(self):
        """
        Get the nodes in the model.

        :returns: Topologically sorted list of nodes
        :rtype: list[TNode]
        """
        return self._nodes

    @property
    def cache(self) -> TCache:
        """
        Get the shared cache for the model.

        :returns: Cache instance shared by all nodes
        :rtype: TCache
        """
        return self._cache

    @property
    def arguments(self) -> TArgument:
        """
        Get the shared arguments for the model.

        :returns: Arguments instance shared by all nodes
        :rtype: TArgument
        """
        return self._arguments
