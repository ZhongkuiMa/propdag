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
    This is a breadth-first search algorithm to sort the nodes in topological order.
    We need breadth-first search because we do not want to cache the nodes close to
    the input node. In neural networks, a layer close to the input has more dimensions.
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
    backward_sorts = {}
    for node in nodes:
        # Find all nodes can reach this node.
        # Iterate all previous nodes as a set.
        backward_sort = [node]
        queue = [node]
        while True:
            if not queue:
                break
            current_node = queue.pop(0)
            for pre_node in current_node.pre_nodes:
                if pre_node not in backward_sort:
                    backward_sort.append(pre_node)
                    queue.append(pre_node)

        # Because we use list and the reversed nodes, we do not need to order the nodes.
        backward_sorts[node] = backward_sort

    return backward_sorts


def clear_fwd_cache(cache_counter: dict[TNode, int], nodes: list[TNode]):
    for node in nodes:
        cache_counter[node] -= 1
        if cache_counter[node] <= 0:  # The output node will be -1
            node.clear_fwd_cache()
            del cache_counter[node]


def clear_bwd_cache(cache_counter: dict[TNode, int], nodes: list[TNode]):
    for node in nodes:
        if node in cache_counter:
            # Some next nodes may not be involved in the backward pass.
            cache_counter[node] -= 1
            if cache_counter[node] <= 0:  # The input node will be -1
                node.clear_bwd_cache()
                del cache_counter[node]


class TModel(ABC):
    """
    This is a computation graph template. The overall logic is the whole graph and all
    nodes shares the same cache, their methods will operate on this cache. We do not
    consider subgraph, so there is no nested graphs.
    """

    _nodes: list[TNode]
    _cache: TCache
    _arguments: TArgument
    _all_backward_sorts: dict[TNode, list[TNode]]

    def __init__(self, nodes: list[TNode], verbose: bool = False):
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
        return self._nodes

    @property
    def cache(self) -> TCache:
        return self._cache

    @property
    def arguments(self) -> TArgument:
        return self._arguments
