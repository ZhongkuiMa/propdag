__docformat__ = "restructuredtext"
__all__ = ["TModel"]


from abc import ABC

from ..utils import *
from ._arguments import TArgument
from ._cache import TCache
from ._node import TNode


def _topo_sort_forward(nodes: list[TNode]) -> list[TNode]:
    """
    This is a breadth-first search algorithm to sort the nodes in topological order.
    We need breadth-first search because we do not want to cache the nodes close to
    the input node. In neural networks, a layer close to the input has more dimensions.
    """
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


def _clear_fwd_cache(cache_counter: dict[TNode, int], nodes: list[TNode]):
    for node in nodes:
        cache_counter[node] -= 1
        if cache_counter[node] == 0:
            node.clear_fwd_cache()
            del cache_counter[node]


def _clear_bwd_cache(cache_counter: dict[TNode, int], nodes: list[TNode]):
    for node in nodes:
        if node in cache_counter:
            # Some next nodes may not be involved in the backward pass.
            cache_counter[node] -= 1
            if cache_counter[node] == 0:
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

    def __init__(self, nodes: list[TNode]):
        self._nodes = _topo_sort_forward(nodes)
        self._cache = nodes[0].cache
        self._arguments = nodes[0].argument
        for node in nodes[1:]:
            assert node.cache == self._cache
            assert node.argument == self._arguments

        if self.arguments.prop_mode == PropMode.BACKWARD:
            self._all_backward_sorts = _topo_sort_backward(self.nodes)

    def run(self):
        cache_counter = {node: len(node.next_nodes) for node in self._nodes}
        cache_counter[self.nodes[-1]] = 1  # For the last node

        node = self.nodes[0]
        node.forward()

        if self.arguments.prop_mode == PropMode.BACKWARD:
            self.backsub(node)

        for i in range(1, len(self.nodes)):
            node = self.nodes[i]
            node.forward()

            if self.arguments.prop_mode == PropMode.BACKWARD:
                self.backsub(node)

            _clear_fwd_cache(cache_counter, node.pre_nodes)

        _clear_fwd_cache(cache_counter, [node])

    def backsub(self, node: TNode):
        backward_sort = self._all_backward_sorts[node]
        cache_counter = {node: len(node.pre_nodes) for node in backward_sort}
        cache_counter[self.nodes[0]] = 1  # For the input node

        node.backward()

        for j in range(1, len(backward_sort)):
            node = backward_sort[j]
            node.backward()
            _clear_bwd_cache(cache_counter, node.next_nodes)

        _clear_bwd_cache(cache_counter, [node])

    @property
    def nodes(self):
        return self._nodes

    @property
    def cache(self) -> TCache:
        return self._cache

    @property
    def arguments(self) -> TArgument:
        return self._arguments
