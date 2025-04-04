__docformat__ = "restructuredtext"
__all__ = ["TModel"]


from abc import ABC

from propdag.utils import *
from .arguments import TArguments
from .cache import TCache
from .node import TNode


def _topo_sort_forward(nodes: list[TNode]) -> list[TNode]:
    visited = []
    stack = []

    def visit(node: TNode):
        if node in visited:
            return
        visited.append(node)
        for next_node in node.next_nodes:
            visit(next_node)
        stack.append(node)

    for node in nodes:
        visit(node)

    return stack[::-1]


def _topo_sort_backward(nodes: list[TNode]) -> dict[TNode, list[TNode]]:
    """

    :param nodes: The nodes in topological order.
    :return:
    """

    backward_sorts = {}
    for node in nodes:
        # Find all nodes can reach this node.
        # Iterate all previous nodes as a set.
        backward_sort = [node]
        queue = [node]
        while True:
            if not queue:
                break
            current_node = queue.pop()
            for pre_node in current_node.pre_nodes:
                if pre_node not in backward_sort:
                    backward_sort.append(pre_node)
                    queue.append(pre_node)

        # Because we use list and the reversed nodes, we do not need to order the nodes.
        backward_sorts[node] = backward_sort

    return backward_sorts


def _clear_forward_cache(cache_counter: dict[TNode, int], nodes: list[TNode]):
    for node in nodes:
        cache_counter[node] -= 1
        # If the node is the last node, the cache_counter will be -1.
        if cache_counter[node] in {0, -1}:
            node.clear_forward_cache()
            del cache_counter[node]


def _clear_backward_cache(cache_counter: dict[TNode, int], nodes: list[TNode]):
    for node in nodes:
        # For backward cache, if we use next_nodes, the next node may not in the
        # backward sort, so we need to accept the node is not in the cache_counter.
        counter = cache_counter.get(node)
        if counter is not None:
            cache_counter[node] = counter - 1
            if cache_counter[node] <= 0:
                node.clear_backward_cache()


class TModel(ABC):
    """
    This is a computation graph template. The overall logic is the whole graph and all
    nodes shares the same cache, their methods will operate on this cache. We do not
    consider subgraph, so there is no nested graphs.
    """

    def __init__(self, nodes: list[TNode]):
        self._nodes = nodes
        self._all_backward_sorts = None
        self._cache = None
        self._arguments = None

    def prepare(self, cache: TCache, arguments: TArguments):
        self._nodes = _topo_sort_forward(self._nodes)
        self._cache = cache
        self._arguments = arguments
        for node in self._nodes:
            node.cache = cache
            node.arguments = arguments

        if self.arguments.prop_mode == PropMode.BACKWARD:
            self._all_backward_sorts = _topo_sort_backward(self.nodes)

    def run(self):
        cache_counter = {node: len(node.next_nodes) for node in self._nodes}

        node = self.nodes[0]
        node.forward()

        if self.arguments.prop_mode == PropMode.BACKWARD:
            self._backsub(node)

        for i in range(1, len(self.nodes)):
            node = self.nodes[i]
            node.forward()

            if self.arguments.prop_mode == PropMode.BACKWARD:
                self._backsub(node)

            _clear_forward_cache(cache_counter, node.pre_nodes)

        _clear_forward_cache(cache_counter, [node])

    def _backsub(self, node: TNode):
        backward_sort = self._all_backward_sorts[node]
        cache_counter = {node: len(node.pre_nodes) for node in backward_sort}

        node.backward()

        for j in range(1, len(backward_sort)):
            node = backward_sort[j]
            node.backward()
            _clear_backward_cache(cache_counter, node.next_nodes)

        _clear_backward_cache(cache_counter, [node])

    @property
    def nodes(self):
        return self._nodes

    @property
    def cache(self) -> TCache:
        return self._cache

    @property
    def arguments(self) -> TArguments:
        return self._arguments
