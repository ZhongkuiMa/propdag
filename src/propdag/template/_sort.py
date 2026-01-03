__docformat__ = "restructuredtext"
__all__ = ["topo_sort_backward", "topo_sort_forward_bfs", "topo_sort_forward_dfs"]

from collections.abc import Sequence

from propdag.custom_types import NodeType


def _check_input_output_number(nodes: Sequence[NodeType], verbose: bool = False):
    """
    Verify that the graph has exactly one input node and one output node.

    :param nodes: Sequence of nodes in the computational graph
    :param verbose: Whether to print diagnostics
    :raises ValueError: When number of input or output nodes is not equal to 1
    """
    n_inputs = 0
    n_outputs = 0
    for node in nodes:
        if len(node.pre_nodes) == 0:
            n_inputs += 1
        if len(node.next_nodes) == 0:
            n_outputs += 1

    if verbose:
        print(f"The DAG graph has {n_inputs} inputs and {n_outputs} outputs.")


def topo_sort_forward_dfs(nodes: Sequence[NodeType], verbose: bool = False) -> list[NodeType]:
    """
    Perform a DFS (depth-first search) for topological sort of nodes.

    This is a DFS algorithm to sort the nodes in topological order.
    Sometimes, we need depth-first search because we want to cache the nodes close to
    the input node. If a layer close to the input has fewer dimensions.

    :param nodes: Sequence of nodes to sort
    :param verbose: Whether to print diagnostics
    :return: Topologically sorted list of nodes
    :raises ValueError: If the graph contains a cycle
    """
    _check_input_output_number(nodes, verbose)

    visited = set()
    temp_mark = set()
    sorted_nodes = []

    def dfs(node):
        if node in temp_mark:
            raise ValueError("Graph has a cycle, cannot perform topological sort")
        if node not in visited:
            temp_mark.add(node)
            for next_node in node.next_nodes:
                dfs(next_node)
            temp_mark.remove(node)
            visited.add(node)
            sorted_nodes.append(node)

    for node in nodes:
        if len(node.pre_nodes) == 0:
            dfs(node)

    if len(sorted_nodes) != len(nodes):
        raise ValueError("Graph has a cycle, cannot perform topological sort")

    # Reverse to get correct topological order (from input to output)
    return sorted_nodes[::-1]


def topo_sort_forward_bfs(nodes: Sequence[NodeType], verbose: bool = False) -> list[NodeType]:
    """
    Perform a BFS (breadth-first search) for topological sort of nodes.

    This is a BFS algorithm to sort the nodes in topological order.
    We need breadth-first search because we do not want to cache the nodes close to
    the input node. In neural networks, a layer close to the input has more dimensions.

    :param nodes: Sequence of nodes to sort
    :param verbose: Whether to print diagnostics
    :return: Topologically sorted list of nodes
    :raises ValueError: If the graph contains a cycle
    """
    _check_input_output_number(nodes, verbose)

    # Use unique pre_nodes count to handle cases like x * x where
    # the same input appears twice in pre_nodes
    in_degrees: dict[NodeType, int] = {node: len(set(node.pre_nodes)) for node in nodes}

    queue: list[NodeType] = [node for node in nodes if in_degrees[node] == 0]
    sorted_nodes: list[NodeType] = []
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


def topo_sort_backward(
    nodes: Sequence[NodeType], verbose: bool = False
) -> dict[NodeType, list[NodeType]]:
    """
    Generate backward topological sorts for each node.

    For each node, computes a topological sort of all nodes required
    for back-substitution from that node.
    Here, the DFS (Depth-First Search) algorithm is used to traverse the graph.

    :param nodes: Sequence of nodes in the computational graph
    :param verbose: Whether to print diagnostics
    :return: Dictionary mapping each node to its backward topological sort
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
        visited: set[NodeType] = set()
        result: list[NodeType] = []
        dfs(node, visited, result)
        backward_sorts[node] = result  # already in correct topological order

    for node, backward_sort in backward_sorts.items():
        backward_sorts[node] = backward_sort[::-1]  # reverse to get the correct order

    # THE FOLLOWING IS FOR DEBUGGING PURPOSES
    # for node, backward_sort in backward_sorts.items():
    #     print(node)
    #     print([n.name for n in backward_sort])

    return backward_sorts
