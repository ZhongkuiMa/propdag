__docformat__ = "restructuredtext"
__all__ = ["topo_sort_forward_bfs", "topo_sort_forward_dfs", "topo_sort_backward"]

from ._node import TNode


def _check_input_output_number(nodes: list[TNode], verbose: bool = False):
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

    if verbose:
        print(f"The DAG graph has {n_inputs} inputs and {n_outputs} outputs.")


def topo_sort_forward_dfs(nodes: list[TNode], verbose: bool = False) -> list[TNode]:
    """
    Perform a DFS (depth-first search) for topological sort of nodes.

    This is a DFS algorithm to sort the nodes in topological order.
    Sometimes, we need depth-first search because we want to cache the nodes close to
    the input node. If a layer close to the input has fewer dimensions.

    :param nodes: List of nodes to sort
    :type nodes: list[TNode]
    :return: Topologically sorted list of nodes
    :rtype: list[TNode]
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


def topo_sort_forward_bfs(nodes: list[TNode], verbose: bool = False) -> list[TNode]:
    """
    Perform a BFS (breadth-first search) for topological sort of nodes.

    This is a BFS algorithm to sort the nodes in topological order.
    We need breadth-first search because we do not want to cache the nodes close to
    the input node. In neural networks, a layer close to the input has more dimensions.

    :param nodes: List of nodes to sort
    :type nodes: list[TNode]
    :returns: Topologically sorted list of nodes
    :rtype: list[TNode]
    :raises ValueError: If the graph contains a cycle
    """
    _check_input_output_number(nodes, verbose)

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


def topo_sort_backward(
    nodes: list[TNode], verbose: bool = False
) -> dict[TNode, list[TNode]]:
    """
    Generate backward topological sorts for each node.

    For each node, computes a topological sort of all nodes required
    for back-substitution from that node.
    Here, the DFS (Depth-First Search) algorithm is used to traverse the graph.

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
