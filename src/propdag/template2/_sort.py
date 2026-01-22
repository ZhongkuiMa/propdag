__docformat__ = "restructuredtext"
__all__ = ["topo_sort_backward_t2", "topo_sort_forward_bfs_t2", "topo_sort_forward_dfs_t2"]

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from propdag.template2._node import T2Node


def _t2_check_input_output_number(nodes: Sequence["T2Node"], verbose: bool = False):
    """
    Verify that the reversed graph has exactly one input and one output.

    **Important**: This is called AFTER graph reversal, so:
    - "Input" node (pre_nodes=[]) = user's OUTPUT node
    - "Output" node (next_nodes=[]) = user's INPUT node

    :param nodes: Sequence of nodes in the reversed computational graph
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
        print(f"The reversed DAG has {n_inputs} inputs and {n_outputs} outputs.")


def topo_sort_forward_dfs_t2(nodes: Sequence["T2Node"], verbose: bool = False) -> list["T2Node"]:
    """
    Perform DFS topological sort on reversed graph.

    **Reversed Graph Semantics**:
    This function is called AFTER T2Model reverses the graph edges.
    The traversal direction (pre→next) remains the same as template/,
    but because the graph is reversed, we're actually sorting from
    user's output to input.

    Example:
        User builds: Input → Hidden → Output
        After reversal: Output → Hidden → Input
        This function sorts: [Output, Hidden, Input]

    :param nodes: Sequence of nodes in REVERSED graph
    :param verbose: Whether to print diagnostics
    :return: Topologically sorted list (Output → Input in user's view)
    :raises ValueError: If the graph contains a cycle
    """
    _t2_check_input_output_number(nodes, verbose)

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

    # Start from nodes with no predecessors (user's output after reversal)
    for node in nodes:
        if len(node.pre_nodes) == 0:
            dfs(node)

    if len(sorted_nodes) != len(nodes):
        raise ValueError("Graph has a cycle, cannot perform topological sort")

    # Reverse to get correct topological order (output to input in reversed graph)
    return sorted_nodes[::-1]


def topo_sort_backward_t2(
    nodes: Sequence["T2Node"], verbose: bool = False
) -> dict["T2Node", list["T2Node"]]:
    """
    Generate backward topological sorts for each node in reversed graph.

    For each node, computes a topological sort of all nodes required
    for back-substitution from that node. Uses DFS traversal.

    In the reversed graph context, this traverses backward through pre_nodes
    which point toward the user's output (graph input).

    :param nodes: Sequence of nodes in the REVERSED computational graph
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

    backward_sorts: dict[T2Node, list[T2Node]] = {}
    for node in nodes:
        visited: set[T2Node] = set()
        result: list[T2Node] = []
        dfs(node, visited, result)
        backward_sorts[node] = result  # already in correct topological order

    for node, backward_sort in backward_sorts.items():
        backward_sorts[node] = backward_sort[::-1]  # reverse to get correct order

    return backward_sorts


def topo_sort_forward_bfs_t2(nodes: Sequence["T2Node"], verbose: bool = False) -> list["T2Node"]:
    """
    Perform BFS topological sort on reversed graph.

    **Reversed Graph Semantics**:
    This function is called AFTER T2Model reverses the graph edges.
    BFS traverses layer-by-layer from the "input" (user's output) to
    "output" (user's input) in the reversed graph.

    Example:
        User builds: Input → Hidden → Output
        After reversal: Output → Hidden → Input
        BFS processes: [Output] → [Hidden] → [Input]

    :param nodes: Sequence of nodes in REVERSED graph
    :param verbose: Whether to print diagnostics
    :return: Topologically sorted list (Output → Input in user's view)
    :raises ValueError: If the graph contains a cycle
    """
    _t2_check_input_output_number(nodes, verbose)

    # Use unique pre_nodes count to handle cases like x * x where
    # the same input appears twice in pre_nodes
    in_degrees: dict[T2Node, int] = {node: len(set(node.pre_nodes)) for node in nodes}

    # Start from nodes with in_degree 0 (user's output after reversal)
    queue: list[T2Node] = [node for node in nodes if in_degrees[node] == 0]
    sorted_nodes: list[T2Node] = []

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
