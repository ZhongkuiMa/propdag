__docformat__ = "restructuredtext"
__all__ = [
    "T2Model",
    "clear_bwd_cache_t2",
    "reverse_dag",
]

from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from propdag.template2._arguments import T2Argument
from propdag.template2._cache import T2Cache
from propdag.template2._sort import (
    topo_sort_backward_t2,
    topo_sort_forward_bfs_t2,
    topo_sort_forward_dfs_t2,
)

if TYPE_CHECKING:
    from propdag.template2._node import T2Node


def reverse_dag(user_nodes: Sequence["T2Node"], verbose: bool = False) -> tuple["T2Node", "T2Node"]:
    """
    Validate and reverse the user's computational graph.

    **CRITICAL OPERATION**: This function reverses all edges in-place by swapping
    pre_nodes ↔ next_nodes. After this operation:
    - User's OUTPUT node becomes the graph INPUT (pre_nodes=[])
    - User's INPUT node becomes the graph OUTPUT (next_nodes=[])

    :param user_nodes: Sequence of nodes in user's forward graph
    :param verbose: Whether to print diagnostic messages
    :return: Tuple of (user_input, user_output) from the original forward graph
    :raises ValueError: If graph doesn't have exactly one input and one output
    """
    # STEP 1: Validate user's forward graph (BEFORE reversal)
    user_input_nodes = [n for n in user_nodes if len(n.pre_nodes) == 0]
    user_output_nodes = [n for n in user_nodes if len(n.next_nodes) == 0]

    if len(user_input_nodes) == 0:
        raise ValueError(
            "User graph must have exactly one input node (no predecessors), but found zero."
        )
    if len(user_input_nodes) > 1:
        raise ValueError(
            f"User graph must have exactly one input node, but found "
            f"{len(user_input_nodes)} nodes: {[n.name for n in user_input_nodes]}"
        )
    if len(user_output_nodes) == 0:
        raise ValueError(
            "User graph must have exactly one output node (no successors), but found zero."
        )
    if len(user_output_nodes) > 1:
        raise ValueError(
            f"User graph must have exactly one output node, but found "
            f"{len(user_output_nodes)} nodes: {[n.name for n in user_output_nodes]}"
        )

    user_input = user_input_nodes[0]
    user_output = user_output_nodes[0]

    if verbose:
        print(f"Before reversal: Input={user_input.name}, Output={user_output.name}")

    # STEP 2: REVERSE ALL EDGES (CORE INNOVATION!)
    for node in user_nodes:
        # Swap pre_nodes ↔ next_nodes in-place
        node._pre_nodes, node._next_nodes = node._next_nodes, node._pre_nodes  # noqa: SLF001

    if verbose:
        print(f"Reversed edges: {user_output.name} becomes input (pre_nodes=[])")
        print(f"                {user_input.name} becomes output (next_nodes=[])")

    return user_input, user_output


def clear_bwd_cache_t2(cache_counter: dict["T2Node", int], nodes: Sequence["T2Node"]):
    """
    Clear backward caches (symbolic bounds) for nodes when they are no longer needed.

    T2Model only clears backward cache (symbnds), not forward bounds (bnds).
    This preserves the computed bounds while freeing symbolic expressions.

    :param cache_counter: Dictionary tracking how many next nodes still need each node's cache
    :param nodes: List of nodes whose cache counters to decrement
    """
    for node in set(nodes):  # Use set to handle duplicate nodes
        cache_counter[node] -= 1
        if cache_counter[node] <= 0:  # The output node (user's input) will be -1
            node.clear_bwd_cache()
            del cache_counter[node]


class T2Model(ABC):
    """
    Reversed graph model for backward bound propagation.

    **KEY INNOVATION**: Automatically reverses user-provided graph edges so
    "forward" propagation through the reversed graph achieves backward bound
    propagation without semantic confusion.

    **User Experience**:
    - User builds: Input → Hidden → Output (normal graph construction)
    - T2Model reverses to: Output → Hidden → Input (internal representation)
    - run() propagates "forward": Output → Input (backward propagation!)

    **Comparison to TModel**:
    - TModel: Has multiple modes (FORWARD/BACKWARD)
    - T2Model: Single-purpose (backward bound propagation only)
    - TModel: backward() goes backward through forward graph
    - T2Model: forward() goes forward through reversed graph (clear!)

    **Graph Constraints**:
    - Must have exactly one input and one output node (before reversal)
    - No cycles allowed
    - All nodes must share same cache and arguments

    :ivar _nodes: Topologically sorted nodes (Output → Input after reversal)
    :ivar _sort_strategy: Sorting strategy used (dfs or bfs)
    :ivar _cache: Shared cache across all nodes
    :ivar _arguments: Shared arguments across all nodes
    :ivar verbose: Whether to print execution diagnostics
    :ivar clear_cache_during_running: Whether to clear caches during execution
    """

    _nodes: list["T2Node"]
    _sort_strategy: Literal["dfs", "bfs"]
    _cache: T2Cache
    _arguments: T2Argument
    _user_input: "T2Node"
    _user_output: "T2Node"
    _all_backward_sorts: dict["T2Node", list["T2Node"]]

    verbose: bool
    clear_cache_during_running: bool

    def __init__(
        self,
        user_nodes: Sequence["T2Node"],
        sort_strategy: Literal["dfs", "bfs"] = "bfs",
        verbose: bool = False,
        clear_cache_during_running: bool = False,
    ):
        """
        Initialize reversed graph model.

        **CRITICAL**: This method reverses the user-provided graph edges!

        Process:
        1. Validate user's forward graph (Input → Output)
        2. REVERSE ALL EDGES (swap pre_nodes ↔ next_nodes)
        3. Topological sort from new "input" (user's output)
        4. Verify reversal (first node = user's output, last = user's input)

        :param user_nodes: Nodes in user's forward graph (Input → Output)
        :param sort_strategy: Topological sort strategy (dfs or bfs)
        :param verbose: Enable verbose output during execution
        :param clear_cache_during_running: Clear caches during execution
        :raises ValueError: If graph doesn't have exactly one input and one output
        :raises ValueError: If graph has cycles
        :raises AssertionError: If not all nodes share the same cache and arguments
        """
        self.verbose = verbose
        self.clear_cache_during_running = clear_cache_during_running
        self._sort_strategy = sort_strategy

        # STEP 1 & 2: Validate and reverse graph edges
        self._user_input, self._user_output = reverse_dag(user_nodes, self.verbose)
        user_input, user_output = self._user_input, self._user_output

        # STEP 3: Topological sort from new "input" (user's output)
        if sort_strategy == "dfs":
            self._nodes = topo_sort_forward_dfs_t2(list(user_nodes), self.verbose)
        elif sort_strategy == "bfs":
            self._nodes = topo_sort_forward_bfs_t2(list(user_nodes), self.verbose)
        else:
            raise ValueError(f"Unknown sort strategy: {sort_strategy}")

        # STEP 4: Verify reversal succeeded
        assert self._nodes[0] == user_output, (
            f"First node after reversal should be user's output, "
            f"but got {self._nodes[0].name} (expected {user_output.name})"
        )
        assert self._nodes[-1] == user_input, (
            f"Last node after reversal should be user's input, "
            f"but got {self._nodes[-1].name} (expected {user_input.name})"
        )

        if self.verbose:
            print(f"Topological order (reversed): {[n.name for n in self._nodes]}")

        # Validate shared cache/arguments
        self._cache = user_nodes[0].cache
        self._arguments = user_nodes[0].argument
        for node in user_nodes[1:]:
            assert node.cache == self._cache, "All nodes must share same cache"
            assert node.argument == self._arguments, "All nodes must share same arguments"

        # Compute backward sorts for backsub (from each node back to Output/graph input)
        # In the reversed graph, pre_nodes point toward the graph input (user's output)
        self._all_backward_sorts = topo_sort_backward_t2(self._nodes, self.verbose)

    def run(self, *args, **kwargs):
        """
        Execute backward bound propagation via forward traversal.

        **SIMPLIFIED vs TModel**: No mode switching, no backsub, no backprop_bounds.
        Just a single forward pass through the reversed graph.

        Because the graph is reversed:
        - First node (user's output): Initialize bounds from output constraints
        - Middle nodes: Propagate bounds using inverse relaxations
        - Last node (user's input): Final tightened bounds

        Process at each node:
        1. forward(): Build inverse relaxations and propagate bounds
        2. Optionally clear cache when no longer needed

        :param args: Positional arguments (unused, for future extensions)
        :param kwargs: Keyword arguments (unused, for future extensions)
        """
        cache_counter = {node: len(node.next_nodes) for node in self._nodes}

        # Process all nodes in topological order (Output → Input in user's view)
        for i, node in enumerate(self._nodes):
            if self.verbose:
                print(f"Propagate bounds through {node.name}")

            node.forward()  # Actually does backward propagation!

            # Clear predecessor caches when no longer needed
            if self.clear_cache_during_running and i > 0:
                clear_bwd_cache_t2(cache_counter, node.pre_nodes)

        # Clear final node cache
        if self.clear_cache_during_running:
            clear_bwd_cache_t2(cache_counter, [self._nodes[-1]])

    @property
    def sort_strategy(self):
        """
        Get the sorting strategy used in the model.

        :returns: Sorting strategy (either 'dfs' or 'bfs')
        """
        return self._sort_strategy

    @property
    def nodes(self) -> list["T2Node"]:
        """
        Get the nodes in REVERSED topological order.

        **IMPORTANT**: These are in reversed graph order (Output → Input in user's view).

        :returns: Topologically sorted list of nodes in reversed graph
        """
        return self._nodes

    @property
    def cache(self) -> T2Cache:
        """
        Get the shared cache for the model.

        :returns: Cache instance shared by all nodes
        """
        return self._cache

    @property
    def arguments(self) -> T2Argument:
        """
        Get the shared arguments for the model.

        :returns: Arguments instance shared by all nodes
        """
        return self._arguments

    @property
    def user_input(self) -> "T2Node":
        """
        Get the original input node from user's forward graph.

        **IMPORTANT**: This is the node that was the INPUT before graph reversal.
        After reversal, this becomes the OUTPUT node in the reversed graph.

        :returns: User's input node (before reversal)
        """
        return self._user_input

    @property
    def user_output(self) -> "T2Node":
        """
        Get the original output node from user's forward graph.

        **IMPORTANT**: This is the node that was the OUTPUT before graph reversal.
        After reversal, this becomes the INPUT node in the reversed graph.

        :returns: User's output node (before reversal)
        """
        return self._user_output
