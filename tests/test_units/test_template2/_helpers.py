"""Helpers for building T2 (template2/toy2) test models with minimal boilerplate."""

__docformat__ = "restructuredtext"

from collections.abc import Sequence

from propdag import Toy2Argument, Toy2Cache, Toy2Model, Toy2Node


def _new_cache_args_t2() -> tuple[Toy2Cache, Toy2Argument]:
    """Return a fresh (cache, arguments) pair with fwd_bnds seeded for Node-1."""
    cache = Toy2Cache()
    cache.fwd_bnds["Node-1"] = ("input bounds",)
    return cache, Toy2Argument()


def make_nodes_t2(
    count: int,
    cache: Toy2Cache,
    arguments: Toy2Argument,
    *,
    start: int = 1,
    prefix: str = "Node",
) -> list[Toy2Node]:
    """Create ``count`` Toy2Node instances named ``f'{prefix}-{i}'`` starting at ``start``."""
    return [Toy2Node(f"{prefix}-{i}", cache, arguments) for i in range(start, start + count)]


def build_chain_nodes_t2(length: int) -> tuple[Toy2Cache, list[Toy2Node]]:
    """Build a linear chain of Toy2Nodes WITHOUT constructing a model.

    Useful for tests that want to call ``topo_sort_*_t2`` directly on the
    user-orientation graph (Toy2Model construction reverses the edges).

    :param length: number of nodes in the chain.

    :return: ``(cache, nodes)`` with edges pre_nodes/next_nodes wired
    """
    cache, arguments = _new_cache_args_t2()
    nodes = make_nodes_t2(length, cache, arguments)
    for i in range(length - 1):
        nodes[i].next_nodes = [nodes[i + 1]]
        nodes[i + 1].pre_nodes = [nodes[i]]
    return cache, nodes


def build_chain_model_t2(
    length: int,
    *,
    sort_strategy: str = "bfs",
    verbose: bool = False,
    clear_cache_during_running: bool = False,
) -> tuple[Toy2Model, Toy2Cache, list[Toy2Node]]:
    """Build a linear chain Node-1 -> ... -> Node-N as a Toy2Model.

    :param length: number of nodes.

    :param sort_strategy: "bfs" or "dfs".

    :return: ``(model, cache, nodes)``
    """
    cache, arguments = _new_cache_args_t2()
    nodes = make_nodes_t2(length, cache, arguments)
    for i in range(length - 1):
        nodes[i].next_nodes = [nodes[i + 1]]
        nodes[i + 1].pre_nodes = [nodes[i]]
    model = Toy2Model(
        nodes,
        sort_strategy=sort_strategy,
        verbose=verbose,
        clear_cache_during_running=clear_cache_during_running,
    )
    return model, cache, nodes


def build_diamond_model_t2(
    *,
    sort_strategy: str = "bfs",
    verbose: bool = False,
    clear_cache_during_running: bool = False,
) -> tuple[Toy2Model, Toy2Cache, list[Toy2Node]]:
    """Build the canonical diamond Node-1 -> {Node-2, Node-3} -> Node-4 as a Toy2Model."""
    cache, arguments = _new_cache_args_t2()
    n1, n2, n3, n4 = make_nodes_t2(4, cache, arguments)
    n1.next_nodes = [n2, n3]
    n2.pre_nodes = [n1]
    n2.next_nodes = [n4]
    n3.pre_nodes = [n1]
    n3.next_nodes = [n4]
    n4.pre_nodes = [n2, n3]
    model = Toy2Model(
        [n1, n2, n3, n4],
        sort_strategy=sort_strategy,
        verbose=verbose,
        clear_cache_during_running=clear_cache_during_running,
    )
    return model, cache, [n1, n2, n3, n4]


def build_y_model_t2(
    *, sort_strategy: str = "bfs", **model_kwargs
) -> tuple[Toy2Model, Toy2Cache, list[Toy2Node]]:
    """Build a Y-shape Node-1 -> {Node-2, Node-3} -> Node-4 (alias of diamond at 4 nodes)."""
    return build_diamond_model_t2(sort_strategy=sort_strategy, **model_kwargs)


def build_skip_model_t2(
    *,
    sort_strategy: str = "bfs",
    verbose: bool = False,
    clear_cache_during_running: bool = False,
) -> tuple[Toy2Model, Toy2Cache, list[Toy2Node]]:
    """Build a skip connection used by t2 tests: Node-1 -> Node-2 -> {Node-3, Node-4}; Node-3 -> Node-4."""
    cache, arguments = _new_cache_args_t2()
    n1, n2, n3, n4 = make_nodes_t2(4, cache, arguments)
    n1.next_nodes = [n2]
    n2.pre_nodes = [n1]
    n2.next_nodes = [n3, n4]
    n3.pre_nodes = [n2]
    n3.next_nodes = [n4]
    n4.pre_nodes = [n2, n3]
    model = Toy2Model(
        [n1, n2, n3, n4],
        sort_strategy=sort_strategy,
        verbose=verbose,
        clear_cache_during_running=clear_cache_during_running,
    )
    return model, cache, [n1, n2, n3, n4]


def build_invalid_io_nodes_t2(scenario: str) -> Sequence[Toy2Node]:
    """Build invalid input/output topologies by name (no model construction).

    Used by parametrized error tests that pass the result to ``Toy2Model`` and
    assert ``ValueError``. Supported scenarios:
    ``multi_input``, ``multi_output``, ``no_input``, ``two_node_cycle``, ``three_node_cycle``.
    """
    cache, arguments = _new_cache_args_t2()
    if scenario == "multi_input":
        cache.fwd_bnds["Node-2"] = ("input bounds",)
        n1, n2, n3 = make_nodes_t2(3, cache, arguments)
        n1.next_nodes = [n3]
        n2.next_nodes = [n3]
        n3.pre_nodes = [n1, n2]
        return [n1, n2, n3]
    if scenario == "multi_output":
        n1, n2, n3 = make_nodes_t2(3, cache, arguments)
        n1.next_nodes = [n2, n3]
        n2.pre_nodes = [n1]
        n3.pre_nodes = [n1]
        return [n1, n2, n3]
    if scenario == "no_input":
        n1, n2 = make_nodes_t2(2, cache, arguments)
        n1.next_nodes = [n2]
        n1.pre_nodes = [n2]
        n2.next_nodes = [n1]
        n2.pre_nodes = [n1]
        return [n1, n2]
    if scenario == "two_node_cycle":
        n1, n2 = make_nodes_t2(2, cache, arguments)
        n1.next_nodes = [n2]
        n1.pre_nodes = [n2]
        n2.next_nodes = [n1]
        n2.pre_nodes = [n1]
        return [n1, n2]
    if scenario == "three_node_cycle":
        n1, n2, n3 = make_nodes_t2(3, cache, arguments)
        n1.next_nodes = [n2]
        n1.pre_nodes = [n3]
        n2.next_nodes = [n3]
        n2.pre_nodes = [n1]
        n3.next_nodes = [n1]
        n3.pre_nodes = [n2]
        return [n1, n2, n3]
    raise ValueError(f"Unknown invalid-topology scenario: {scenario}")
