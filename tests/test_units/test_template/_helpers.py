"""Helpers for building T1 (template/toy) test models with minimal boilerplate."""

__docformat__ = "restructuredtext"

from collections.abc import Sequence

from propdag import (
    BackwardToyNode,
    ForwardToyNode,
    PropMode,
    ToyArgument,
    ToyCache,
    ToyModel,
)


def make_node(
    name: str,
    cache: ToyCache,
    arguments: ToyArgument,
    prop_mode: PropMode = PropMode.FORWARD,
) -> ForwardToyNode | BackwardToyNode:
    """Construct a ForwardToyNode or BackwardToyNode based on prop_mode.

    :param name: node name.

    :param cache: shared ToyCache.

    :param arguments: shared ToyArgument.

    :param prop_mode: propagation mode controlling node concrete class.

    :return: a node of the appropriate concrete class
    """
    cls = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
    return cls(name, cache, arguments)


def make_nodes(
    count: int,
    cache: ToyCache,
    arguments: ToyArgument,
    prop_mode: PropMode = PropMode.FORWARD,
    *,
    start: int = 1,
    prefix: str = "Node",
) -> list[ForwardToyNode | BackwardToyNode]:
    """Create ``count`` nodes named ``f'{prefix}-{i}'`` starting at ``start``.

    :param count: number of nodes to create.

    :param cache: shared ToyCache.

    :param arguments: shared ToyArgument.

    :param prop_mode: propagation mode for the concrete class.

    :param start: first numeric suffix (default 1).

    :param prefix: name prefix (default "Node").

    :return: list of newly constructed nodes
    """
    return [
        make_node(f"{prefix}-{i}", cache, arguments, prop_mode) for i in range(start, start + count)
    ]


def _new_cache_args(prop_mode: PropMode) -> tuple[ToyCache, ToyArgument]:
    """Return a fresh (cache, arguments) pair with input bounds seeded for Node-1."""
    cache = ToyCache()
    cache.bnds["Node-1"] = ("input bounds",)
    return cache, ToyArgument(prop_mode=prop_mode)


def build_chain_model(
    length: int,
    *,
    sort_strategy: str = "bfs",
    prop_mode: PropMode = PropMode.FORWARD,
    verbose: bool = False,
    clear_cache_during_running: bool = False,
) -> tuple[ToyModel, ToyCache, list]:
    """Build a linear chain Node-1 -> ... -> Node-N as a ToyModel.

    :param length: number of nodes in the chain.

    :param sort_strategy: "bfs" or "dfs".

    :param prop_mode: propagation mode.

    :param verbose: forwarded to ToyModel.

    :param clear_cache_during_running: forwarded to ToyModel.

    :return: ``(model, cache, nodes)`` for assertions
    """
    cache, arguments = _new_cache_args(prop_mode)
    nodes = make_nodes(length, cache, arguments, prop_mode)
    for i in range(length - 1):
        nodes[i].next_nodes = [nodes[i + 1]]
        nodes[i + 1].pre_nodes = [nodes[i]]
    model = ToyModel(
        nodes,
        sort_strategy=sort_strategy,
        verbose=verbose,
        clear_cache_during_running=clear_cache_during_running,
    )
    return model, cache, nodes


def build_diamond_model(
    *,
    sort_strategy: str = "bfs",
    prop_mode: PropMode = PropMode.FORWARD,
    verbose: bool = False,
    clear_cache_during_running: bool = False,
) -> tuple[ToyModel, ToyCache, list]:
    """Build the canonical diamond Node-1 -> {Node-2, Node-3} -> Node-4 as a ToyModel.

    :return: ``(model, cache, nodes)``
    """
    cache, arguments = _new_cache_args(prop_mode)
    n1, n2, n3, n4 = make_nodes(4, cache, arguments, prop_mode)
    n1.next_nodes = [n2, n3]
    n2.pre_nodes = [n1]
    n2.next_nodes = [n4]
    n3.pre_nodes = [n1]
    n3.next_nodes = [n4]
    n4.pre_nodes = [n2, n3]
    model = ToyModel(
        [n1, n2, n3, n4],
        sort_strategy=sort_strategy,
        verbose=verbose,
        clear_cache_during_running=clear_cache_during_running,
    )
    return model, cache, [n1, n2, n3, n4]


def build_y_model(
    *,
    sort_strategy: str = "bfs",
    prop_mode: PropMode = PropMode.FORWARD,
    **model_kwargs,
) -> tuple[ToyModel, ToyCache, list]:
    """Build a Y-shape Node-1 -> {Node-2, Node-3} -> Node-4.

    Equivalent to the 4-node diamond; kept as a separate name to read more
    naturally in tests that talk about Y topology vs diamond topology.
    """
    return build_diamond_model(sort_strategy=sort_strategy, prop_mode=prop_mode, **model_kwargs)


def build_skip_model(
    *,
    sort_strategy: str = "bfs",
    prop_mode: PropMode = PropMode.FORWARD,
    verbose: bool = False,
    clear_cache_during_running: bool = False,
) -> tuple[ToyModel, ToyCache, list]:
    """Build a single skip connection: Node-1 -> {Node-2, Node-4}; Node-2 -> Node-3 -> Node-4."""
    cache, arguments = _new_cache_args(prop_mode)
    n1, n2, n3, n4 = make_nodes(4, cache, arguments, prop_mode)
    n1.next_nodes = [n2, n4]
    n2.pre_nodes = [n1]
    n2.next_nodes = [n3]
    n3.pre_nodes = [n2]
    n3.next_nodes = [n4]
    n4.pre_nodes = [n1, n3]
    model = ToyModel(
        [n1, n2, n3, n4],
        sort_strategy=sort_strategy,
        verbose=verbose,
        clear_cache_during_running=clear_cache_during_running,
    )
    return model, cache, [n1, n2, n3, n4]


def build_cycle_nodes(
    scenario: str,
    *,
    prop_mode: PropMode = PropMode.FORWARD,
) -> Sequence[ForwardToyNode | BackwardToyNode]:
    """Build a cyclic-or-otherwise-invalid topology by name (no model construction).

    Intended for parametrized error tests that pass the result to ``ToyModel``
    and assert ``ValueError`` is raised.

    Supported scenarios: ``two_node``, ``three_node``, ``self_loop``, ``embedded``,
    ``multi_input``, ``multi_output``, ``no_input``, ``no_output``.

    :param scenario: topology name.

    :param prop_mode: propagation mode for the constructed nodes.

    :return: list of nodes ready to pass to ``ToyModel``
    """
    cache, arguments = _new_cache_args(prop_mode)
    if scenario == "two_node":
        n1, n2 = make_nodes(2, cache, arguments, prop_mode)
        n1.next_nodes = [n2]
        n1.pre_nodes = [n2]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n1]
        return [n1, n2]
    if scenario == "three_node":
        n1, n2, n3 = make_nodes(3, cache, arguments, prop_mode)
        n1.next_nodes = [n2]
        n1.pre_nodes = [n3]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n3]
        n3.pre_nodes = [n2]
        n3.next_nodes = [n1]
        return [n1, n2, n3]
    if scenario == "self_loop":
        n1, n2 = make_nodes(2, cache, arguments, prop_mode)
        n1.next_nodes = [n2]
        n2.pre_nodes = [n1, n2]
        n2.next_nodes = [n2]
        return [n1, n2]
    if scenario == "embedded":
        n1, n2, n3 = make_nodes(3, cache, arguments, prop_mode)
        n1.next_nodes = [n2]
        n2.pre_nodes = [n1, n3]
        n2.next_nodes = [n3]
        n3.pre_nodes = [n2]
        n3.next_nodes = [n2]
        return [n1, n2, n3]
    if scenario == "multi_input":
        cache.bnds["Node-2"] = ("input bounds",)
        n1, n2, n3 = make_nodes(3, cache, arguments, prop_mode)
        n1.next_nodes = [n3]
        n2.next_nodes = [n3]
        n3.pre_nodes = [n1, n2]
        return [n1, n2, n3]
    if scenario == "multi_output":
        n1, n2, n3 = make_nodes(3, cache, arguments, prop_mode)
        n1.next_nodes = [n2, n3]
        n2.pre_nodes = [n1]
        n3.pre_nodes = [n1]
        return [n1, n2, n3]
    if scenario == "no_input":
        n1, n2, n3 = make_nodes(3, cache, arguments, prop_mode)
        n1.pre_nodes = [n3]
        n1.next_nodes = [n2]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n3]
        n3.pre_nodes = [n2]
        n3.next_nodes = [n1]
        return [n1, n2, n3]
    if scenario == "no_output":
        n1, n2, n3 = make_nodes(3, cache, arguments, prop_mode)
        n1.next_nodes = [n2]
        n1.pre_nodes = [n3]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n3]
        n3.pre_nodes = [n2]
        n3.next_nodes = [n1]
        return [n1, n2, n3]
    raise ValueError(f"Unknown invalid-topology scenario: {scenario}")
