"""
Test suite for topological sorting algorithms (BFS and DFS) in template/.

Validates that:
1. Both BFS and DFS produce valid topological orders.
2. Every node appears AFTER all of its predecessors.
3. Both algorithms handle representative DAG topologies correctly.
4. Edge cases like minimal and long chains work correctly.
5. ``topo_sort_backward`` (previously uncovered) returns valid per-node sorts.
"""

import pytest
from _helpers import verify_topological_order

from propdag import ForwardToyNode, PropMode, ToyArgument, ToyCache
from propdag.template._sort import (
    topo_sort_backward,
    topo_sort_forward_bfs,
    topo_sort_forward_dfs,
)
from test_template._helpers import build_chain_model

_FORWARD_SORTS = [topo_sort_forward_bfs, topo_sort_forward_dfs]


def _build_sort_topology_nodes(topology: str) -> list[ForwardToyNode]:
    """Build nodes for a named topology, returning the user-orientation node list.

    :param topology: one of ``"diamond"``, ``"skip_connection"``, ``"wide_merge"``
    :return: list of wired ForwardToyNode instances
    """
    cache = ToyCache()
    cache.bnds["Node-1"] = ("input bounds",)
    arguments = ToyArgument(prop_mode=PropMode.FORWARD)

    if topology == "diamond":
        n1 = ForwardToyNode("Node-1", cache, arguments)
        n2 = ForwardToyNode("Node-2", cache, arguments)
        n3 = ForwardToyNode("Node-3", cache, arguments)
        n4 = ForwardToyNode("Node-4", cache, arguments)
        n1.next_nodes = [n2, n3]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n4]
        n3.pre_nodes = [n1]
        n3.next_nodes = [n4]
        n4.pre_nodes = [n2, n3]
        return [n1, n2, n3, n4]

    if topology == "skip_connection":
        n1 = ForwardToyNode("Node-1", cache, arguments)
        n2 = ForwardToyNode("Node-2", cache, arguments)
        n3 = ForwardToyNode("Node-3", cache, arguments)
        n4 = ForwardToyNode("Node-4", cache, arguments)
        n1.next_nodes = [n2, n4]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n3]
        n3.pre_nodes = [n2]
        n3.next_nodes = [n4]
        n4.pre_nodes = [n1, n3]
        return [n1, n2, n3, n4]

    if topology == "wide_merge":
        n1 = ForwardToyNode("Node-1", cache, arguments)
        intermediate = [ForwardToyNode(f"Node-{i}", cache, arguments) for i in range(2, 6)]
        n6 = ForwardToyNode("Node-6", cache, arguments)
        n1.next_nodes = intermediate
        for n in intermediate:
            n.pre_nodes = [n1]
            n.next_nodes = [n6]
        n6.pre_nodes = intermediate
        return [n1, *intermediate, n6]

    msg = f"Unknown topology: {topology}"
    raise ValueError(msg)


class TestTopologicalSortValidity:
    """Topological sorts produce valid orders for representative topologies."""

    @pytest.mark.parametrize("sort_func", _FORWARD_SORTS)
    @pytest.mark.parametrize("length", [2, 5, 10, 15])
    def test_linear_chain_is_valid_order(self, sort_func, length):
        """Both BFS and DFS sort a linear chain into the unique valid order."""
        _, _, nodes = build_chain_model(length)
        sorted_nodes = sort_func(nodes, verbose=False)
        verify_topological_order(sorted_nodes)
        for i, (orig, ordered) in enumerate(zip(nodes, sorted_nodes, strict=True)):
            assert orig == ordered, f"Linear chain must preserve order at position {i}"

    @pytest.mark.parametrize(
        "topology",
        [
            pytest.param("diamond", id="diamond"),
            pytest.param("skip_connection", id="skip_connection"),
            pytest.param("wide_merge", id="wide_merge"),
        ],
    )
    def test_bfs_and_dfs_validity_on_various_topologies(self, topology):
        """Various topologies produce valid topological orders under both BFS and DFS."""
        nodes = _build_sort_topology_nodes(topology)
        for sort_func in _FORWARD_SORTS:
            sorted_nodes = sort_func(nodes, verbose=False)
            assert len(sorted_nodes) == len(nodes), "Sort must return all nodes"
            verify_topological_order(sorted_nodes)


class TestSortingStrategies:
    """BFS and DFS may produce different but equally valid orders."""

    def test_bfs_vs_dfs_may_differ_but_both_valid(self):
        """For an asymmetric tree, both BFS and DFS produce valid topological orders."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)
        n1 = ForwardToyNode("Node-1", cache, arguments)
        n2 = ForwardToyNode("Node-2", cache, arguments)
        n3 = ForwardToyNode("Node-3", cache, arguments)
        n4 = ForwardToyNode("Node-4", cache, arguments)
        n5 = ForwardToyNode("Node-5", cache, arguments)
        n1.next_nodes = [n2, n3]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n4]
        n3.pre_nodes = [n1]
        n3.next_nodes = [n5]
        n4.pre_nodes = [n2]
        n5.pre_nodes = [n3]
        nodes = [n1, n2, n3, n4, n5]

        for sort_func in _FORWARD_SORTS:
            verify_topological_order(sort_func(nodes, verbose=False))

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_model_works_with_both_sort_strategies(self, sort_strategy):
        """ToyModel runs to completion for both BFS and DFS on a diamond."""
        from test_template._helpers import build_diamond_model

        model, cache, _ = build_diamond_model(sort_strategy=sort_strategy)
        model.run()
        assert "Node-1" in cache.bnds
        assert "Node-4" in cache.bnds


class TestEdgeCases:
    """Topological sorting on edge-case topologies."""

    def test_minimal_two_node_dag(self):
        """A minimal 2-node DAG sorts correctly under both algorithms."""
        _, _, nodes = build_chain_model(2)
        for sort_func in _FORWARD_SORTS:
            sorted_nodes = sort_func(nodes, verbose=False)
            assert len(sorted_nodes) == 2
            assert sorted_nodes[0] == nodes[0]
            assert sorted_nodes[1] == nodes[1]

    @pytest.mark.parametrize("length", [10, 20, 30])
    def test_long_chain_maintains_order(self, length):
        """Long linear chains preserve their input order under both algorithms."""
        _, _, nodes = build_chain_model(length)
        for sort_func in _FORWARD_SORTS:
            sorted_nodes = sort_func(nodes, verbose=False)
            for orig, ordered in zip(nodes, sorted_nodes, strict=True):
                assert orig == ordered

    def test_same_node_multiple_times_as_predecessor(self):
        """Duplicate predecessor entries do not break the topological sort."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)
        n1 = ForwardToyNode("Node-1", cache, arguments)
        n2 = ForwardToyNode("Node-2", cache, arguments)
        n3 = ForwardToyNode("Node-3", cache, arguments)
        n1.next_nodes = [n2]
        n2.pre_nodes = [n1, n1]  # Duplicate edge
        n2.next_nodes = [n3]
        n3.pre_nodes = [n2]
        nodes = [n1, n2, n3]

        for sort_func in _FORWARD_SORTS:
            sorted_nodes = sort_func(nodes, verbose=False)
            position = {node.name: i for i, node in enumerate(sorted_nodes)}
            assert position["Node-1"] < position["Node-2"]
            assert position["Node-2"] < position["Node-3"]


class TestBackwardSort:
    """COV2: ``topo_sort_backward`` produces valid per-node backward sorts."""

    def test_returns_one_entry_per_node(self):
        """``topo_sort_backward`` returns a dict with every node as a key."""
        _, _, nodes = build_chain_model(4)
        result = topo_sort_backward(nodes, verbose=False)
        assert set(result.keys()) == set(nodes)

    def test_each_value_starts_with_target_and_lists_only_ancestors(self):
        """Each per-node sequence starts with the target and contains only the target plus its ancestors."""
        _, _, nodes = build_chain_model(4)
        result = topo_sort_backward(nodes, verbose=False)
        for target, sequence in result.items():
            assert sequence[0] is target, f"target {target.name} must be first in its backward sort"
            # Every entry after the target must be an ancestor (reachable via pre_nodes).
            ancestor_names = {n.name for n in sequence}
            target_index = nodes.index(target)
            expected_ancestors = {nodes[i].name for i in range(target_index + 1)}
            assert ancestor_names == expected_ancestors, (
                f"backward sort for {target.name} must contain only it plus its ancestors"
            )
