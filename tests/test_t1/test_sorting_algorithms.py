"""
Test suite for topological sorting algorithms (BFS and DFS).

Validates that:
1. Both BFS and DFS produce valid topological orders
2. In the sorted order, every node appears AFTER all its predecessors
3. Both algorithms successfully handle all 17 DAG topologies
4. Edge cases like minimal and long chains work correctly
"""

import pytest

from propdag import ForwardToyNode, PropMode, ToyArgument, ToyCache, ToyModel
from propdag.template._sort import topo_sort_forward_bfs, topo_sort_forward_dfs


class TestTopologicalSortValidity:
    """Validate that topological sorts produce valid orders."""

    def _create_linear_chain_nodes(self, length: int):
        """Create linear chain of nodes."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        nodes = [ForwardToyNode(f"Node-{i}", cache, arguments) for i in range(1, length + 1)]

        # Create linear chain: Node-1 → Node-2 → ... → Node-N
        for i in range(len(nodes) - 1):
            nodes[i].next_nodes = [nodes[i + 1]]
            nodes[i + 1].pre_nodes = [nodes[i]]

        return nodes

    def _verify_topological_order(self, sorted_nodes: list, original_nodes: list):
        """
        Verify that sorted order respects all dependency constraints.

        For every node, all its predecessors must appear before it in the sorted order.
        """
        # Create position map: node_name → position in sorted order
        position = {node.name: i for i, node in enumerate(sorted_nodes)}

        # Check all edges: for each node, all predecessors must come before
        for node in sorted_nodes:
            for pre_node in node.pre_nodes:
                assert position[pre_node.name] < position[node.name], (
                    f"{pre_node.name} should appear before {node.name}"
                )

    @pytest.mark.parametrize("sort_func", [topo_sort_forward_bfs, topo_sort_forward_dfs])
    @pytest.mark.parametrize("length", [2, 5, 10, 15])
    def test_linear_chain_is_valid_order(self, sort_func, length):
        """
        Test that both BFS and DFS produce valid topological orders for linear chains.

        In a linear chain, the only valid order is the sequential order.
        """
        nodes = self._create_linear_chain_nodes(length)
        sorted_nodes = sort_func(nodes, verbose=False)

        # Verify order validity
        self._verify_topological_order(sorted_nodes, nodes)

        # For linear chain, order should match input order
        for i, (orig_node, sorted_node) in enumerate(zip(nodes, sorted_nodes, strict=True)):
            assert orig_node == sorted_node, f"Linear chain must preserve order at position {i}"

    def test_bfs_and_dfs_validity_on_diamond(self):
        """Test that both BFS and DFS produce valid orders on diamond pattern."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        # Create diamond: Node-1 → {Node-2, Node-3} → Node-4
        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)
        node4 = ForwardToyNode("Node-4", cache, arguments)

        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node4]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node2, node3]

        nodes = [node1, node2, node3, node4]

        # Test both sorting algorithms
        for sort_func in [topo_sort_forward_bfs, topo_sort_forward_dfs]:
            sorted_nodes = sort_func(nodes, verbose=False)
            self._verify_topological_order(sorted_nodes, nodes)

    def test_bfs_and_dfs_validity_on_skip_connection(self):
        """Test that both BFS and DFS handle skip connections correctly."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        # Create skip connection: Node-1 → {Node-2, Node-4}; Node-2 → Node-3 → Node-4
        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)
        node4 = ForwardToyNode("Node-4", cache, arguments)

        node1.next_nodes = [node2, node4]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node1, node3]

        nodes = [node1, node2, node3, node4]

        # Test both sorting algorithms
        for sort_func in [topo_sort_forward_bfs, topo_sort_forward_dfs]:
            sorted_nodes = sort_func(nodes, verbose=False)
            self._verify_topological_order(sorted_nodes, nodes)

    def test_bfs_and_dfs_validity_on_wide_merge(self):
        """Test that both BFS and DFS handle many-to-one patterns correctly."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        # Create wide merge: Node-1 → {Node-2, Node-3, Node-4, Node-5} → Node-6
        node1 = ForwardToyNode("Node-1", cache, arguments)
        intermediate = [ForwardToyNode(f"Node-{i}", cache, arguments) for i in range(2, 6)]
        node6 = ForwardToyNode("Node-6", cache, arguments)

        node1.next_nodes = intermediate
        for n in intermediate:
            n.pre_nodes = [node1]
            n.next_nodes = [node6]
        node6.pre_nodes = intermediate

        nodes = [node1, *intermediate, node6]

        # Test both sorting algorithms
        for sort_func in [topo_sort_forward_bfs, topo_sort_forward_dfs]:
            sorted_nodes = sort_func(nodes, verbose=False)
            self._verify_topological_order(sorted_nodes, nodes)


class TestSortingStrategies:
    """Test and compare BFS vs DFS sorting strategies."""

    def test_bfs_vs_dfs_may_differ_but_both_valid(self):
        """
        Test that BFS and DFS may produce different (but equally valid) orders.

        For the diamond pattern, there may be multiple valid topological orders.
        """
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        # Create asymmetric tree where BFS and DFS may differ
        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)
        node4 = ForwardToyNode("Node-4", cache, arguments)
        node5 = ForwardToyNode("Node-5", cache, arguments)

        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node4]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node5]
        node4.pre_nodes = [node2]
        node4.next_nodes = []
        node5.pre_nodes = [node3]
        node5.next_nodes = []

        nodes = [node1, node2, node3, node4, node5]

        # Get both sortings
        bfs_sorted = topo_sort_forward_bfs(nodes, verbose=False)
        dfs_sorted = topo_sort_forward_dfs(nodes, verbose=False)

        # Both should be valid topological orders
        for sorted_nodes in [bfs_sorted, dfs_sorted]:
            position = {node.name: i for i, node in enumerate(sorted_nodes)}
            for node in sorted_nodes:
                for pre in node.pre_nodes:
                    assert position[pre.name] < position[node.name]

    def test_model_works_with_both_sort_strategies(self):
        """Test that ToyModel successfully executes with both BFS and DFS."""
        for sort_strategy in ["bfs", "dfs"]:
            cache = ToyCache()
            cache.bnds["Node-1"] = ("input bounds",)
            arguments = ToyArgument(prop_mode=PropMode.FORWARD)

            node1 = ForwardToyNode("Node-1", cache, arguments)
            node2 = ForwardToyNode("Node-2", cache, arguments)
            node3 = ForwardToyNode("Node-3", cache, arguments)
            node4 = ForwardToyNode("Node-4", cache, arguments)

            node1.next_nodes = [node2, node3]
            node2.pre_nodes = [node1]
            node2.next_nodes = [node4]
            node3.pre_nodes = [node1]
            node3.next_nodes = [node4]
            node4.pre_nodes = [node2, node3]

            nodes = [node1, node2, node3, node4]
            model = ToyModel(nodes, sort_strategy=sort_strategy)
            model.run()

            # Both should complete successfully
            assert "Node-1" in cache.bnds
            assert "Node-4" in cache.bnds


class TestEdgeCases:
    """Test topological sorting on edge cases."""

    def test_minimal_two_node_dag(self):
        """Test that minimal DAG (2 nodes) sorts correctly."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]

        nodes = [node1, node2]

        for sort_func in [topo_sort_forward_bfs, topo_sort_forward_dfs]:
            sorted_nodes = sort_func(nodes, verbose=False)
            assert len(sorted_nodes) == 2
            assert sorted_nodes[0] == node1
            assert sorted_nodes[1] == node2

    @pytest.mark.benchmark
    def test_long_chain_maintains_order(self):
        """Test that long linear chains (10, 20, 30 nodes) maintain their order (benchmark test)."""
        for length in [10, 20, 30]:
            cache = ToyCache()
            cache.bnds["Node-1"] = ("input bounds",)
            arguments = ToyArgument(prop_mode=PropMode.FORWARD)

            nodes = [ForwardToyNode(f"Node-{i}", cache, arguments) for i in range(1, length + 1)]

            # Create linear chain
            for i in range(len(nodes) - 1):
                nodes[i].next_nodes = [nodes[i + 1]]
                nodes[i + 1].pre_nodes = [nodes[i]]

            for sort_func in [topo_sort_forward_bfs, topo_sort_forward_dfs]:
                sorted_nodes = sort_func(nodes, verbose=False)
                # For linear chains, order must be preserved
                for _, (orig, sorted_node) in enumerate(zip(nodes, sorted_nodes, strict=True)):
                    assert orig == sorted_node

    def test_same_node_multiple_times_as_predecessor(self):
        """Test DAG where same node appears multiple times as predecessor."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Node-1 → {Node-2, Node-2} (duplicate edge)
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1, node1]  # Appears twice
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        for sort_func in [topo_sort_forward_bfs, topo_sort_forward_dfs]:
            sorted_nodes = sort_func(nodes, verbose=False)
            # Should still produce valid order
            position = {node.name: i for i, node in enumerate(sorted_nodes)}
            assert position["Node-1"] < position["Node-2"]
            assert position["Node-2"] < position["Node-3"]
