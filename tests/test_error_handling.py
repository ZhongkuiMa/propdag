"""
Test suite for error handling and constraint validation in propdag.

Validates that:
1. DAGs with multiple input nodes raise ValueError
2. DAGs with multiple output nodes raise ValueError
3. DAGs with no input node raise ValueError
4. DAGs with no output node raise ValueError
5. DAGs with cycles raise ValueError during topological sorting
6. All constraints are enforced during TModel initialization
"""

import pytest

from propdag import ForwardToyNode, PropMode, ToyArgument, ToyCache, ToyModel


class TestInputOutputConstraints:
    """Test enforcement of single input/output constraint."""

    def test_multiple_input_nodes_raises_error(self):
        """Test that DAG with 2 input nodes (no predecessors) raises ValueError."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        cache.bnds["Node-2"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        # Create two input nodes feeding into one output
        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Both Node-1 and Node-2 have no predecessors (both are inputs)
        node1.next_nodes = [node3]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node1, node2]

        nodes = [node1, node2, node3]

        # Should raise ValueError due to multiple inputs
        with pytest.raises(ValueError, match=r"input.*output|multiple.*input"):
            ToyModel(nodes, sort_strategy="bfs")

    def test_multiple_output_nodes_raises_error(self):
        """Test that DAG with 2 output nodes (no successors) raises ValueError."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        # Create one input node feeding into two output nodes
        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Both Node-2 and Node-3 have no successors (both are outputs)
        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node3.pre_nodes = [node1]

        nodes = [node1, node2, node3]

        # Should raise ValueError due to multiple outputs
        with pytest.raises(ValueError, match=r"input.*output|multiple.*output"):
            ToyModel(nodes, sort_strategy="bfs")

    def test_no_input_node_raises_error(self):
        """Test that DAG with no input node (all nodes have predecessors) raises ValueError."""
        cache = ToyCache()
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        # Create a cycle: all nodes have predecessors, none are inputs
        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Every node has a predecessor
        node1.pre_nodes = [node3]
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node1]

        nodes = [node1, node2, node3]

        # Should raise ValueError (either for cycle or for no input)
        with pytest.raises(ValueError, match=r"cycle|topological|input"):
            ToyModel(nodes, sort_strategy="bfs")

    def test_no_output_node_raises_error(self):
        """Test that DAG with no output node (all nodes have successors) raises ValueError."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        # Create a cycle: all nodes have successors, none are outputs
        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Every node has a successor
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node1]
        node1.pre_nodes = [node3]

        nodes = [node1, node2, node3]

        # Should raise ValueError (either for cycle or for no output)
        with pytest.raises(ValueError, match=r"cycle|topological|output"):
            ToyModel(nodes, sort_strategy="bfs")


class TestCycleDetection:
    """Test that cycles in the DAG are properly detected and raise ValueError."""

    def test_simple_two_node_cycle(self):
        """Test that a simple 2-node cycle is detected."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)

        # Create cycle: Node-1 → Node-2 → Node-1
        node1.next_nodes = [node2]
        node1.pre_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node1]

        nodes = [node1, node2]

        # Should raise ValueError due to cycle
        with pytest.raises(ValueError, match=r"cycle|topological"):
            ToyModel(nodes, sort_strategy="bfs")

    def test_three_node_cycle(self):
        """Test that a 3-node cycle is detected."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Create cycle: Node-1 → Node-2 → Node-3 → Node-1
        node1.next_nodes = [node2]
        node1.pre_nodes = [node3]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node1]

        nodes = [node1, node2, node3]

        # Should raise ValueError due to cycle
        with pytest.raises(ValueError, match=r"cycle|topological"):
            ToyModel(nodes, sort_strategy="bfs")

    def test_self_loop_cycle(self):
        """Test that a self-loop is detected as a cycle."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)

        # Create self-loop on Node-2
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1, node2]  # Self-loop
        node2.next_nodes = [node2]  # Self-loop

        nodes = [node1, node2]

        # Should raise ValueError due to self-loop cycle
        with pytest.raises(ValueError, match=r"cycle|topological"):
            ToyModel(nodes, sort_strategy="bfs")

    def test_cycle_in_larger_dag(self):
        """
        Test that cycles are detected in larger DAGs.

        DAG with a cycle embedded in a larger structure:
        Node-1 → Node-2 → Node-3 → Node-2 (cycle: 2 → 3 → 2)
        """
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Create DAG with embedded cycle
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node2]  # Back edge creates cycle
        node2.pre_nodes.append(node3)  # Complete the cycle

        nodes = [node1, node2, node3]

        # Should raise ValueError due to cycle
        with pytest.raises(ValueError, match=r"cycle|topological"):
            ToyModel(nodes, sort_strategy="bfs")


class TestBFSAndDFSErrorDetection:
    """Test that both BFS and DFS detect the same constraint violations."""

    def test_both_sort_strategies_detect_multiple_inputs(self):
        """Verify both BFS and DFS detect multiple input nodes."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        cache.bnds["Node-2"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        node1.next_nodes = [node3]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node1, node2]

        nodes = [node1, node2, node3]

        # Both should raise ValueError
        for sort_strategy in ["bfs", "dfs"]:
            with pytest.raises(ValueError, match=r"input|output|multiple"):
                ToyModel(nodes, sort_strategy=sort_strategy)

    def test_both_sort_strategies_detect_cycles(self):
        """Verify both BFS and DFS detect cycles."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)

        # Create cycle
        node1.next_nodes = [node2]
        node1.pre_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node1]

        nodes = [node1, node2]

        # Both should raise ValueError
        for sort_strategy in ["bfs", "dfs"]:
            with pytest.raises(ValueError, match=r"cycle|topological"):
                ToyModel(nodes, sort_strategy=sort_strategy)


class TestValidDAGsAccepted:
    """Test that valid DAGs are accepted without error."""

    def test_linear_chain_accepted(self):
        """Test that a simple linear chain is accepted."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]

        nodes = [node1, node2]

        # Should not raise any error
        model = ToyModel(nodes, sort_strategy="bfs")
        assert model is not None

    def test_diamond_accepted(self):
        """Test that a diamond DAG is accepted."""
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

        # Should not raise any error
        model = ToyModel(nodes, sort_strategy="bfs")
        assert model is not None

    def test_skip_connection_accepted(self):
        """Test that DAGs with skip connections are accepted."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

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

        # Should not raise any error
        model = ToyModel(nodes, sort_strategy="bfs")
        assert model is not None


class TestErrorMessageQuality:
    """Test that error messages are informative."""

    def test_multiple_input_error_is_informative(self):
        """Test that the error message for multiple inputs is clear."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        cache.bnds["Node-2"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        node1.next_nodes = [node3]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node1, node2]

        nodes = [node1, node2, node3]

        # Error should mention input/output constraints
        with pytest.raises(ValueError, match=r"input|output|multiple") as exc_info:
            ToyModel(nodes, sort_strategy="bfs")
        # Message should be informative
        assert len(str(exc_info.value)) > 0

    def test_cycle_error_is_informative(self):
        """Test that the error message for cycles is clear."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)

        node1.next_nodes = [node2]
        node1.pre_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node1]

        nodes = [node1, node2]

        # Error should mention cycle or topological sort
        with pytest.raises(ValueError, match=r"cycle|topological") as exc_info:
            ToyModel(nodes, sort_strategy="bfs")
        # Message should be informative
        assert len(str(exc_info.value)) > 0
