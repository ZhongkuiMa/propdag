"""
Test suite for error handling and validation in T2.

Tests constraint enforcement and error detection in reversed graph model.
Adapted from test_t1/test_error_handling.py.
"""

import pytest

from propdag import Toy2Argument, Toy2Cache, Toy2Model, Toy2Node


class TestInputOutputConstraints:
    """Test enforcement of single input/output constraint."""

    def test_multiple_input_nodes_raises_error(self):
        """Test that DAG with 2 input nodes (no predecessors) raises ValueError."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        cache.fwd_bnds["Node-2"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        # Two independent inputs
        node1.next_nodes = [node3]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node1, node2]

        nodes = [node1, node2, node3]

        with pytest.raises(ValueError, match=r"exactly one input"):
            Toy2Model(nodes)

    def test_multiple_output_nodes_raises_error(self):
        """Test that DAG with 2 output nodes (no successors) raises ValueError."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        # One input, two outputs
        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node3.pre_nodes = [node1]

        nodes = [node1, node2, node3]

        with pytest.raises(ValueError, match=r"exactly one output"):
            Toy2Model(nodes)

    def test_no_input_node_raises_error(self):
        """Test that DAG with no input (cycle) raises ValueError."""
        cache = Toy2Cache()
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)

        # Create cycle: no input
        node1.next_nodes = [node2]
        node1.pre_nodes = [node2]
        node2.next_nodes = [node1]
        node2.pre_nodes = [node1]

        nodes = [node1, node2]

        with pytest.raises(ValueError, match=r"exactly one input.*zero"):
            Toy2Model(nodes)

    def test_no_output_node_raises_error(self):
        """Test that DAG with no output (cycle) raises ValueError."""
        cache = Toy2Cache()
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)

        # Create cycle: no output
        node1.next_nodes = [node2]
        node1.pre_nodes = [node2]
        node2.next_nodes = [node1]
        node2.pre_nodes = [node1]

        nodes = [node1, node2]

        # Will error on no input first
        with pytest.raises(ValueError, match=r"exactly one"):
            Toy2Model(nodes)


class TestCycleDetection:
    """Test that cycles in the DAG are properly detected and raise ValueError."""

    def test_simple_two_node_cycle(self):
        """Test that a simple 2-node cycle is detected."""
        cache = Toy2Cache()
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)

        # Create cycle
        node1.next_nodes = [node2]
        node1.pre_nodes = [node2]
        node2.next_nodes = [node1]
        node2.pre_nodes = [node1]

        nodes = [node1, node2]

        with pytest.raises(ValueError, match=r"cycle|topological|input"):
            Toy2Model(nodes, sort_strategy="bfs")

    def test_three_node_cycle(self):
        """Test that a 3-node cycle is detected."""
        cache = Toy2Cache()
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        # Create cycle: 1 → 2 → 3 → 1
        node1.next_nodes = [node2]
        node1.pre_nodes = [node3]
        node2.next_nodes = [node3]
        node2.pre_nodes = [node1]
        node3.next_nodes = [node1]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        with pytest.raises(ValueError, match=r"cycle|topological|input"):
            Toy2Model(nodes, sort_strategy="bfs")


class TestBFSAndDFSErrorDetection:
    """Test that both BFS and DFS detect the same constraint violations."""

    def test_both_sort_strategies_detect_multiple_inputs(self):
        """Verify both BFS and DFS detect multiple input nodes."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        cache.fwd_bnds["Node-2"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        # Two inputs
        node1.next_nodes = [node3]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node1, node2]

        nodes = [node1, node2, node3]

        # Both should raise ValueError
        for sort_strategy in ["bfs", "dfs"]:
            with pytest.raises(ValueError, match=r"exactly one input"):
                Toy2Model(nodes, sort_strategy=sort_strategy)

    def test_both_sort_strategies_detect_cycles(self):
        """Verify both BFS and DFS detect cycles."""
        cache = Toy2Cache()
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)

        # Cycle
        node1.next_nodes = [node2]
        node1.pre_nodes = [node2]
        node2.next_nodes = [node1]
        node2.pre_nodes = [node1]

        nodes = [node1, node2]

        # Both should raise ValueError
        for sort_strategy in ["bfs", "dfs"]:
            with pytest.raises(ValueError, match=r"cycle|topological|input"):
                Toy2Model(nodes, sort_strategy=sort_strategy)


class TestValidDAGsAccepted:
    """Test that valid DAGs are accepted without error."""

    def test_linear_chain_accepted(self):
        """Test that a simple linear chain is accepted."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Should not raise any error
        model = Toy2Model(nodes, sort_strategy="bfs")
        assert model is not None

    def test_diamond_accepted(self):
        """Test that diamond pattern is accepted."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)
        node4 = Toy2Node("Node-4", cache, arguments)

        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node4]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node2, node3]

        nodes = [node1, node2, node3, node4]

        # Should not raise any error
        model = Toy2Model(nodes, sort_strategy="bfs")
        assert model is not None


class TestErrorMessageQuality:
    """Test that error messages are informative."""

    def test_multiple_input_error_is_informative(self):
        """Test that the error message for multiple inputs is clear."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        cache.fwd_bnds["Node-2"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        node1.next_nodes = [node3]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node1, node2]

        nodes = [node1, node2, node3]

        with pytest.raises(ValueError, match=r"exactly one input") as excinfo:
            Toy2Model(nodes)

        # Error message should mention node names
        error_msg = str(excinfo.value)
        assert "Node-1" in error_msg or "Node-2" in error_msg
