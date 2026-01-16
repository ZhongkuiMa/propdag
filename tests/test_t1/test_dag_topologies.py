"""
Test suite for various DAG topologies in propdag.

Tests 17 different directed acyclic graph structures including:
- Linear chains
- Skip connections (ResNet-like)
- Multi-input nodes (concatenation/merge)
- Complex branching patterns
- Edge cases and realistic neural network patterns

Each test is parametrized over both BFS/DFS sorting strategies and
forward/backward propagation modes, resulting in 68 total test cases
(17 topologies x 2 sort strategies x 2 propagation modes).
"""

import pytest

from propdag import (
    BackwardToyNode,
    ForwardToyNode,
    PropMode,
    ToyArgument,
    ToyCache,
    ToyModel,
)


class TestBasicStructures:
    """Category 1: Basic structures (linear and simple branching)."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_linear_chain(self, sort_strategy, prop_mode):
        """Test basic linear chain DAG: Node-1 → Node-2 → Node-3 → Node-4."""
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        node1 = node_class("Node-1", cache, arguments)
        node2 = node_class("Node-2", cache, arguments)
        node3 = node_class("Node-3", cache, arguments)
        node4 = node_class("Node-4", cache, arguments)

        # Define topology: linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node3]

        # Execute
        model = ToyModel([node1, node2, node3, node4], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert "Node-4" in cache.bnds, "Output node must have bounds"
        assert cache.bnds["Node-4"] is not None
        assert "Node-1" in cache.bnds, "Input node bounds must be preserved"

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_y_shape(self, sort_strategy, prop_mode):
        """
        Test Y-shape DAG (single branch point and merge).

        Converted from example_forward.py and example_backward.py.
        Node-1 splits into Node-2 and Node-3, which merge at Node-4.
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        node1 = node_class("Node-1", cache, arguments)
        node2 = node_class("Node-2", cache, arguments)
        node3 = node_class("Node-3", cache, arguments)
        node4 = node_class("Node-4", cache, arguments)

        # Define topology: Y-shape
        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node4]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node2, node3]

        # Execute
        model = ToyModel([node1, node2, node3, node4], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-4"] is not None
        assert "Node-1" in cache.bnds, "Input node bounds preserved"

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_sequential_branches(self, sort_strategy, prop_mode):
        r"""
        Test late-stage branching and merging.

        Node-1 -> Node-2 -> Node-3
                   \       /
                   Node-4
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        node1 = node_class("Node-1", cache, arguments)
        node2 = node_class("Node-2", cache, arguments)
        node3 = node_class("Node-3", cache, arguments)
        node4 = node_class("Node-4", cache, arguments)

        # Define topology
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3, node4]  # Node-2 goes to both Node-3 and Node-4
        node3.pre_nodes = [node2]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node2, node3]

        # Execute
        model = ToyModel([node1, node2, node3, node4], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-4"] is not None


class TestSkipConnections:
    """Category 2: Skip connections and residual patterns."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_single_skip_connection(self, sort_strategy, prop_mode):
        r"""
        Test single skip connection bypassing intermediate nodes.

        Node-1 -> Node-2 -> Node-3
           |________________v Node-4
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        node1 = node_class("Node-1", cache, arguments)
        node2 = node_class("Node-2", cache, arguments)
        node3 = node_class("Node-3", cache, arguments)
        node4 = node_class("Node-4", cache, arguments)

        # Define topology with skip connection
        node1.next_nodes = [node2, node4]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node1, node3]

        # Execute
        model = ToyModel([node1, node2, node3, node4], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-4"] is not None

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_multiple_skip_connections(self, sort_strategy, prop_mode):
        """
        Test multiple parallel skip connections (ResNet-like pattern).

        Node-1 → Node-2 → Node-3 → Node-4
           |______↓         |_______↓
                Node-5            |
                                  ↓
                               Node-6
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        nodes = [node_class(f"Node-{i}", cache, arguments) for i in range(1, 7)]
        node1, node2, node3, node4, node5, node6 = nodes

        # Define topology with multiple skip connections
        node1.next_nodes = [node2, node5]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node4, node6]
        node4.pre_nodes = [node3]
        node4.next_nodes = [node6]
        node5.pre_nodes = [node1]
        node5.next_nodes = [node6]  # Node-5 feeds into Node-6
        node6.pre_nodes = [node3, node4, node5]  # Node-6 receives from multiple sources

        # Execute
        model = ToyModel(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-6"] is not None

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_nested_skip_connections(self, sort_strategy, prop_mode):
        """
        Test overlapping skip connections.

        Node-1 → Node-2 → Node-3
           |       |_______↓
           |_______________↓
                        Node-4
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        node1 = node_class("Node-1", cache, arguments)
        node2 = node_class("Node-2", cache, arguments)
        node3 = node_class("Node-3", cache, arguments)
        node4 = node_class("Node-4", cache, arguments)

        # Define topology with nested skips
        node1.next_nodes = [node2, node4]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3, node4]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node1, node2, node3]

        # Execute
        model = ToyModel([node1, node2, node3, node4], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-4"] is not None


class TestMultiInputCases:
    """Category 3: Multi-input edge cases (same source to target, diamonds, etc.)."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_same_source_twice(self, sort_strategy, prop_mode):
        """
        Test same source appearing twice in pre_nodes (like x² = x * x).

        Node-1 ⇉ Node-2 (two edges from Node-1)
                 ↓
              Node-3
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        node1 = node_class("Node-1", cache, arguments)
        node2 = node_class("Node-2", cache, arguments)
        node3 = node_class("Node-3", cache, arguments)

        # Define topology: Node-1 appears twice in Node-2's pre_nodes
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1, node1]  # Duplicate edge
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        # Execute
        model = ToyModel([node1, node2, node3], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-3"] is not None

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_diamond_pattern(self, sort_strategy, prop_mode):
        r"""
        Test standard diamond merge pattern.

              Node-1
              /    \
          Node-2  Node-3
              \    /
              Node-4 (receives from both Node-2 and Node-3)
                 v
              Node-5
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        nodes = [node_class(f"Node-{i}", cache, arguments) for i in range(1, 6)]
        node1, node2, node3, node4, node5 = nodes

        # Define topology: diamond
        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node4]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node2, node3]
        node4.next_nodes = [node5]
        node5.pre_nodes = [node4]

        # Execute
        model = ToyModel(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-5"] is not None

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_wide_merge(self, sort_strategy, prop_mode):
        """
        Test wide merge (many-to-one): 4 nodes merging into 1.

        Node-1 → Node-2 ↘
        Node-1 → Node-3 → Node-6
        Node-1 → Node-4 ↗
                 Node-5 ↗
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        nodes = [node_class(f"Node-{i}", cache, arguments) for i in range(1, 7)]
        node1, node2, node3, node4, node5, node6 = nodes

        # Define topology: wide merge
        node1.next_nodes = [node2, node3, node4, node5]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node6]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node6]
        node4.pre_nodes = [node1]
        node4.next_nodes = [node6]
        node5.pre_nodes = [node1]
        node5.next_nodes = [node6]
        node6.pre_nodes = [node2, node3, node4, node5]

        # Execute
        model = ToyModel(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-6"] is not None


class TestComplexBranching:
    """Category 4: Complex branching patterns."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_wide_broadcast(self, sort_strategy, prop_mode):
        r"""
        Test one node broadcasting to many (one-to-many).

                 Node-1
               /  |  |  \
        Node-2 Node-3 Node-4 Node-5
               \  |  |  /
                 Node-6
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        nodes = [node_class(f"Node-{i}", cache, arguments) for i in range(1, 7)]
        node1, node2, node3, node4, node5, node6 = nodes

        # Define topology: wide broadcast
        node1.next_nodes = [node2, node3, node4, node5]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node6]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node6]
        node4.pre_nodes = [node1]
        node4.next_nodes = [node6]
        node5.pre_nodes = [node1]
        node5.next_nodes = [node6]
        node6.pre_nodes = [node2, node3, node4, node5]

        # Execute
        model = ToyModel(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-6"] is not None

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_asymmetric_tree(self, sort_strategy, prop_mode):
        r"""
        Test unbalanced graph structure.

              Node-1
              /    \
          Node-2   Node-3
            |      /    \
          Node-4  Node-5 Node-6
            |      \    /
          Node-7   Node-8
              \    /
              Node-9
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        nodes = [node_class(f"Node-{i}", cache, arguments) for i in range(1, 10)]
        (
            node1,
            node2,
            node3,
            node4,
            node5,
            node6,
            node7,
            node8,
            node9,
        ) = nodes

        # Define topology: asymmetric tree
        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node4]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node5, node6]
        node4.pre_nodes = [node2]
        node4.next_nodes = [node7]
        node5.pre_nodes = [node3]
        node5.next_nodes = [node8]
        node6.pre_nodes = [node3]
        node6.next_nodes = [node8]
        node7.pre_nodes = [node4]
        node7.next_nodes = [node9]
        node8.pre_nodes = [node5, node6]
        node8.next_nodes = [node9]
        node9.pre_nodes = [node7, node8]

        # Execute
        model = ToyModel(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-9"] is not None

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_multiple_merge_points(self, sort_strategy, prop_mode):
        """
        Test serial merging at different depths.

        Node-1 → Node-2 → Node-4 → Node-6
           |       ↓       ↓       ↓
           └────→ Node-3 → Node-5 → Node-7
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        nodes = [node_class(f"Node-{i}", cache, arguments) for i in range(1, 8)]
        node1, node2, node3, node4, node5, node6, node7 = nodes

        # Define topology: progressive merging
        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node4]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node5]
        node4.pre_nodes = [node2]
        node4.next_nodes = [node6]
        node5.pre_nodes = [node3]
        node5.next_nodes = [node7]
        node6.pre_nodes = [node4]
        node6.next_nodes = [node7]
        node7.pre_nodes = [node5, node6]

        # Execute
        model = ToyModel(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-7"] is not None


class TestBoundaryAndEdgeCases:
    """Category 5: Boundary and degenerate cases."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_minimal_dag(self, sort_strategy, prop_mode):
        """Test minimal DAG with just 2 nodes."""
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        node1 = node_class("Node-1", cache, arguments)
        node2 = node_class("Node-2", cache, arguments)

        # Define topology: minimal
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]

        # Execute
        model = ToyModel([node1, node2], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-2"] is not None
        assert cache.bnds["Node-1"] is not None

    @pytest.mark.benchmark
    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_long_chain(self, sort_strategy, prop_mode):
        """Test long linear chain of 15 nodes (benchmark test)."""
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        nodes = [node_class(f"Node-{i}", cache, arguments) for i in range(1, 16)]

        # Define topology: linear chain
        for i in range(len(nodes) - 1):
            nodes[i].next_nodes = [nodes[i + 1]]
            nodes[i + 1].pre_nodes = [nodes[i]]

        # Execute
        model = ToyModel(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-15"] is not None

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_deep_branching(self, sort_strategy, prop_mode):
        """Test early broadcast, late merge pattern."""
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes (Node-1 broadcasts to 6 nodes, all converge to Node-8)
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        nodes = [node_class(f"Node-{i}", cache, arguments) for i in range(1, 9)]
        node1 = nodes[0]
        intermediate = nodes[1:7]
        node8 = nodes[7]

        # Define topology: broadcast then merge
        node1.next_nodes = intermediate
        for n in intermediate:
            n.pre_nodes = [node1]
            n.next_nodes = [node8]
        node8.pre_nodes = intermediate

        # Execute
        model = ToyModel(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-8"] is not None


class TestRealisticNetworkPatterns:
    """Category 6: Realistic neural network patterns."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_inception_like_module(self, sort_strategy, prop_mode):
        r"""
        Test Inception-like parallel processing paths.

                Node-1
            /    |    |    \
          N-2   N-3  N-4   N-5
           |     |    |     |
          N-6   N-7  N-8   N-9
            \    |    |    /
                Node-10
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        nodes = [node_class(f"Node-{i}", cache, arguments) for i in range(1, 11)]
        (
            node1,
            node2,
            node3,
            node4,
            node5,
            node6,
            node7,
            node8,
            node9,
            node10,
        ) = nodes

        # Define topology: Inception-like
        node1.next_nodes = [node2, node3, node4, node5]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node6]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node7]
        node4.pre_nodes = [node1]
        node4.next_nodes = [node8]
        node5.pre_nodes = [node1]
        node5.next_nodes = [node9]
        node6.pre_nodes = [node2]
        node6.next_nodes = [node10]
        node7.pre_nodes = [node3]
        node7.next_nodes = [node10]
        node8.pre_nodes = [node4]
        node8.next_nodes = [node10]
        node9.pre_nodes = [node5]
        node9.next_nodes = [node10]
        node10.pre_nodes = [node6, node7, node8, node9]

        # Execute
        model = ToyModel(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-10"] is not None

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    @pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
    def test_densenet_like_connection(self, sort_strategy, prop_mode):
        r"""
        Test DenseNet-like dense connectivity pattern.

        Node-1 -> Node-2 -> Node-3 -> Node-4
           |________v________v________v
                               Node-5
        """
        # Setup
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)

        # Create nodes
        node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode
        nodes = [node_class(f"Node-{i}", cache, arguments) for i in range(1, 6)]
        node1, node2, node3, node4, node5 = nodes

        # Define topology: DenseNet-like (all previous to one)
        node1.next_nodes = [node2, node5]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3, node5]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node4, node5]
        node4.pre_nodes = [node3]
        node4.next_nodes = [node5]
        node5.pre_nodes = [node1, node2, node3, node4]

        # Execute
        model = ToyModel(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert cache.bnds["Node-5"] is not None
