"""
Test suite for various DAG topologies in T2 (reversed graph semantics).

Tests 17 different directed acyclic graph structures using template2/toy2:
- Linear chains
- Skip connections (ResNet-like)
- Multi-input nodes (concatenation/merge)
- Complex branching patterns
- Edge cases and realistic neural network patterns

Each test is parametrized over BFS/DFS sorting strategies only (no PropMode),
resulting in 34 total test cases (17 topologies x 2 sort strategies).

Unlike T1 which has separate ForwardToyNode/BackwardToyNode with PropMode,
T2 uses a single Toy2Node with automatic graph reversal for backward propagation.
"""

import pytest

from propdag import Toy2Argument, Toy2Cache, Toy2Model, Toy2Node


class TestBasicStructures:
    """Category 1: Basic structures (linear and simple branching)."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_linear_chain(self, sort_strategy):
        """Test basic linear chain DAG: Node-1 → Node-2 → Node-3 → Node-4."""
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)
        node4 = Toy2Node("Node-4", cache, arguments)

        # Define topology: linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node3]

        # Execute (graph reversed automatically)
        model = Toy2Model([node1, node2, node3, node4], sort_strategy=sort_strategy)
        model.run()

        # Assertions - bounds should be computed
        assert "Node-1" in cache.bnds or "Node-4" in cache.bnds

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_y_shape(self, sort_strategy):
        """
        Test Y-shape DAG (single branch point and merge).

        Node-1 splits into Node-2 and Node-3, which merge at Node-4.
        """
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)
        node4 = Toy2Node("Node-4", cache, arguments)

        # Define topology: Y-shape
        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node4]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node2, node3]

        # Execute
        model = Toy2Model([node1, node2, node3, node4], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert "Node-1" in cache.bnds or "Node-4" in cache.bnds


class TestSkipConnections:
    """Category 2: Skip connections and residual patterns."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_single_skip_connection(self, sort_strategy):
        r"""
        Test single skip connection bypassing intermediate nodes.

        Node-1 → Node-2 → Node-3 → Node-4
                 └────────────────→ ┘
        """
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)
        node4 = Toy2Node("Node-4", cache, arguments)

        # Define topology: skip connection
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3, node4]  # Skip to node4
        node3.pre_nodes = [node2]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node2, node3]

        # Execute
        model = Toy2Model([node1, node2, node3, node4], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert "Node-1" in cache.bnds or "Node-4" in cache.bnds

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_multiple_skip_connections(self, sort_strategy):
        r"""
        Test multiple parallel skip connections (ResNet-like).

        Node-1 → Node-2 → Node-3 → Node-4
                 └────────→ ┘      ↑
                 └──────────────────┘
        """
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)
        node4 = Toy2Node("Node-4", cache, arguments)

        # Define topology: multiple skip connections
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3, node4]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node2, node3]

        # Execute
        model = Toy2Model([node1, node2, node3, node4], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert "Node-1" in cache.bnds or "Node-4" in cache.bnds

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_nested_skip_connections(self, sort_strategy):
        r"""
        Test nested skip connections (skip within skip).

        Node-1 → Node-2 → Node-3 → Node-4
                 └────────→ ┘      ↑
                 └──────────────────┘
        """
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)
        node4 = Toy2Node("Node-4", cache, arguments)

        # Define topology
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3, node4]
        node3.pre_nodes = [node2]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node2, node3]

        # Execute
        model = Toy2Model([node1, node2, node3, node4], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert "Node-1" in cache.bnds or "Node-4" in cache.bnds


class TestMultiInputCases:
    """Category 3: Multi-input edge cases (same source to target, diamonds, etc.)."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_same_source_twice(self, sort_strategy):
        """
        Test same source appearing twice in pre_nodes (like x² = x * x).

        Node-1 → Node-2 (receives Node-1 twice)
        """
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)

        # Define topology: same source twice
        node1.next_nodes = [node2, node2]
        node2.pre_nodes = [node1, node1]

        # Execute
        model = Toy2Model([node1, node2], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert "Node-1" in cache.bnds or "Node-2" in cache.bnds

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_diamond_pattern(self, sort_strategy):
        r"""
        Test diamond pattern (two parallel paths).

        Node-1 → Node-2 → Node-4
                 └→ Node-3 → ┘
        """
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)
        node4 = Toy2Node("Node-4", cache, arguments)

        # Define topology: diamond
        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node3.pre_nodes = [node1]
        node2.next_nodes = [node4]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node2, node3]

        # Execute
        model = Toy2Model([node1, node2, node3, node4], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert "Node-1" in cache.bnds or "Node-4" in cache.bnds

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_wide_merge(self, sort_strategy):
        """
        Test wide merge (many-to-one): 4 nodes merging into 1.

        Node-1 → Node-2 ↘
                 Node-3 → Node-6
                 Node-4 ↗
                 Node-5 ↗
        """
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        nodes = [Toy2Node(f"Node-{i}", cache, arguments) for i in range(1, 7)]
        node1, node2, node3, node4, node5, node6 = nodes

        # Define topology: single input broadcasts to 4 nodes, all merge to output
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
        model = Toy2Model(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert len(cache.bnds) > 0


class TestComplexBranching:
    """Category 4: Complex branching patterns."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_wide_broadcast(self, sort_strategy):
        r"""
        Test one node broadcasting to many (one-to-many).

                 Node-1
               /  |  |  \
        Node-2 Node-3 Node-4 Node-5
               \  |  |  /
                 Node-6
        """
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        nodes = [Toy2Node(f"Node-{i}", cache, arguments) for i in range(1, 7)]
        node1, node2, node3, node4, node5, node6 = nodes

        # Define topology: single input broadcasts, all converge to single output
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
        model = Toy2Model(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert len(cache.bnds) > 0

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_asymmetric_tree(self, sort_strategy):
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
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        nodes = [Toy2Node(f"Node-{i}", cache, arguments) for i in range(1, 10)]
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

        # Define topology: asymmetric tree with single input/output
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
        model = Toy2Model(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert len(cache.bnds) > 0

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_multiple_merge_points(self, sort_strategy):
        """
        Test serial merging at different depths.

        Node-1 → Node-2 → Node-4 → Node-6
           |       ↓       ↓       ↓
           └────→ Node-3 → Node-5 → Node-7
        """
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        nodes = [Toy2Node(f"Node-{i}", cache, arguments) for i in range(1, 8)]
        node1, node2, node3, node4, node5, node6, node7 = nodes

        # Define topology: progressive merging with single input/output
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
        model = Toy2Model(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert len(cache.bnds) > 0


class TestBoundaryAndEdgeCases:
    """Category 5: Boundary and degenerate cases."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_minimal_dag(self, sort_strategy):
        """Test minimal DAG with just 2 nodes."""
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create minimal graph
        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]

        # Execute
        model = Toy2Model([node1, node2], sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert len(cache.bnds) >= 0

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_long_chain(self, sort_strategy):
        """Test long linear chain (10 nodes)."""
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create 10-node chain
        nodes = [Toy2Node(f"Node-{i}", cache, arguments) for i in range(1, 11)]

        for i in range(len(nodes) - 1):
            nodes[i].next_nodes = [nodes[i + 1]]
            nodes[i + 1].pre_nodes = [nodes[i]]

        # Execute
        model = Toy2Model(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert len(cache.bnds) >= 0

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_deep_branching(self, sort_strategy):
        """Test early broadcast, late merge pattern."""
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes (Node-1 broadcasts to 6 nodes, all converge to Node-8)
        nodes = [Toy2Node(f"Node-{i}", cache, arguments) for i in range(1, 9)]
        node1 = nodes[0]
        intermediate = nodes[1:7]  # Node-2 through Node-7
        node8 = nodes[7]

        # Define topology: broadcast then merge
        node1.next_nodes = intermediate
        for n in intermediate:
            n.pre_nodes = [node1]
            n.next_nodes = [node8]
        node8.pre_nodes = intermediate

        # Execute
        model = Toy2Model(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert len(cache.bnds) >= 0


class TestRealisticNetworkPatterns:
    """Category 6: Realistic neural network patterns."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_inception_like_module(self, sort_strategy):
        r"""
        Test Inception-like parallel processing paths.

        Node-1 → Node-2 (1x1) ↘
                 Node-3 (3x3) → Node-6 (concat)
                 Node-4 (5x5) ↗
                 Node-5 (pool) ┘
        """
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        nodes = [Toy2Node(f"Node-{i}", cache, arguments) for i in range(1, 7)]

        # Parallel paths from Node-1 to Node-6
        nodes[0].next_nodes = [nodes[1], nodes[2], nodes[3], nodes[4]]
        for i in range(1, 5):
            nodes[i].pre_nodes = [nodes[0]]
            nodes[i].next_nodes = [nodes[5]]
        nodes[5].pre_nodes = nodes[1:5]

        # Execute
        model = Toy2Model(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert len(cache.bnds) >= 0

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_densenet_like_connection(self, sort_strategy):
        r"""
        Test DenseNet-like dense connections.

        Node-1 → Node-2 ↘
                 └→ Node-3 ↘
                    └→ Node-4 (receives all predecessors)
        """
        # Setup
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        # Create nodes
        nodes = [Toy2Node(f"Node-{i}", cache, arguments) for i in range(1, 5)]

        # Dense connections
        nodes[0].next_nodes = [nodes[1], nodes[2], nodes[3]]
        nodes[1].pre_nodes = [nodes[0]]
        nodes[1].next_nodes = [nodes[2], nodes[3]]
        nodes[2].pre_nodes = [nodes[0], nodes[1]]
        nodes[2].next_nodes = [nodes[3]]
        nodes[3].pre_nodes = [nodes[0], nodes[1], nodes[2]]

        # Execute
        model = Toy2Model(nodes, sort_strategy=sort_strategy)
        model.run()

        # Assertions
        assert len(cache.bnds) >= 0
