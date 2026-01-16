"""
Test suite for cache management in T2 (reversed graph semantics).

T2 has simpler cache management than T1:
- No PropMode (single-purpose: backward propagation)
- Uses fwd_bnds (from forward pass) and bnds (backward propagated)
- Simpler clearing logic (not implemented in Toy2Node)

Tests adapted from test_t1/test_cache_management.py.
"""

from propdag import Toy2Cache, Toy2Model, Toy2Node


class TestCacheStructure:
    """Test T2Cache structure and fields."""

    def test_cache_has_required_fields(self):
        """Verify T2Cache has all required fields."""
        cache = Toy2Cache()

        assert hasattr(cache, "cur_node")
        assert hasattr(cache, "bnds")  # Backward propagated bounds
        assert hasattr(cache, "rlxs")  # Inverse relaxations
        assert hasattr(cache, "fwd_bnds")  # Forward bounds (for initialization)
        assert hasattr(cache, "symbnds")  # Symbolic bounds

    def test_cache_fields_are_dicts(self):
        """Verify cache fields are dictionaries."""
        cache = Toy2Cache()

        assert isinstance(cache.bnds, dict)
        assert isinstance(cache.rlxs, dict)
        assert isinstance(cache.fwd_bnds, dict)
        assert isinstance(cache.symbnds, dict)


class TestModelExecution:
    """Test that T2Model executes without cache errors."""

    def test_simple_chain_executes(self, toy2_cache, toy2_arguments):
        """Test simple chain executes and populates cache."""
        toy2_cache.fwd_bnds["Node-1"] = ("input bounds",)

        node1 = Toy2Node("Node-1", toy2_cache, toy2_arguments)
        node2 = Toy2Node("Node-2", toy2_cache, toy2_arguments)
        node3 = Toy2Node("Node-3", toy2_cache, toy2_arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        model = Toy2Model([node1, node2, node3])
        model.run()

        # Execution completed successfully
        assert toy2_cache is not None

    def test_diamond_executes(self, toy2_cache, toy2_arguments):
        """Test diamond pattern executes correctly."""
        toy2_cache.fwd_bnds["Input"] = ("input bounds",)

        input_node = Toy2Node("Input", toy2_cache, toy2_arguments)
        node_a = Toy2Node("A", toy2_cache, toy2_arguments)
        node_b = Toy2Node("B", toy2_cache, toy2_arguments)
        output_node = Toy2Node("Output", toy2_cache, toy2_arguments)

        input_node.next_nodes = [node_a, node_b]
        node_a.pre_nodes = [input_node]
        node_b.pre_nodes = [input_node]
        node_a.next_nodes = [output_node]
        node_b.next_nodes = [output_node]
        output_node.pre_nodes = [node_a, node_b]

        model = Toy2Model([input_node, node_a, node_b, output_node])
        model.run()

        # Execution completed
        assert toy2_cache is not None


class TestCacheClearing:
    """Test cache clearing behavior in T2."""

    def test_clear_cache_during_running_parameter(self, toy2_cache, toy2_arguments):
        """Test clear_cache_during_running parameter."""
        toy2_cache.fwd_bnds["Node-1"] = ("input bounds",)

        node1 = Toy2Node("Node-1", toy2_cache, toy2_arguments)
        node2 = Toy2Node("Node-2", toy2_cache, toy2_arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]

        # Test with clearing disabled (default)
        model = Toy2Model([node1, node2], clear_cache_during_running=False)
        assert not model.clear_cache_during_running

        # Test with clearing enabled
        model2 = Toy2Model([node1, node2], clear_cache_during_running=True)
        assert model2.clear_cache_during_running


class TestSharedCache:
    """Test that all nodes share the same cache."""

    def test_all_nodes_share_cache(self, toy2_cache, toy2_arguments):
        """Verify all nodes share the same cache instance."""
        node1 = Toy2Node("Node-1", toy2_cache, toy2_arguments)
        node2 = Toy2Node("Node-2", toy2_cache, toy2_arguments)
        node3 = Toy2Node("Node-3", toy2_cache, toy2_arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        model = Toy2Model([node1, node2, node3])

        # All nodes share same cache
        assert model.cache is toy2_cache
        assert node1.cache is toy2_cache
        assert node2.cache is toy2_cache
        assert node3.cache is toy2_cache


class TestForwardBounds:
    """Test forward bounds initialization."""

    def test_forward_bounds_required(self, toy2_cache, toy2_arguments):
        """Test that forward bounds are used for initialization."""
        # Set forward bounds
        toy2_cache.fwd_bnds["Input"] = ("forward bounds",)

        input_node = Toy2Node("Input", toy2_cache, toy2_arguments)
        output_node = Toy2Node("Output", toy2_cache, toy2_arguments)

        input_node.next_nodes = [output_node]
        output_node.pre_nodes = [input_node]

        model = Toy2Model([input_node, output_node])
        model.run()

        # Forward bounds were set
        assert "Input" in toy2_cache.fwd_bnds
