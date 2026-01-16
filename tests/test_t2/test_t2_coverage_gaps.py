"""
Test suite to improve code coverage for T2 (Template2/Toy2).

Focuses on previously untested code paths:
1. Verbose mode in T2Model (print statements)
2. Invalid sort_strategy error handling
3. Input node handling (in reversed graph)
4. Property getters
5. Cache clearing behavior

Adapted from test_t1/test_coverage_gaps.py for Template2/Toy2.
"""

import pytest

from propdag import Toy2Argument, Toy2Cache, Toy2Model, Toy2Node


class TestVerboseMode:
    """Test verbose output during T2Model execution."""

    def test_verbose_forward_pass(self, capsys):
        """Test that verbose=True prints forward pass messages."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        # Build linear chain: Node-1 -> Node-2 -> Node-3
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Create model with verbose=True
        model = Toy2Model(nodes, sort_strategy="bfs", verbose=True)

        # Run the model
        model.run()

        # Capture output
        captured = capsys.readouterr()

        # Verify that execution messages are printed
        # T2 prints "Propagate bounds through" for each node
        assert "Propagate bounds through" in captured.out
        assert "Running Toy2Model" in captured.out
        assert "Node-1" in captured.out or "Node-2" in captured.out or "Node-3" in captured.out

    def test_non_verbose_mode(self, capsys):
        """Test that verbose=False produces no forward pass messages."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Create model with verbose=False (default)
        model = Toy2Model(nodes, sort_strategy="bfs", verbose=False)

        # Run the model
        model.run()

        # Capture output
        captured = capsys.readouterr()

        # Verify that verbose messages are NOT printed
        assert "Propagate bounds through" not in captured.out
        assert "Running Toy2Model" not in captured.out


class TestInvalidSortStrategy:
    """Test error handling for invalid sort strategies."""

    def test_invalid_sort_strategy_raises_error(self):
        """Test that invalid sort_strategy raises ValueError."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Should raise ValueError for unknown sort strategy
        with pytest.raises(ValueError, match="Unknown sort strategy"):
            Toy2Model(nodes, sort_strategy="invalid")

    def test_valid_sort_strategies(self):
        """Test that both 'dfs' and 'bfs' sort strategies work."""
        for strategy in ["dfs", "bfs"]:
            cache = Toy2Cache()
            cache.fwd_bnds["Node-1"] = ("input bounds",)
            arguments = Toy2Argument()

            node1 = Toy2Node("Node-1", cache, arguments)
            node2 = Toy2Node("Node-2", cache, arguments)
            node3 = Toy2Node("Node-3", cache, arguments)

            # Build linear chain
            node1.next_nodes = [node2]
            node2.pre_nodes = [node1]
            node2.next_nodes = [node3]
            node3.pre_nodes = [node2]

            nodes = [node1, node2, node3]

            # Should not raise for valid strategies
            model = Toy2Model(nodes, sort_strategy=strategy)
            assert model.sort_strategy == strategy


class TestInputNodeHandling:
    """Test handling of all nodes in reversed graph (no skipping)."""

    def test_all_nodes_processed(self, capsys):
        """Test that all nodes are processed in reversed graph (no skipping)."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Create model with verbose to see all node processing
        model = Toy2Model(nodes, sort_strategy="bfs", verbose=True)

        # Run the model
        model.run()

        # Capture output
        captured = capsys.readouterr()

        # Verify all nodes are processed (no skipping in T2)
        # After reversal: Node-3 (output) → Node-2 → Node-1 (input)
        assert "Propagate bounds through Node-3" in captured.out
        assert "Propagate bounds through Node-2" in captured.out
        assert "Propagate bounds through Node-1" in captured.out


class TestModelProperties:
    """Test property getters in T2Model."""

    def test_model_sort_strategy_property(self):
        """Test sort_strategy property getter."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Test both strategies
        for strategy in ["dfs", "bfs"]:
            model = Toy2Model(nodes, sort_strategy=strategy)
            assert model.sort_strategy == strategy

    def test_model_cache_property(self):
        """Test cache property getter."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        model = Toy2Model(nodes, sort_strategy="bfs")
        assert model.cache is cache

    def test_model_arguments_property(self):
        """Test arguments property getter."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        model = Toy2Model(nodes, sort_strategy="bfs")
        assert model.arguments is arguments


class TestCacheClearingParameter:
    """Test cache clearing parameter in T2Model."""

    def test_clear_cache_parameter_true(self):
        """Test that clear_cache_during_running=True is respected."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]

        nodes = [node1, node2]

        # Create model with cache clearing enabled
        model = Toy2Model(nodes, sort_strategy="bfs", clear_cache_during_running=True)
        assert model.clear_cache_during_running is True

    def test_clear_cache_parameter_false(self):
        """Test that clear_cache_during_running=False is respected."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]

        nodes = [node1, node2]

        # Create model with cache clearing disabled (default)
        model = Toy2Model(nodes, sort_strategy="bfs", clear_cache_during_running=False)
        assert model.clear_cache_during_running is False


class TestGraphReversalFlag:
    """Test that graph reversal flag is correctly set."""

    def test_user_input_and_output_stored(self):
        """Test that user_input and user_output are stored after reversal."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("input bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Create model (reverses graph automatically)
        model = Toy2Model(nodes, sort_strategy="bfs")

        # Verify user_input and user_output are stored
        assert hasattr(model, "user_input")
        assert hasattr(model, "user_output")
        assert model.user_input is not None
        assert model.user_output is not None
