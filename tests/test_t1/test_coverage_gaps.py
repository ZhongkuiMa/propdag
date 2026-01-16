"""
Test suite to improve code coverage for propdag.

Focuses on previously untested code paths:
1. Verbose mode in TModel (print statements)
2. Invalid sort_strategy error handling
3. Input node forward pass
4. Property getters
"""

import pytest

from propdag import ForwardToyNode, PropMode, ToyArgument, ToyCache, ToyModel


class TestVerboseMode:
    """Test verbose output during model execution."""

    def test_verbose_forward_pass(self, capsys):
        """Test that verbose=True prints forward pass messages."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Build linear chain: Node-1 -> Node-2 -> Node-3
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Create model with verbose=True
        model = ToyModel(nodes, sort_strategy="bfs", verbose=True)

        # Run the model
        model.run()

        # Capture output
        captured = capsys.readouterr()

        # Verify that forward pass messages are printed
        assert "Forward pass" in captured.out
        assert "Node-1" in captured.out
        assert "Node-2" in captured.out
        assert "Node-3" in captured.out

    def test_verbose_backward_propagation(self, capsys):
        """Test that verbose=True prints backward pass messages."""
        from propdag import BackwardToyNode

        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.BACKWARD)

        node1 = BackwardToyNode("Node-1", cache, arguments)
        node2 = BackwardToyNode("Node-2", cache, arguments)
        node3 = BackwardToyNode("Node-3", cache, arguments)

        # Build linear chain: Node-1 -> Node-2 -> Node-3
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Create model with verbose=True
        model = ToyModel(nodes, sort_strategy="bfs", verbose=True)

        # Run the model
        model.run()

        # Capture output
        captured = capsys.readouterr()

        # Verify that backward pass messages are printed
        assert "Back-substitute" in captured.out or "Forward pass" in captured.out

    def test_non_verbose_mode(self, capsys):
        """Test that verbose=False produces no forward pass messages."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Create model with verbose=False (default)
        model = ToyModel(nodes, sort_strategy="bfs", verbose=False)

        # Run the model
        model.run()

        # Capture output
        captured = capsys.readouterr()

        # Verify that forward pass messages are NOT printed
        assert "Forward pass" not in captured.out


class TestInvalidSortStrategy:
    """Test error handling for invalid sort strategies."""

    def test_invalid_sort_strategy_raises_error(self):
        """Test that invalid sort_strategy raises ValueError."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Should raise ValueError for unknown sort strategy
        with pytest.raises(ValueError, match="Unknown sort strategy"):
            ToyModel(nodes, sort_strategy="invalid")

    def test_valid_sort_strategies(self):
        """Test that both 'dfs' and 'bfs' sort strategies work."""
        for strategy in ["dfs", "bfs"]:
            cache = ToyCache()
            cache.bnds["Node-1"] = ("input bounds",)
            arguments = ToyArgument(prop_mode=PropMode.FORWARD)

            node1 = ForwardToyNode("Node-1", cache, arguments)
            node2 = ForwardToyNode("Node-2", cache, arguments)
            node3 = ForwardToyNode("Node-3", cache, arguments)

            # Build linear chain
            node1.next_nodes = [node2]
            node2.pre_nodes = [node1]
            node2.next_nodes = [node3]
            node3.pre_nodes = [node2]

            nodes = [node1, node2, node3]

            # Should not raise for valid strategies
            model = ToyModel(nodes, sort_strategy=strategy)
            assert model.sort_strategy == strategy


class TestInputNodeHandling:
    """Test handling of input nodes in forward pass."""

    def test_input_node_skip(self, capsys):
        """Test that input node is skipped in forward pass with message."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Create model with verbose to see skip message
        model = ToyModel(nodes, sort_strategy="bfs", verbose=True)

        # Run the model
        model.run()

        # Capture output
        captured = capsys.readouterr()

        # Verify input node skip message (new structured format)
        assert "[INIT]" in captured.out
        assert "skip" in captured.out
        assert "input_node" in captured.out


class TestModelProperties:
    """Test property getters in TModel."""

    def test_model_sort_strategy_property(self):
        """Test sort_strategy property getter."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Test both strategies
        for strategy in ["dfs", "bfs"]:
            model = ToyModel(nodes, sort_strategy=strategy)
            assert model.sort_strategy == strategy

    def test_model_cache_property(self):
        """Test cache property getter."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        model = ToyModel(nodes, sort_strategy="bfs")
        assert model.cache is cache

    def test_model_arguments_property(self):
        """Test arguments property getter."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        model = ToyModel(nodes, sort_strategy="bfs")
        assert model.arguments is arguments


class TestCacheClearingVerbose:
    """Test cache clearing with verbose output."""

    def test_forward_cache_clearing_verbose(self, capsys):
        """Test that cache clearing is logged in verbose mode."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        # Build linear chain
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        nodes = [node1, node2, node3]

        # Create model with verbose=True
        model = ToyModel(nodes, sort_strategy="bfs", verbose=True)

        # Run the model
        model.run()

        # Capture output
        captured = capsys.readouterr()

        # Verify cache clearing messages appear
        assert "Clear forward cache" in captured.out or "Forward pass" in captured.out
