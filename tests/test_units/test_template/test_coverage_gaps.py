"""
Test suite to improve code coverage for propdag.

Focuses on previously untested code paths:
1. Verbose mode in TModel (print statements)
2. Invalid sort_strategy error handling
3. Input node forward pass (skip behavior)
4. Property getters (sort_strategy / cache / arguments)
5. Cache clearing in verbose mode
"""

import pytest

from propdag import PropMode, ToyModel
from test_template._helpers import build_chain_model


class TestVerboseMode:
    """Verbose output during TModel execution."""

    def test_verbose_forward_pass(self, capsys):
        """``verbose=True`` prints forward-pass messages for every non-input node."""
        model, _, _ = build_chain_model(3, verbose=True)
        model.run()
        captured = capsys.readouterr()
        assert "Forward pass" in captured.out
        for name in ("Node-1", "Node-2", "Node-3"):
            assert name in captured.out

    def test_verbose_backward_propagation(self, capsys):
        """``verbose=True`` in BACKWARD mode prints back-substitute or forward-pass messages."""
        model, _, _ = build_chain_model(3, prop_mode=PropMode.BACKWARD, verbose=True)
        model.run()
        captured = capsys.readouterr()
        assert "Back-substitute" in captured.out or "Forward pass" in captured.out

    def test_non_verbose_mode(self, capsys):
        """``verbose=False`` (default) suppresses forward-pass messages."""
        model, _, _ = build_chain_model(3, verbose=False)
        model.run()
        captured = capsys.readouterr()
        assert "Forward pass" not in captured.out


class TestInvalidSortStrategy:
    """Error handling for invalid sort strategies."""

    def test_raises_error(self):
        """An unknown sort_strategy raises ValueError."""
        # Build chain with default bfs, then re-instantiate with bad strategy.
        _, _, nodes = build_chain_model(3)
        with pytest.raises(ValueError, match="Unknown sort strategy"):
            ToyModel(nodes, sort_strategy="invalid")

    @pytest.mark.parametrize("strategy", ["dfs", "bfs"])
    def test_valid_sort_strategies(self, strategy):
        """Both ``dfs`` and ``bfs`` strategies are accepted."""
        model, _, _ = build_chain_model(3, sort_strategy=strategy)
        assert model.sort_strategy == strategy


class TestInputNodeHandling:
    """Input nodes are skipped during forward pass."""

    def test_skip(self, capsys):
        """The input node prints a skip diagnostic in verbose mode."""
        model, _, _ = build_chain_model(3, verbose=True)
        model.run()
        captured = capsys.readouterr()
        assert "[INIT]" in captured.out
        assert "skip" in captured.out
        assert "input_node" in captured.out


class TestModelProperties:
    """Property getters on TModel."""

    @pytest.mark.parametrize("strategy", ["dfs", "bfs"])
    def test_sort_strategy_property(self, strategy):
        """``model.sort_strategy`` returns the strategy passed at construction."""
        model, _, _ = build_chain_model(3, sort_strategy=strategy)
        assert model.sort_strategy == strategy

    def test_cache_property_identity(self):
        """``model.cache`` returns the shared cache object the nodes were built with."""
        model, cache, _ = build_chain_model(3)
        assert model.cache is cache

    def test_arguments_property_identity(self):
        """``model.arguments`` returns the shared arguments object."""
        model, _, nodes = build_chain_model(3)
        assert model.arguments is nodes[0].argument


class TestCacheClearingVerbose:
    """Cache clearing diagnostics in verbose mode."""

    def test_forward_logged(self, capsys):
        """Cache clearing prints either a clear-cache or forward-pass diagnostic."""
        model, _, _ = build_chain_model(3, verbose=True)
        model.run()
        captured = capsys.readouterr()
        assert "Clear forward cache" in captured.out or "Forward pass" in captured.out
