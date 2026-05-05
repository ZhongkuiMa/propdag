"""
Test suite to improve code coverage for T2 (Template2/Toy2).

Focuses on previously untested code paths:
1. Verbose mode in T2Model
2. Invalid sort_strategy error handling
3. Input node handling in the reversed graph (no skipping)
4. Property getters
5. Cache clearing parameter behavior
6. Graph-reversal book-keeping (user_input / user_output)
"""

import pytest

from propdag import Toy2Model
from test_template2._helpers import build_chain_model_t2


class TestVerboseMode:
    """Verbose output during T2Model execution."""

    def test_verbose_forward_pass(self, capsys):
        """``verbose=True`` emits per-node propagation diagnostics."""
        model, _, _ = build_chain_model_t2(3, verbose=True)
        model.run()
        captured = capsys.readouterr()
        assert "Propagate bounds through" in captured.out
        assert "Running Toy2Model" in captured.out

    def test_non_verbose_mode(self, capsys):
        """``verbose=False`` (default) suppresses diagnostics."""
        model, _, _ = build_chain_model_t2(3, verbose=False)
        model.run()
        captured = capsys.readouterr()
        assert "Propagate bounds through" not in captured.out
        assert "Running Toy2Model" not in captured.out


class TestInvalidSortStrategy:
    """Error handling for invalid sort strategies."""

    def test_raises_error(self):
        """An unknown sort_strategy raises ValueError."""
        _, _, nodes = build_chain_model_t2(3)
        with pytest.raises(ValueError, match="Unknown sort strategy"):
            Toy2Model(nodes, sort_strategy="invalid")

    @pytest.mark.parametrize("strategy", ["dfs", "bfs"])
    def test_valid_sort_strategies(self, strategy):
        """Both ``dfs`` and ``bfs`` strategies are accepted."""
        model, _, _ = build_chain_model_t2(3, sort_strategy=strategy)
        assert model.sort_strategy == strategy


class TestInputNodeHandling:
    """All nodes are processed in the reversed graph (no skipping)."""

    def test_all_nodes_processed(self, capsys):
        """Every node emits a propagation diagnostic in verbose mode."""
        model, _, _ = build_chain_model_t2(3, verbose=True)
        model.run()
        captured = capsys.readouterr()
        for name in ("Node-1", "Node-2", "Node-3"):
            assert f"Propagate bounds through {name}" in captured.out


class TestModelProperties:
    """Property getters on T2Model."""

    @pytest.mark.parametrize("strategy", ["dfs", "bfs"])
    def test_sort_strategy_property(self, strategy):
        """``model.sort_strategy`` returns the strategy passed at construction."""
        model, _, _ = build_chain_model_t2(3, sort_strategy=strategy)
        assert model.sort_strategy == strategy

    def test_cache_property_identity(self):
        """``model.cache`` returns the shared cache the nodes were built with."""
        model, cache, _ = build_chain_model_t2(3)
        assert model.cache is cache

    def test_arguments_property_identity(self):
        """``model.arguments`` returns the shared arguments object."""
        model, _, nodes = build_chain_model_t2(3)
        assert model.arguments is nodes[0].argument


class TestCacheClearingParameter:
    """``clear_cache_during_running`` parameter is forwarded to the model."""

    @pytest.mark.parametrize("clear", [True, False])
    def test_clear(self, clear):
        """The flag round-trips through model construction."""
        model, _, _ = build_chain_model_t2(2, clear_cache_during_running=clear)
        assert model.clear_cache_during_running is clear


class TestGraphReversalFlag:
    """Graph-reversal book-keeping is exposed on the model."""

    def test_user_input_and_output_stored(self):
        """``user_input`` / ``user_output`` are populated after construction."""
        model, _, _ = build_chain_model_t2(3)
        assert hasattr(model, "user_input")
        assert hasattr(model, "user_output")
        assert model.user_input is not None
        assert model.user_output is not None
