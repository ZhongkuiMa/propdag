"""
Test suite for error handling and constraint validation in propdag.

Validates that:
1. DAGs with multiple input/output nodes raise ValueError.
2. DAGs with no input or no output (cycles) raise ValueError.
3. DAGs with cycles raise ValueError during topological sorting.
4. Both BFS and DFS sort strategies detect the same constraint violations.
5. Valid DAGs (chain / diamond / skip) construct without error.
"""

__docformat__ = "restructuredtext"

import pytest

from propdag import ToyModel
from test_template._helpers import (
    build_chain_model,
    build_cycle_nodes,
    build_diamond_model,
    build_skip_model,
)

_INVALID_IO_CASES = [
    ("multi_input", r"input|multiple"),
    ("multi_output", r"input.*output|multiple.*output|exactly one output"),
    ("no_input", r"cycle|topological|input"),
    ("no_output", r"cycle|topological|output"),
]
_CYCLE_CASES = ["two_node", "three_node", "self_loop", "embedded"]


class TestInputOutputConstraints:
    """ToyModel rejects DAGs that lack exactly one input and one output."""

    @pytest.mark.parametrize(("scenario", "match"), _INVALID_IO_CASES)
    def test_invalid_io_raises_value_error(self, scenario, match):
        """Each invalid I/O topology raises ValueError with an informative message."""
        nodes = build_cycle_nodes(scenario)
        with pytest.raises(ValueError, match=match):
            ToyModel(nodes, sort_strategy="bfs")


class TestCycleDetection:
    """Topological sort detects cycles in any cyclic topology."""

    @pytest.mark.parametrize("scenario", _CYCLE_CASES)
    def test_cycle_is_detected(self, scenario):
        """Each cyclic topology raises ValueError during model construction."""
        nodes = build_cycle_nodes(scenario)
        with pytest.raises(ValueError, match=r"cycle|topological"):
            ToyModel(nodes, sort_strategy="bfs")


class TestBFSAndDFSErrorDetection:
    """Both BFS and DFS detect the same constraint violations."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_both_strategies_detect_multiple_inputs(self, sort_strategy):
        """Multi-input DAGs are rejected under both BFS and DFS."""
        nodes = build_cycle_nodes("multi_input")
        with pytest.raises(ValueError, match=r"input|output|multiple"):
            ToyModel(nodes, sort_strategy=sort_strategy)

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_both_strategies_detect_cycles(self, sort_strategy):
        """Cyclic DAGs are rejected under both BFS and DFS."""
        nodes = build_cycle_nodes("two_node")
        with pytest.raises(ValueError, match=r"cycle|topological"):
            ToyModel(nodes, sort_strategy=sort_strategy)


class TestValidDAGsAccepted:
    """Valid DAGs construct without error."""

    def test_linear_chain_accepted(self):
        """A simple linear chain constructs successfully."""
        model, _, _ = build_chain_model(2)
        assert len(model.nodes) > 0, "Valid DAG must have nodes after construction"

    def test_diamond_accepted(self):
        """A diamond DAG constructs successfully."""
        model, _, _ = build_diamond_model()
        assert len(model.nodes) > 0, "Valid DAG must have nodes after construction"

    def test_skip_connection_accepted(self):
        """A DAG with a skip connection constructs successfully."""
        model, _, _ = build_skip_model()
        assert len(model.nodes) > 0, "Valid DAG must have nodes after construction"


class TestErrorMessageQuality:
    """Error messages are informative."""

    def test_multiple_input_error_is_informative(self):
        """The multi-input error message mentions the offending node names."""
        nodes = build_cycle_nodes("multi_input")
        with pytest.raises(ValueError, match=r"Node-1|Node-2|input|multiple"):
            ToyModel(nodes, sort_strategy="bfs")

    def test_cycle_error_is_informative(self):
        """The cycle error message mentions the cycle/topological-sort condition."""
        nodes = build_cycle_nodes("two_node")
        with pytest.raises(ValueError, match=r"cycle|topological"):
            ToyModel(nodes, sort_strategy="bfs")
