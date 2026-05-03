"""
Test suite for error handling and validation in T2.

Tests constraint enforcement and error detection in the reversed graph model.
Adapted from test_template/test_error_handling.py.
"""

import pytest

from propdag import Toy2Model
from test_template2._helpers import (
    build_chain_model_t2,
    build_diamond_model_t2,
    build_invalid_io_nodes_t2,
)

_INVALID_IO_CASES = [
    ("multi_input", r"exactly one input"),
    ("multi_output", r"exactly one output"),
    ("no_input", r"exactly one"),
]


class TestInputOutputConstraints:
    """Toy2Model rejects DAGs that lack exactly one input and one output."""

    @pytest.mark.parametrize(("scenario", "match"), _INVALID_IO_CASES)
    def test_invalid_io_raises_value_error(self, scenario, match):
        """Each invalid I/O topology raises ValueError with an informative message."""
        nodes = build_invalid_io_nodes_t2(scenario)
        with pytest.raises(ValueError, match=match):
            Toy2Model(nodes)


class TestCycleDetection:
    """Cycles raise ValueError during topological sorting."""

    @pytest.mark.parametrize("scenario", ["two_node_cycle", "three_node_cycle"])
    def test_cycle_is_detected(self, scenario):
        """Each cyclic topology raises ValueError during model construction."""
        nodes = build_invalid_io_nodes_t2(scenario)
        with pytest.raises(ValueError, match=r"cycle|topological|input"):
            Toy2Model(nodes, sort_strategy="bfs")


class TestBFSAndDFSErrorDetection:
    """Both BFS and DFS detect the same constraint violations."""

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_both_strategies_detect_multiple_inputs(self, sort_strategy):
        """Multi-input DAGs are rejected under both BFS and DFS."""
        nodes = build_invalid_io_nodes_t2("multi_input")
        with pytest.raises(ValueError, match=r"exactly one input"):
            Toy2Model(nodes, sort_strategy=sort_strategy)

    @pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
    def test_both_strategies_detect_cycles(self, sort_strategy):
        """Cyclic DAGs are rejected under both BFS and DFS."""
        nodes = build_invalid_io_nodes_t2("two_node_cycle")
        with pytest.raises(ValueError, match=r"cycle|topological|input"):
            Toy2Model(nodes, sort_strategy=sort_strategy)


class TestValidDAGsAccepted:
    """Valid DAGs construct without error."""

    def test_linear_chain_accepted(self):
        """A simple linear chain constructs successfully."""
        model, _, _ = build_chain_model_t2(3)
        assert isinstance(model, Toy2Model)
        assert model.sort_strategy == "bfs"
        assert len(model.nodes) == 3

    def test_diamond_accepted(self):
        """A diamond DAG constructs successfully."""
        model, _, _ = build_diamond_model_t2()
        assert isinstance(model, Toy2Model)
        assert model.sort_strategy == "bfs"
        assert len(model.nodes) == 4


class TestErrorMessageQuality:
    """Error messages are informative."""

    def test_multiple_input_error_mentions_node_names(self):
        """The multi-input error message references at least one offending node name."""
        nodes = build_invalid_io_nodes_t2("multi_input")
        with pytest.raises(ValueError, match=r"exactly one input") as excinfo:
            Toy2Model(nodes)
        assert "Node-1" in str(excinfo.value) or "Node-2" in str(excinfo.value)
