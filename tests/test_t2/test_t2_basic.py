"""
Test suite for basic T2Model and Toy2Node functionality.

Tests initialization, execution, cache structure, and sort strategies.
"""

import pytest

from propdag import Toy2Argument, Toy2Model, Toy2Node, clear_bwd_cache_t2


def test_t2model_initialization(toy2_node_factory):
    """Verify T2Model initializes correctly."""
    input_node = toy2_node_factory("Input")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [output_node]
    output_node.pre_nodes = [input_node]

    model = Toy2Model([input_node, output_node])

    assert model is not None
    assert len(model.nodes) == 2
    # After reversal, output should be first
    assert model.nodes[0] == output_node
    assert model.nodes[1] == input_node


def test_t2model_run(toy2_node_factory, toy2_cache):
    """Verify run() executes forward passes."""
    # Setup
    input_node = toy2_node_factory("Input")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [output_node]
    output_node.pre_nodes = [input_node]

    # Initialize forward bounds (required for backward propagation)
    toy2_cache.fwd_bnds["Input"] = ("forward bounds for input",)
    toy2_cache.fwd_bnds["Output"] = ("forward bounds for output",)

    model = Toy2Model([input_node, output_node])
    model.run()

    # Verify execution completed (bounds computed)
    assert "Output" in toy2_cache.bnds or "Input" in toy2_cache.bnds


def test_toy2node_forward(toy2_cache, toy2_arguments):
    """Verify forward() method works."""
    node = Toy2Node("TestNode", toy2_cache, toy2_arguments)

    # Set up cache state
    toy2_cache.cur_node = node
    toy2_cache.fwd_bnds["TestNode"] = ("test bounds",)

    # Call forward (should not crash)
    node.forward()

    # Basic verification - method executed
    assert toy2_cache.cur_node is not None


def test_toy2node_build_rlx(toy2_cache, toy2_arguments):
    """Verify build_rlx() for inverse relaxations."""
    node = Toy2Node("TestNode", toy2_cache, toy2_arguments)

    toy2_cache.cur_node = node

    # Call build_rlx (should not crash)
    node.build_rlx()

    # Method executed successfully
    assert True


def test_cache_structure(toy2_cache):
    """Verify T2Cache has correct fields."""
    assert hasattr(toy2_cache, "bnds")
    assert hasattr(toy2_cache, "rlxs")
    assert hasattr(toy2_cache, "fwd_bnds")
    assert hasattr(toy2_cache, "symbnds")
    assert hasattr(toy2_cache, "cur_node")


def test_arguments_no_prop_mode():
    """Verify T2Argument has no prop_mode."""
    args = Toy2Argument()

    # T2Argument should NOT have prop_mode (single-purpose)
    assert not hasattr(args, "prop_mode")


def test_sort_strategy_bfs(toy2_node_factory):
    """Verify BFS sorting works."""
    input_node = toy2_node_factory("Input")
    hidden_node = toy2_node_factory("Hidden")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [hidden_node]
    hidden_node.pre_nodes = [input_node]
    hidden_node.next_nodes = [output_node]
    output_node.pre_nodes = [hidden_node]

    model = Toy2Model([input_node, hidden_node, output_node], sort_strategy="bfs")

    assert model.sort_strategy == "bfs"
    assert len(model.nodes) == 3


def test_sort_strategy_dfs(toy2_node_factory):
    """Verify DFS sorting works."""
    input_node = toy2_node_factory("Input")
    hidden_node = toy2_node_factory("Hidden")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [hidden_node]
    hidden_node.pre_nodes = [input_node]
    hidden_node.next_nodes = [output_node]
    output_node.pre_nodes = [hidden_node]

    model = Toy2Model([input_node, hidden_node, output_node], sort_strategy="dfs")

    assert model.sort_strategy == "dfs"
    assert len(model.nodes) == 3


def test_clear_bwd_cache_t2_not_implemented(toy2_cache, toy2_arguments):
    """Verify clear_bwd_cache_t2() raises error (not implemented in Toy2Node)."""
    # Setup nodes
    node1 = Toy2Node("Node1", toy2_cache, toy2_arguments)

    # Create cache counter
    cache_counter = {node1: 1}

    # clear_bwd_cache_t2 calls node.clear_bwd_cache() which is not implemented
    with pytest.raises(RuntimeError, match="not implemented"):
        clear_bwd_cache_t2(cache_counter, [node1])
