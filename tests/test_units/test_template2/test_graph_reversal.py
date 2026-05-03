"""
Test suite for graph reversal functionality in template2.

Tests the core innovation of T2: automatic graph edge reversal so that
"forward" propagation through the reversed graph achieves backward bound
propagation. Also covers the validation logic and error messages of
``reverse_dag`` (formerly tested separately in ``test_reverse_dag.py``).
"""

import pytest

from propdag import Toy2Model, reverse_dag


def test_reverse_dag_simple_chain(toy2_node_factory):
    """Verify reverse_dag correctly reverses a simple chain."""
    input_node = toy2_node_factory("Input")
    hidden_node = toy2_node_factory("Hidden")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [hidden_node]
    hidden_node.pre_nodes = [input_node]
    hidden_node.next_nodes = [output_node]
    output_node.pre_nodes = [hidden_node]

    assert len(input_node.pre_nodes) == 0
    assert len(output_node.next_nodes) == 0

    user_input, user_output = reverse_dag([input_node, hidden_node, output_node])

    assert user_input == input_node
    assert user_output == output_node
    assert len(output_node.pre_nodes) == 0
    assert len(input_node.next_nodes) == 0


def test_reverse_dag_diamond(toy2_node_factory):
    """Verify reverse_dag correctly reverses diamond topology."""
    input_node = toy2_node_factory("Input")
    node_a = toy2_node_factory("A")
    node_b = toy2_node_factory("B")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [node_a, node_b]
    node_a.pre_nodes = [input_node]
    node_b.pre_nodes = [input_node]
    node_a.next_nodes = [output_node]
    node_b.next_nodes = [output_node]
    output_node.pre_nodes = [node_a, node_b]

    user_input, user_output = reverse_dag([input_node, node_a, node_b, output_node])

    assert user_input == input_node
    assert user_output == output_node
    assert len(output_node.pre_nodes) == 0
    assert len(input_node.next_nodes) == 0
    assert input_node in node_a.next_nodes
    assert input_node in node_b.next_nodes
    assert output_node in node_a.pre_nodes
    assert output_node in node_b.pre_nodes


def test_reverse_dag_returns_correct_nodes(toy2_node_factory):
    """Verify reverse_dag returns (user_input, user_output) tuple."""
    input_node = toy2_node_factory("Input")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [output_node]
    output_node.pre_nodes = [input_node]

    user_input, user_output = reverse_dag([input_node, output_node])

    assert user_input == input_node
    assert user_output == output_node


def test_nodes_reordered_after_reversal(toy2_node_factory):
    """Verify T2Model reorders nodes correctly after reversal."""
    input_node = toy2_node_factory("Input")
    hidden_node = toy2_node_factory("Hidden")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [hidden_node]
    hidden_node.pre_nodes = [input_node]
    hidden_node.next_nodes = [output_node]
    output_node.pre_nodes = [hidden_node]

    model = Toy2Model([input_node, hidden_node, output_node])

    assert model.nodes[0] == output_node
    assert model.nodes[1] == hidden_node
    assert model.nodes[2] == input_node


def test_single_node_accepted(toy2_node_factory):
    """A single node is both input and output, so reverse_dag accepts it."""
    node = toy2_node_factory("SingleNode")

    user_input, user_output = reverse_dag([node])
    assert user_input == node
    assert user_output == node


def _build_invalid_io_topology(scenario: str, factory):
    """Construct invalid I/O topologies for parametrized rejection tests."""
    if scenario == "multi_input":
        a, b, c = factory("Input1"), factory("Input2"), factory("Output")
        a.next_nodes = [c]
        b.next_nodes = [c]
        c.pre_nodes = [a, b]
        return [a, b, c]
    if scenario == "multi_output":
        a, b, c = factory("Input"), factory("Output1"), factory("Output2")
        a.next_nodes = [b, c]
        b.pre_nodes = [a]
        c.pre_nodes = [a]
        return [a, b, c]
    if scenario == "no_input":
        a, b = factory("A"), factory("B")
        a.next_nodes = [b]
        a.pre_nodes = [b]
        b.next_nodes = [a]
        b.pre_nodes = [a]
        return [a, b]
    raise ValueError(f"Unknown scenario: {scenario}")


@pytest.mark.parametrize(
    ("scenario", "match"),
    [
        ("multi_input", r"exactly one input.*found 2"),
        ("multi_output", r"exactly one output.*found 2"),
        ("no_input", r"exactly one input.*zero"),
    ],
)
def test_reverse_dag_rejects_invalid_io(toy2_node_factory, scenario, match):
    """``reverse_dag`` raises ValueError when graph lacks exactly one I/O node."""
    nodes = _build_invalid_io_topology(scenario, toy2_node_factory)
    with pytest.raises(ValueError, match=match):
        reverse_dag(nodes)


def test_reverse_dag_verbose_output(toy2_node_factory, capsys):
    """Verbose mode prints diagnostic messages before and after reversal."""
    input_node = toy2_node_factory("Input")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [output_node]
    output_node.pre_nodes = [input_node]

    reverse_dag([input_node, output_node], verbose=True)

    captured = capsys.readouterr()
    assert "Before reversal" in captured.out
    assert "Input=Input" in captured.out
    assert "Output=Output" in captured.out
    assert "Reversed edges" in captured.out


def test_reverse_dag_idempotence(toy2_node_factory):
    """Calling reverse_dag twice restores the original edge orientation."""
    input_node = toy2_node_factory("Input")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [output_node]
    output_node.pre_nodes = [input_node]

    orig_input_pre = len(input_node.pre_nodes)
    orig_input_next = len(input_node.next_nodes)
    orig_output_pre = len(output_node.pre_nodes)
    orig_output_next = len(output_node.next_nodes)

    reverse_dag([input_node, output_node])
    reverse_dag([input_node, output_node])

    assert len(input_node.pre_nodes) == orig_input_pre
    assert len(input_node.next_nodes) == orig_input_next
    assert len(output_node.pre_nodes) == orig_output_pre
    assert len(output_node.next_nodes) == orig_output_next
