"""
Test suite for reverse_dag() helper function.

Tests validation logic and error messages in the reverse_dag() function.
"""

import pytest

from propdag import reverse_dag


def test_reverse_dag_validation_no_input(toy2_node_factory):
    """Test error when no input node exists."""
    # Create cycle: A → B → A (no input)
    node_a = toy2_node_factory("A")
    node_b = toy2_node_factory("B")

    node_a.next_nodes = [node_b]
    node_a.pre_nodes = [node_b]
    node_b.next_nodes = [node_a]
    node_b.pre_nodes = [node_a]

    with pytest.raises(ValueError, match=r"exactly one input.*zero"):
        reverse_dag([node_a, node_b])


def test_reverse_dag_validation_no_output(toy2_node_factory):
    """Test error when no output node exists."""
    # Create cycle: A → B → A (no input or output)
    node_a = toy2_node_factory("A")
    node_b = toy2_node_factory("B")

    node_a.next_nodes = [node_b]
    node_a.pre_nodes = [node_b]
    node_b.next_nodes = [node_a]
    node_b.pre_nodes = [node_a]

    # Cycle means no input/output, will error on "no input" first
    with pytest.raises(ValueError, match=r"exactly one input.*zero|exactly one output.*zero"):
        reverse_dag([node_a, node_b])


def test_reverse_dag_validation_multi_input(toy2_node_factory):
    """Test error when multiple input nodes exist."""
    input1 = toy2_node_factory("Input1")
    input2 = toy2_node_factory("Input2")
    output = toy2_node_factory("Output")

    # Two inputs → one output
    input1.next_nodes = [output]
    input2.next_nodes = [output]
    output.pre_nodes = [input1, input2]

    with pytest.raises(ValueError, match=r"found 2 nodes.*Input1.*Input2"):
        reverse_dag([input1, input2, output])


def test_reverse_dag_validation_multi_output(toy2_node_factory):
    """Test error when multiple output nodes exist."""
    input_node = toy2_node_factory("Input")
    output1 = toy2_node_factory("Output1")
    output2 = toy2_node_factory("Output2")

    # One input → two outputs
    input_node.next_nodes = [output1, output2]
    output1.pre_nodes = [input_node]
    output2.pre_nodes = [input_node]

    with pytest.raises(ValueError, match=r"found 2 nodes.*Output1.*Output2"):
        reverse_dag([input_node, output1, output2])


def test_reverse_dag_verbose_output(toy2_node_factory, capsys):
    """Test verbose output during reversal."""
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
    """Test that calling reverse_dag twice restores original."""
    input_node = toy2_node_factory("Input")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [output_node]
    output_node.pre_nodes = [input_node]

    # Store original state
    orig_input_pre = len(input_node.pre_nodes)
    orig_input_next = len(input_node.next_nodes)
    orig_output_pre = len(output_node.pre_nodes)
    orig_output_next = len(output_node.next_nodes)

    # Reverse twice
    reverse_dag([input_node, output_node])
    reverse_dag([input_node, output_node])

    # Should be back to original
    assert len(input_node.pre_nodes) == orig_input_pre
    assert len(input_node.next_nodes) == orig_input_next
    assert len(output_node.pre_nodes) == orig_output_pre
    assert len(output_node.next_nodes) == orig_output_next
