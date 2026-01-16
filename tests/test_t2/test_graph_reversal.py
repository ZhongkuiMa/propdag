"""
Test suite for graph reversal functionality in template2.

Tests the core innovation of T2: automatic graph edge reversal so that
"forward" propagation through the reversed graph achieves backward bound
propagation.
"""

import pytest

from propdag import Toy2Model, reverse_dag


def test_reverse_dag_simple_chain(toy2_node_factory):
    """Verify reverse_dag correctly reverses a simple chain."""
    # User builds: Input → Hidden → Output
    input_node = toy2_node_factory("Input")
    hidden_node = toy2_node_factory("Hidden")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [hidden_node]
    hidden_node.pre_nodes = [input_node]
    hidden_node.next_nodes = [output_node]
    output_node.pre_nodes = [hidden_node]

    # Before reversal
    assert len(input_node.pre_nodes) == 0
    assert len(output_node.next_nodes) == 0

    # Reverse
    user_input, user_output = reverse_dag([input_node, hidden_node, output_node])

    # After reversal: edges swapped
    assert user_input == input_node
    assert user_output == output_node
    assert len(output_node.pre_nodes) == 0  # Now the input
    assert len(input_node.next_nodes) == 0  # Now the output


def test_reverse_dag_diamond(toy2_node_factory):
    """Verify reverse_dag correctly reverses diamond topology."""
    # User builds: Input → A → Output
    #                  └→ B → ┘
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

    # Reverse
    user_input, user_output = reverse_dag([input_node, node_a, node_b, output_node])

    # After reversal
    assert user_input == input_node
    assert user_output == output_node
    assert len(output_node.pre_nodes) == 0  # Now the input
    assert len(input_node.next_nodes) == 0  # Now the output
    # A and B should have swapped connections
    assert input_node in node_a.next_nodes  # A now points to input
    assert input_node in node_b.next_nodes  # B now points to input
    assert output_node in node_a.pre_nodes  # A receives from output
    assert output_node in node_b.pre_nodes  # B receives from output


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
    # Build chain: Input → Hidden → Output
    input_node = toy2_node_factory("Input")
    hidden_node = toy2_node_factory("Hidden")
    output_node = toy2_node_factory("Output")

    input_node.next_nodes = [hidden_node]
    hidden_node.pre_nodes = [input_node]
    hidden_node.next_nodes = [output_node]
    output_node.pre_nodes = [hidden_node]

    # T2Model reverses and sorts
    model = Toy2Model([input_node, hidden_node, output_node])

    # After reversal, topological order should be Output → Hidden → Input
    assert model.nodes[0] == output_node
    assert model.nodes[1] == hidden_node
    assert model.nodes[2] == input_node


def test_single_node_accepted(toy2_node_factory):
    """Test that single node is accepted (it's both input and output)."""
    node = toy2_node_factory("SingleNode")

    # A single node is both input (no pre) and output (no next)
    user_input, user_output = reverse_dag([node])
    assert user_input == node
    assert user_output == node


def test_multiple_inputs_rejected(toy2_node_factory):
    """Test that multiple input nodes are rejected."""
    input1 = toy2_node_factory("Input1")
    input2 = toy2_node_factory("Input2")
    output = toy2_node_factory("Output")

    # Two inputs, one output
    input1.next_nodes = [output]
    input2.next_nodes = [output]
    output.pre_nodes = [input1, input2]

    with pytest.raises(ValueError, match=r"exactly one input"):
        reverse_dag([input1, input2, output])


def test_multiple_outputs_rejected(toy2_node_factory):
    """Test that multiple output nodes are rejected."""
    input_node = toy2_node_factory("Input")
    output1 = toy2_node_factory("Output1")
    output2 = toy2_node_factory("Output2")

    # One input, two outputs
    input_node.next_nodes = [output1, output2]
    output1.pre_nodes = [input_node]
    output2.pre_nodes = [input_node]

    with pytest.raises(ValueError, match=r"exactly one output"):
        reverse_dag([input_node, output1, output2])
