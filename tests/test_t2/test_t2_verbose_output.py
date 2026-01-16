"""
Test suite for verbose output format verification in T2 (Toy2 models).

Verifies that toy2 models produce correctly formatted messages that
accurately represent the template2 logic flow.
"""

import sys
from pathlib import Path

import pytest

from propdag import Toy2Argument, Toy2Cache, Toy2Model, Toy2Node

# Add test_t1 to path to import TOPOLOGY_SPECS
test_t1_path = Path(__file__).parent.parent / "test_t1"
sys.path.insert(0, str(test_t1_path))

# Reuse helper classes from T1 tests
from test_verbose_output import (  # noqa: E402
    TOPOLOGY_SPECS,
    CacheStateVerifier,
    VerboseOutputCapture,
)

# Import golden sequences (try relative first, then absolute)
try:
    from .golden_sequences_t2 import GOLDEN_SEQUENCES_T2  # noqa: TID252
except ImportError:
    from golden_sequences_t2 import GOLDEN_SEQUENCES_T2


class TestToy2NodeVerboseOutput:
    """Test verbose output format for Toy2Node."""

    def test_reversed_graph_verbose_output(self, capsys):
        """Test verbose output for reversed graph execution."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        model = Toy2Model([node1, node2, node3], verbose=True)
        model.run()

        captured = capsys.readouterr()
        output = VerboseOutputCapture(captured.out)

        # After reversal, Node-3 is the input
        node3_messages = output.get_messages_for_node("Node-3")
        assert len(node3_messages) > 0

        # First message should be INIT
        parsed = output.verify_message_format(node3_messages[0])
        assert parsed["phase"] == "INIT"
        assert "reversed_input" in parsed["context"]

        # Verify all messages follow format
        for line in output.lines:
            output.verify_message_format(line)

    def test_t2_phase_order(self, capsys):
        """Verify T2 phases occur in correct order."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]

        model = Toy2Model([node1, node2], verbose=True)
        model.run()

        captured = capsys.readouterr()
        output = VerboseOutputCapture(captured.out)

        # Get Node-1 messages (after reversal, this is output node)
        node1_messages = output.get_messages_for_node("Node-1")
        phases = [output.verify_message_format(msg)["phase"] for msg in node1_messages]

        # T2 expected phases: RELAX, PROPAGATE, CACHE, COMPUTE
        assert "RELAX" in phases
        assert "PROPAGATE" in phases
        assert "CACHE" in phases
        assert "COMPUTE" in phases

    def test_t2_node_name_verification(self, capsys):
        """Verify node names in messages match expected nodes."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        model = Toy2Model([node1, node2, node3], verbose=True)
        model.run()

        captured = capsys.readouterr()
        output = VerboseOutputCapture(captured.out)

        # Verify Node-2 messages all have correct node name
        node2_messages = output.get_messages_for_node("Node-2")
        for msg in node2_messages:
            output.verify_node_name(msg, "Node-2")
            # Passes if no assertion error

    def test_t2_cache_state_verification(self, capsys):
        """Verify cache state changes correctly after CACHE messages."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        model = Toy2Model([node1, node2, node3], verbose=True)
        model.run()

        captured = capsys.readouterr()
        output = VerboseOutputCapture(captured.out)
        verifier = CacheStateVerifier()

        # Verify cache state after each CACHE message for Node-1
        node1_messages = output.get_messages_for_node("Node-1")
        for msg in node1_messages:
            parsed = output.verify_message_format(msg)
            if parsed["phase"] == "CACHE":
                verifier.verify_cache_after_message(cache, msg, parsed)

        # Verify final cache state
        # After reversed execution, bnds should contain all nodes
        verifier.verify_cache_contains(cache, "bnds", ["Node-1", "Node-2", "Node-3"])

    def test_t2_diamond_pattern(self, capsys):
        """Test T2 with diamond merge pattern."""
        cache = Toy2Cache()
        cache.fwd_bnds["Node-1"] = ("forward bounds",)
        arguments = Toy2Argument()

        node1 = Toy2Node("Node-1", cache, arguments)
        node2 = Toy2Node("Node-2", cache, arguments)
        node3 = Toy2Node("Node-3", cache, arguments)
        node4 = Toy2Node("Node-4", cache, arguments)

        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node4]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node2, node3]

        model = Toy2Model([node1, node2, node3, node4], verbose=True)
        model.run()

        captured = capsys.readouterr()
        output = VerboseOutputCapture(captured.out)

        # Verify all messages follow format
        for line in output.lines:
            output.verify_message_format(line)

        # After reversal, Node-4 is input
        node4_messages = output.get_messages_for_node("Node-4")
        assert len(node4_messages) > 0
        # First message should be INIT
        parsed = output.verify_message_format(node4_messages[0])
        assert parsed["phase"] == "INIT"


def build_toy2_topology(topology_name, cache=None):
    """
    Build a topology for Toy2Model.

    Args:
        topology_name: Name of topology from TOPOLOGY_SPECS
        cache: Optional Toy2Cache (creates new if None)

    Returns:
        Tuple of (nodes_list, cache, input_name)
    """
    spec = TOPOLOGY_SPECS[topology_name]
    if cache is None:
        cache = Toy2Cache()

    arguments = Toy2Argument()

    # Create nodes
    nodes = [Toy2Node(f"Node-{i}", cache, arguments) for i in range(1, spec["nodes"] + 1)]

    # Build connections
    for from_idx, to_idx in spec["connections"]:
        from_node = nodes[from_idx - 1]
        to_node = nodes[to_idx - 1]

        if to_node not in from_node.next_nodes:
            from_node.next_nodes.append(to_node)
        if from_node not in to_node.pre_nodes:
            to_node.pre_nodes.append(from_node)

    # Set forward bounds for first node (T2 needs fwd_bnds for intersection)
    input_name = "Node-1"
    cache.fwd_bnds[input_name] = ("forward bounds",)

    return nodes, cache, input_name


def verify_node_first_execution_order_t2(nodes, actual_sequence):
    """
    Verify nodes' first execution respects topological order in T2 (reversed graph).

    Each node's FIRST message (INIT or RELAX) must appear after ALL predecessors'
    FIRST message in the reversed graph.

    Args:
        nodes: List of Toy2Node instances
        actual_sequence: List of (node_name, phase) tuples
    """
    # Track first execution (INIT or RELAX) index for each node
    node_first_exec = {}

    for idx, (node_name, phase) in enumerate(actual_sequence):
        # Track when node starts execution (INIT for reversed input, RELAX for others)
        if phase in {"INIT", "RELAX"} and node_name not in node_first_exec:
            node_first_exec[node_name] = idx

    # Build node map for quick lookup
    node_map = {node.name: node for node in nodes}

    # Verify each node's first execution after all predecessors' first execution
    for node_name in node_first_exec:
        node = node_map[node_name]
        node_start = node_first_exec[node_name]

        # In reversed graph, pre_nodes should execute first
        for pre_node in node.pre_nodes:
            pre_start = node_first_exec.get(pre_node.name, -1)
            assert pre_start < node_start, (
                f"T2 first execution order violation: Node '{node_name}' started first execution "
                f"at index {node_start} before predecessor '{pre_node.name}' (in reversed graph) "
                f"started at index {pre_start}"
            )


def verify_valid_phase_sequences_t2(actual_sequence):
    """
    Verify each node's phases follow valid transitions in T2.

    Args:
        actual_sequence: List of (node_name, phase) tuples
    """
    # Group messages by node
    node_phases = {}
    for node_name, phase in actual_sequence:
        if node_name not in node_phases:
            node_phases[node_name] = []
        node_phases[node_name].append(phase)

    # Define valid phase sets
    valid_phases = {"INIT", "RELAX", "PROPAGATE", "CACHE", "COMPUTE", "CLEAR"}

    # Verify all phases are valid
    for node_name, phases in node_phases.items():
        for phase in phases:
            assert phase in valid_phases, f"Node '{node_name}' has invalid phase '{phase}'"


@pytest.mark.parametrize("topology_name", list(TOPOLOGY_SPECS.keys()))
@pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
class TestToy2TopologyVerboseOutput:
    """Verify T2 message format AND execution logic across all topologies.

    This test class ensures that Toy2Model verbose output works correctly
    for all graph topologies, verifying BOTH message format and algorithm
    correctness in the reversed graph execution model:
    - Message format compliance
    - Topological execution order (in reversed graph)
    - Cache state correctness
    - Output/input node bounds

    Coverage: 17 topologies x 2 sort strategies = 34 test cases
    """

    def test_all_messages_follow_format(self, topology_name, sort_strategy, capsys):
        """Verify exact message sequence against golden output for T2.

        This test validates:
        1. All messages match format: [PHASE] NodeName.method() | operation → target [context]
        2. Exact sequence matches golden output (complete execution correctness)
        3. Output node has valid bounds in cache
        4. Input node fwd_bnds are preserved
        """
        # Build topology
        nodes, cache, input_name = build_toy2_topology(topology_name)

        # Identify output node (node with no next_nodes after reversal)
        output_nodes = [n for n in nodes if len(n.next_nodes) == 0]
        assert len(output_nodes) == 1, (
            f"Expected 1 output node for {topology_name}, got {len(output_nodes)}: "
            f"{[n.name for n in output_nodes]}"
        )
        output_name = output_nodes[0].name

        # Run model with verbose output
        model = Toy2Model(nodes, sort_strategy=sort_strategy, verbose=True)
        model.run()

        # Capture output
        captured = capsys.readouterr()
        output = VerboseOutputCapture(captured.out)
        verifier = CacheStateVerifier()

        # === LAYER 1: Message Format Verification ===
        assert len(output.lines) > 0, (
            f"No messages captured for {topology_name} (strategy={sort_strategy})"
        )

        # Verify ALL messages follow format
        for line in output.lines:
            parsed = output.verify_message_format(line)
            # Each message should have valid phase, node name, method, operation, target
            assert parsed["phase"] in {"INIT", "RELAX", "PROPAGATE", "CACHE", "COMPUTE", "CLEAR"}
            assert parsed["node"] is not None
            assert parsed["method"] is not None
            assert parsed["operation"] is not None
            assert parsed["target"] is not None

        # === LAYER 2: Exact Sequence Verification ===
        # Extract actual sequence
        actual_sequence = []
        for line in output.lines:
            parsed = output.verify_message_format(line)
            actual_sequence.append((parsed["node"], parsed["phase"]))

        # Look up expected golden sequence
        expected_sequence = GOLDEN_SEQUENCES_T2[(topology_name, sort_strategy)]

        # Verify exact match
        assert actual_sequence == expected_sequence, (
            f"Sequence mismatch for {topology_name} (strategy={sort_strategy}):\n"
            f"Expected ({len(expected_sequence)} steps):\n{expected_sequence}\n"
            f"Got ({len(actual_sequence)} steps):\n{actual_sequence}"
        )

        # === LAYER 3: Cache State Verification ===
        # T2 uses bnds for backward bounds after reversal
        verifier.verify_cache_contains(cache, "bnds", [output_name, input_name])

        # Verify bounds are not None
        assert cache.bnds[output_name] is not None, (
            f"Output node {output_name} has None bounds for {topology_name}"
        )

        # Input in original graph has fwd_bnds
        assert input_name in cache.fwd_bnds, (
            f"Input node {input_name} missing fwd_bnds for {topology_name}"
        )
