"""
Test suite for verbose output format verification in T1 (Toy models).

Verifies that toy models produce correctly formatted messages that
accurately represent the template logic flow.
"""

import re

import pytest

from propdag import BackwardToyNode, ForwardToyNode, PropMode, ToyArgument, ToyCache, ToyModel

# Import golden sequences (try relative first, then absolute for when imported from other test modules)
try:
    from .golden_sequences import GOLDEN_SEQUENCES  # noqa: TID252
except ImportError:
    from golden_sequences import GOLDEN_SEQUENCES


class VerboseOutputCapture:
    """Helper to capture and parse verbose output."""

    def __init__(self, captured_output):
        """Initialize with captured output, filtering for formatted messages only."""
        # Only keep lines that start with our phase markers (formatted messages)
        self.lines = [
            line.strip()
            for line in captured_output.strip().split("\n")
            if line.strip() and line.strip().startswith("[")
        ]

    def get_messages_for_node(self, node_name):
        """Get all messages for a specific node."""
        return [line for line in self.lines if f"] {node_name}." in line]

    def get_messages_by_phase(self, phase):
        """Get all messages for a specific phase."""
        pattern = re.escape(f"[{phase}]")
        return [line for line in self.lines if re.match(pattern, line)]

    def verify_message_format(self, message):
        """
        Verify a message matches the standard format.

        Format: [PHASE] NodeName.method() | operation → target [context]
        """
        # Pattern: [PHASE] Node.method() | operation → target [optional_context]
        # Allow spaces in operation (e.g., "compute relaxation")
        # Allow hyphens in target (e.g., symbnds[Node-1])
        pattern = (
            r"^\[([A-Z]+)\] ([\w-]+)\.([\w_]+)\(\) \| ([\w_ ]+) → ([\w\[\]\.-]+)(?: \[(.+)\])?$"
        )
        match = re.match(pattern, message)

        assert match is not None, f"Message doesn't match format: {message}"

        phase, node, method, operation, target, context = match.groups()

        # Validate phase
        valid_phases = {"INIT", "RELAX", "PROPAGATE", "CACHE", "COMPUTE", "CLEAR"}
        assert phase in valid_phases, f"Invalid phase: {phase}"

        return {
            "phase": phase,
            "node": node,
            "method": method,
            "operation": operation,
            "target": target,
            "context": context,
        }

    def verify_node_name(self, message, expected_node_name):
        """
        Verify that the node name in the message matches expected node.

        Args:
            message: The verbose output message
            expected_node_name: The expected node name (e.g., "Node-1")

        Returns:
            Parsed message dict if verification passes
        """
        parsed = self.verify_message_format(message)
        assert parsed["node"] == expected_node_name, (
            f"Node name mismatch: expected '{expected_node_name}', "
            f"got '{parsed['node']}' in message: {message}"
        )
        return parsed

    def verify_execution_sequence(self, expected_sequence):
        """
        Verify messages appear in expected execution sequence.

        Args:
            expected_sequence: List of (node_name, phase) tuples in expected order
        """
        actual_sequence = []
        for line in self.lines:
            parsed = self.verify_message_format(line)
            actual_sequence.append((parsed["node"], parsed["phase"]))

        # Check that expected sequence is a subsequence of actual
        exp_idx = 0
        for act_node, act_phase in actual_sequence:
            if exp_idx < len(expected_sequence):
                exp_node, exp_phase = expected_sequence[exp_idx]
                if act_node == exp_node and act_phase == exp_phase:
                    exp_idx += 1

        assert exp_idx == len(expected_sequence), (
            f"Expected sequence not found.\n"
            f"Expected: {expected_sequence}\n"
            f"Actual: {actual_sequence}"
        )


class CacheStateVerifier:
    """Helper to verify cache state after message execution."""

    @staticmethod
    def verify_cache_after_message(cache, message, parsed_msg):
        """
        Verify cache state matches the operation described in message.

        Args:
            cache: ToyCache or Toy2Cache instance
            message: The verbose output message
            parsed_msg: Parsed message dict from verify_message_format()
        """
        phase = parsed_msg["phase"]
        target = parsed_msg["target"]
        operation = parsed_msg["operation"]

        # Extract cache key from target (e.g., "bnds[Node-1]" -> ("bnds", "Node-1"))
        cache_match = re.match(r"cache\.([\w]+)\[([\w-]+)\]", target)
        if cache_match:
            cache_attr, cache_key = cache_match.groups()
        else:
            # Target might be "bnds[Node-1]" without "cache." prefix
            cache_match = re.match(r"([\w]+)\[([\w-]+)\]", target)
            if cache_match:
                cache_attr, cache_key = cache_match.groups()
            else:
                # Cannot parse target, skip verification
                return

        # Verify cache state based on operation
        if operation in ["store", "update", "cache"]:
            # After CACHE phase with store operation, cache should contain the key
            if phase == "CACHE":
                cache_dict = getattr(cache, cache_attr, None)
                assert cache_dict is not None, f"Cache attribute '{cache_attr}' not found"
                assert cache_key in cache_dict, (
                    f"After message '{message}', "
                    f"cache.{cache_attr} should contain key '{cache_key}', "
                    f"but keys are: {list(cache_dict.keys())}"
                )

        elif operation == "clear" and phase == "CLEAR":
            # After CLEAR phase, cache should NOT contain the key
            cache_dict = getattr(cache, cache_attr, None)
            if cache_dict is not None:
                assert cache_key not in cache_dict, (
                    f"After message '{message}', "
                    f"cache.{cache_attr} should NOT contain key '{cache_key}', "
                    f"but keys are: {list(cache_dict.keys())}"
                )

    @staticmethod
    def verify_cache_contains(cache, cache_attr, expected_keys):
        """
        Verify cache contains all expected keys.

        Args:
            cache: ToyCache or Toy2Cache instance
            cache_attr: Cache attribute name (e.g., "bnds", "symbnds", "rlxs")
            expected_keys: List of expected keys
        """
        cache_dict = getattr(cache, cache_attr, None)
        assert cache_dict is not None, f"Cache attribute '{cache_attr}' not found"

        for key in expected_keys:
            assert key in cache_dict, (
                f"Expected key '{key}' not found in cache.{cache_attr}. "
                f"Available keys: {list(cache_dict.keys())}"
            )

    @staticmethod
    def verify_cache_empty(cache, cache_attr):
        """
        Verify cache attribute is empty.

        Args:
            cache: ToyCache or Toy2Cache instance
            cache_attr: Cache attribute name
        """
        cache_dict = getattr(cache, cache_attr, None)
        assert cache_dict is not None, f"Cache attribute '{cache_attr}' not found"
        assert len(cache_dict) == 0, (
            f"Expected cache.{cache_attr} to be empty, but contains keys: {list(cache_dict.keys())}"
        )


class TestForwardNodeVerboseOutput:
    """Test verbose output format for ForwardToyNode."""

    def test_linear_chain_verbose_output(self, capsys):
        """Test verbose output for simple linear chain."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        model = ToyModel([node1, node2, node3], verbose=True)
        model.run()

        captured = capsys.readouterr()
        output = VerboseOutputCapture(captured.out)
        verifier = CacheStateVerifier()

        # Verify Node-1 (input node) skips forward
        node1_messages = output.get_messages_for_node("Node-1")
        assert len(node1_messages) == 1

        # CRITICAL: Verify node name in message
        parsed = output.verify_node_name(node1_messages[0], "Node-1")
        assert parsed["phase"] == "INIT"
        assert parsed["operation"] == "skip"
        assert "no_predecessors" in parsed["context"]

        # Verify Node-2 (hidden node) full execution
        node2_messages = output.get_messages_for_node("Node-2")

        # Should have: RELAX, PROPAGATE, CACHE, COMPUTE, CACHE
        phases = [output.verify_message_format(msg)["phase"] for msg in node2_messages]
        assert "RELAX" in phases
        assert "PROPAGATE" in phases
        assert "CACHE" in phases
        assert "COMPUTE" in phases

        # CRITICAL: Verify node names for all Node-2 messages
        for msg in node2_messages:
            parsed = output.verify_message_format(msg)
            assert parsed["node"] == "Node-2", f"Expected Node-2, got {parsed['node']}"

        # CRITICAL: Verify cache state after each CACHE message
        for msg in node2_messages:
            parsed = output.verify_message_format(msg)
            if parsed["phase"] == "CACHE":
                # Verify cache state matches the message
                verifier.verify_cache_after_message(cache, msg, parsed)

        # After execution, verify cache contains all expected keys
        # Note: Input nodes have bnds but may not create symbnds (they skip forward pass)
        verifier.verify_cache_contains(cache, "bnds", ["Node-1", "Node-2", "Node-3"])
        # Only non-input nodes create symbnds in forward mode
        verifier.verify_cache_contains(cache, "symbnds", ["Node-2", "Node-3"])

        # Verify execution sequence
        output.verify_execution_sequence(
            [
                ("Node-1", "INIT"),
                ("Node-2", "RELAX"),
                ("Node-2", "PROPAGATE"),
                ("Node-2", "COMPUTE"),
                ("Node-3", "RELAX"),
                ("Node-3", "PROPAGATE"),
                ("Node-3", "COMPUTE"),
            ]
        )

    def test_diamond_pattern_verbose_output(self, capsys):
        """Test verbose output for diamond merge pattern."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)
        node3 = ForwardToyNode("Node-3", cache, arguments)
        node4 = ForwardToyNode("Node-4", cache, arguments)

        node1.next_nodes = [node2, node3]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node4]
        node3.pre_nodes = [node1]
        node3.next_nodes = [node4]
        node4.pre_nodes = [node2, node3]

        model = ToyModel([node1, node2, node3, node4], verbose=True)
        model.run()

        captured = capsys.readouterr()
        output = VerboseOutputCapture(captured.out)

        # Verify Node-4 (merge point) receives from both predecessors
        node4_messages = output.get_messages_for_node("Node-4")
        propagate_msgs = [msg for msg in node4_messages if "[PROPAGATE]" in msg]

        assert len(propagate_msgs) > 0
        parsed = output.verify_message_format(propagate_msgs[0])

        # Context should mention both predecessors
        context = parsed["context"]
        assert "Node-2" in context
        assert "Node-3" in context

    def test_all_messages_follow_format(self, capsys):
        """Verify ALL messages follow the standard format."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        # Build moderately complex graph
        nodes = [ForwardToyNode(f"Node-{i}", cache, arguments) for i in range(1, 6)]

        # Linear chain with branch
        nodes[0].next_nodes = [nodes[1]]
        nodes[1].pre_nodes = [nodes[0]]
        nodes[1].next_nodes = [nodes[2], nodes[3]]
        nodes[2].pre_nodes = [nodes[1]]
        nodes[3].pre_nodes = [nodes[1]]
        nodes[2].next_nodes = [nodes[4]]
        nodes[3].next_nodes = [nodes[4]]
        nodes[4].pre_nodes = [nodes[2], nodes[3]]

        model = ToyModel(nodes, verbose=True)
        model.run()

        captured = capsys.readouterr()
        output = VerboseOutputCapture(captured.out)

        # Verify EVERY message follows format
        for line in output.lines:
            output.verify_message_format(line)  # Will assert if format is wrong


class TestBackwardNodeVerboseOutput:
    """Test verbose output format for BackwardToyNode."""

    def test_backward_mode_message_format(self, capsys):
        """Test backward mode produces correctly formatted messages."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.BACKWARD)

        node1 = BackwardToyNode("Node-1", cache, arguments)
        node2 = BackwardToyNode("Node-2", cache, arguments)
        node3 = BackwardToyNode("Node-3", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]
        node2.next_nodes = [node3]
        node3.pre_nodes = [node2]

        model = ToyModel([node1, node2, node3], verbose=True)
        model.run()

        captured = capsys.readouterr()
        output = VerboseOutputCapture(captured.out)

        # Verify backward substitution messages exist
        bwd_messages = [msg for msg in output.lines if "backward()" in msg]
        assert len(bwd_messages) > 0

        # Verify all messages follow format
        for line in output.lines:
            output.verify_message_format(line)


class TestPhaseSequencing:
    """Test that phases occur in correct logical order."""

    def test_forward_node_phase_order(self, capsys):
        """Verify phases occur in order: RELAX → PROPAGATE → CACHE → COMPUTE → CACHE."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)

        node1 = ForwardToyNode("Node-1", cache, arguments)
        node2 = ForwardToyNode("Node-2", cache, arguments)

        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]

        model = ToyModel([node1, node2], verbose=True)
        model.run()

        captured = capsys.readouterr()
        output = VerboseOutputCapture(captured.out)

        # Get Node-2 messages (non-input node)
        node2_messages = output.get_messages_for_node("Node-2")
        phases = [output.verify_message_format(msg)["phase"] for msg in node2_messages]

        # Expected phase order (with duplicate CACHE phases)
        expected_phase_sequence = ["RELAX", "PROPAGATE", "CACHE", "COMPUTE", "CACHE"]

        # Verify the exact sequence matches (allowing for exact duplicates)
        assert phases == expected_phase_sequence, (
            f"Expected phase sequence {expected_phase_sequence}, but got {phases}"
        )


# Topology Definitions: Each topology defined by node count and connections
# Connections are tuples (from_node, to_node) using 1-based indexing
TOPOLOGY_SPECS = {
    # Basic Structures
    "linear_chain": {
        "nodes": 4,
        "connections": [(1, 2), (2, 3), (3, 4)],
    },
    "y_shape": {
        "nodes": 4,
        "connections": [(1, 2), (1, 3), (2, 4), (3, 4)],
    },
    "sequential_branches": {
        "nodes": 4,
        "connections": [(1, 2), (2, 3), (2, 4), (3, 4)],
    },
    # Skip Connections
    "single_skip_connection": {
        "nodes": 4,
        "connections": [(1, 2), (2, 3), (3, 4), (1, 4)],
    },
    "multiple_skip_connections": {
        "nodes": 6,
        "connections": [(1, 2), (2, 3), (3, 4), (4, 6), (1, 5), (5, 6), (3, 6)],
    },
    "nested_skip_connections": {
        "nodes": 4,
        "connections": [(1, 2), (2, 3), (3, 4), (2, 4), (1, 4)],
    },
    # Multi-Input
    "same_source_twice": {
        "nodes": 3,
        "connections": [(1, 2), (1, 2), (2, 3)],  # Node-1 appears twice as input to Node-2
    },
    "diamond_pattern": {
        "nodes": 4,
        "connections": [(1, 2), (1, 3), (2, 4), (3, 4)],
    },
    "wide_merge": {
        "nodes": 6,
        "connections": [(1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (3, 6), (4, 6), (5, 6)],
    },
    # Complex Branching
    "wide_broadcast": {
        "nodes": 5,
        "connections": [(1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)],
    },
    "asymmetric_tree": {
        "nodes": 9,
        "connections": [
            (1, 2),
            (1, 3),
            (2, 4),
            (2, 5),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 8),
            (7, 8),
            (8, 9),
        ],
    },
    "multiple_merge_points": {
        "nodes": 7,
        "connections": [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7)],
    },
    # Boundary Cases
    "minimal_dag": {
        "nodes": 2,
        "connections": [(1, 2)],
    },
    "long_chain": {
        "nodes": 15,
        "connections": [(i, i + 1) for i in range(1, 15)],
    },
    "deep_branching": {
        "nodes": 8,
        "connections": [
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (2, 8),
            (3, 8),
            (4, 8),
            (5, 8),
            (6, 8),
            (7, 8),
        ],
    },
    # Realistic Networks
    "inception_like_module": {
        "nodes": 10,
        "connections": [
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),  # 1-to-4 broadcast
            (2, 6),
            (3, 7),
            (4, 8),
            (5, 9),  # Parallel processing
            (6, 10),
            (7, 10),
            (8, 10),
            (9, 10),  # 4-to-1 merge
        ],
    },
    "densenet_like_connection": {
        "nodes": 5,
        "connections": [
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 5),
        ],
    },
}


def build_topology(topology_name, prop_mode, cache=None):
    """
    Build a topology by name.

    Args:
        topology_name: Name of topology from TOPOLOGY_SPECS
        prop_mode: PropMode.FORWARD or PropMode.BACKWARD
        cache: Optional ToyCache (creates new if None)

    Returns:
        Tuple of (nodes_list, cache, input_name)
    """
    spec = TOPOLOGY_SPECS[topology_name]
    if cache is None:
        cache = ToyCache()

    arguments = ToyArgument(prop_mode=prop_mode)
    node_class = ForwardToyNode if prop_mode == PropMode.FORWARD else BackwardToyNode

    # Create nodes
    nodes = [node_class(f"Node-{i}", cache, arguments) for i in range(1, spec["nodes"] + 1)]

    # Build connections
    for from_idx, to_idx in spec["connections"]:
        from_node = nodes[from_idx - 1]
        to_node = nodes[to_idx - 1]

        if to_node not in from_node.next_nodes:
            from_node.next_nodes.append(to_node)
        if from_node not in to_node.pre_nodes:
            to_node.pre_nodes.append(from_node)

    # Set input bounds for first node
    input_name = "Node-1"
    cache.bnds[input_name] = ("input bounds",)

    return nodes, cache, input_name


def verify_node_first_execution_order(nodes, actual_sequence):
    """
    Verify nodes' first execution respects topological order.

    Each node's FIRST message (INIT or RELAX) must appear after ALL predecessors'
    FIRST message. This works for both FORWARD and BACKWARD modes.

    Note: In BACKWARD mode, nodes may be revisited later for backward substitution,
    but their initial execution still follows topological order.

    Args:
        nodes: List of TNode instances
        actual_sequence: List of (node_name, phase) tuples
    """
    # Track first execution (INIT or RELAX) index for each node
    node_first_exec = {}

    for idx, (node_name, phase) in enumerate(actual_sequence):
        # Track when node starts execution (INIT for input, RELAX for others)
        if phase in {"INIT", "RELAX"} and node_name not in node_first_exec:
            node_first_exec[node_name] = idx

    # Build node map
    node_map = {node.name: node for node in nodes}

    # Verify each node's first execution after all predecessors' first execution
    for node_name in node_first_exec:
        node = node_map[node_name]
        node_start = node_first_exec[node_name]

        for pre_node in node.pre_nodes:
            pre_start = node_first_exec.get(pre_node.name, -1)
            assert pre_start < node_start, (
                f"First execution order violation: Node '{node_name}' started first execution at "
                f"index {node_start} before predecessor '{pre_node.name}' started at index {pre_start}"
            )


def verify_valid_phase_sequences(actual_sequence):
    """
    Verify each node's phases follow valid transitions.

    Valid phase orders:
    - FORWARD: INIT (input) or RELAX → PROPAGATE → CACHE → COMPUTE → CACHE
    - BACKWARD: More complex with forward init + backward substitution

    This checks that phases don't appear in impossible orders.
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
@pytest.mark.parametrize("prop_mode", [PropMode.FORWARD, PropMode.BACKWARD])
@pytest.mark.parametrize("sort_strategy", ["bfs", "dfs"])
class TestTopologyVerboseOutput:
    """Verify message format AND execution logic across all topology patterns.

    This test class ensures that verbose output works correctly for all graph
    topologies, verifying BOTH message format and algorithm correctness:
    - Message format compliance
    - Topological execution order
    - Cache state correctness
    - Output/input node bounds

    Coverage: 17 topologies x 2 propagation modes x 2 sort strategies = 68 test cases
    """

    def test_all_messages_follow_format(self, topology_name, prop_mode, sort_strategy, capsys):
        """Verify exact message sequence against golden output for given topology.

        This test validates:
        1. All messages match format: [PHASE] NodeName.method() | operation → target [context]
        2. Exact sequence matches golden output (complete execution correctness)
        3. Output node has valid bounds in cache
        4. Input node bounds are preserved
        """
        # Build topology
        nodes, cache, input_name = build_topology(topology_name, prop_mode)

        # Identify output node (node with no next_nodes)
        output_nodes = [n for n in nodes if len(n.next_nodes) == 0]
        assert len(output_nodes) == 1, (
            f"Expected 1 output node for {topology_name}, got {len(output_nodes)}: "
            f"{[n.name for n in output_nodes]}"
        )
        output_name = output_nodes[0].name

        # Run model with verbose output
        model = ToyModel(nodes, sort_strategy=sort_strategy, verbose=True)
        model.run()

        # Capture output
        captured = capsys.readouterr()
        output = VerboseOutputCapture(captured.out)
        verifier = CacheStateVerifier()

        # === LAYER 1: Message Format Verification ===
        assert len(output.lines) > 0, (
            f"No messages captured for {topology_name} (mode={prop_mode}, strategy={sort_strategy})"
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
        mode_str = "FORWARD" if prop_mode == PropMode.FORWARD else "BACKWARD"
        expected_sequence = GOLDEN_SEQUENCES[(topology_name, mode_str, sort_strategy)]

        # Verify exact match
        assert actual_sequence == expected_sequence, (
            f"Sequence mismatch for {topology_name} (mode={mode_str}, strategy={sort_strategy}):\n"
            f"Expected ({len(expected_sequence)} steps):\n{expected_sequence}\n"
            f"Got ({len(actual_sequence)} steps):\n{actual_sequence}"
        )

        # === LAYER 3: Cache State Verification ===
        # Like test_dag_topologies.py - verify output and input nodes have bounds
        verifier.verify_cache_contains(cache, "bnds", [output_name, input_name])

        # Verify bounds are not None
        assert cache.bnds[output_name] is not None, (
            f"Output node {output_name} has None bounds for {topology_name}"
        )
        assert cache.bnds[input_name] is not None, (
            f"Input node {input_name} has None bounds for {topology_name}"
        )
