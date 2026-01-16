#!/usr/bin/env python3
"""Capture golden message sequences for T2 topology test cases.

Generates expected sequences for all 17 topologies x 2 strategies = 34 cases.

Usage
-----
From the test_t2 directory::

    cd tests/test_t2
    python capture_golden_sequences_t2.py > golden_sequences_t2.py
"""

import re
import sys
from io import StringIO
from pathlib import Path

# Add test_t1 to path to import TOPOLOGY_SPECS
test_t1_path = Path(__file__).parent.parent / "test_t1"
sys.path.insert(0, str(test_t1_path))

from test_verbose_output import TOPOLOGY_SPECS  # noqa: E402

from propdag import Toy2Argument, Toy2Cache, Toy2Model, Toy2Node  # noqa: E402


def build_toy2_topology(topology_name, cache=None):
    """Build a topology for Toy2Model.

    :param topology_name: Name of topology from TOPOLOGY_SPECS
    :type topology_name: str
    :param cache: Optional Toy2Cache instance (creates new if None)
    :type cache: Toy2Cache | None
    :returns: Tuple of (nodes_list, cache, input_name)
    :rtype: tuple[list[Toy2Node], Toy2Cache, str]
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


def capture_sequence(topology_name, sort_strategy):
    """Capture exact message sequence for a T2 topology.

    :param topology_name: Name of the topology from TOPOLOGY_SPECS
    :type topology_name: str
    :param sort_strategy: Topological sort strategy ("bfs" or "dfs")
    :type sort_strategy: str
    :returns: List of (node_name, phase) tuples representing execution sequence
    :rtype: list[tuple[str, str]]
    """
    # Build topology
    nodes, _cache, _input_name = build_toy2_topology(topology_name)

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        # Run model with verbose output
        model = Toy2Model(nodes, sort_strategy=sort_strategy, verbose=True)
        model.run()
    finally:
        sys.stdout = old_stdout

    # Parse captured output
    output = captured_output.getvalue()
    sequence = []
    pattern = re.compile(r"^\[([A-Z]+)\] ([\w-]+)\.")

    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("["):
            match = pattern.match(line)
            if match:
                phase, node_name = match.groups()
                sequence.append((node_name, phase))

    return sequence


def main():
    """Generate golden sequences for all T2 test cases."""
    print("# Golden sequences for all T2 topology test cases")
    print("# Generated automatically - do not edit manually")
    print()
    print("GOLDEN_SEQUENCES_T2 = {")

    for topology_name in TOPOLOGY_SPECS:
        for sort_strategy in ["bfs", "dfs"]:
            key = f"({topology_name!r}, {sort_strategy!r})"

            sequence = capture_sequence(topology_name, sort_strategy)

            print(f"    {key}: [")
            for node_name, phase in sequence:
                print(f"        ({node_name!r}, {phase!r}),")
            print("    ],")

    print("}")


if __name__ == "__main__":
    main()
