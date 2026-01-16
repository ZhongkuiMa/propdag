#!/usr/bin/env python3
"""Capture golden message sequences for T1 topology test cases.

Generates expected sequences for all 17 topologies x 2 modes x 2 strategies = 68 cases.

Usage
-----
From the test_t1 directory::

    cd tests/test_t1
    python capture_golden_sequences.py > golden_sequences.py
"""

import re
import sys
from io import StringIO

from test_verbose_output import TOPOLOGY_SPECS, build_topology

from propdag import PropMode, ToyModel


def capture_sequence(topology_name, prop_mode, sort_strategy):
    """Capture exact message sequence for a topology.

    :param topology_name: Name of the topology from TOPOLOGY_SPECS
    :type topology_name: str
    :param prop_mode: Propagation mode (FORWARD or BACKWARD)
    :type prop_mode: PropMode
    :param sort_strategy: Topological sort strategy ("bfs" or "dfs")
    :type sort_strategy: str
    :returns: List of (node_name, phase) tuples representing execution sequence
    :rtype: list[tuple[str, str]]
    """
    # Build topology
    nodes, _cache, _input_name = build_topology(topology_name, prop_mode)

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        # Run model with verbose output
        model = ToyModel(nodes, sort_strategy=sort_strategy, verbose=True)
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
    """Generate golden sequences for all test cases."""
    print("# Golden sequences for all topology test cases")
    print("# Generated automatically - do not edit manually")
    print()
    print("GOLDEN_SEQUENCES = {")

    for topology_name in TOPOLOGY_SPECS:
        for prop_mode in [PropMode.FORWARD, PropMode.BACKWARD]:
            for sort_strategy in ["bfs", "dfs"]:
                mode_str = "FORWARD" if prop_mode == PropMode.FORWARD else "BACKWARD"
                key = f"({topology_name!r}, {mode_str!r}, {sort_strategy!r})"

                sequence = capture_sequence(topology_name, prop_mode, sort_strategy)

                print(f"    {key}: [")
                for node_name, phase in sequence:
                    print(f"        ({node_name!r}, {phase!r}),")
                print("    ],")

    print("}")


if __name__ == "__main__":
    main()
