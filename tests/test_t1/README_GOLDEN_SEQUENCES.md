# Golden Sequences for T1 Tests

This directory contains golden sequence files for verifying exact execution order in T1 (Toy model) tests.

## Files

- **`golden_sequences.py`**: Contains 68 golden sequences for all topology test cases
  - 17 topologies × 2 propagation modes (FORWARD/BACKWARD) × 2 sort strategies (BFS/DFS)
  - Generated automatically - do not edit manually

- **`capture_golden_sequences.py`**: Script to regenerate golden sequences
  - Use this when you need to update golden sequences after code changes

## Regenerating Golden Sequences

When you make changes to the propagation algorithm or template logic, you need to regenerate the golden sequences:

```bash
cd tests/test_t1
python capture_golden_sequences.py > golden_sequences.py
```

**Important:** Only regenerate golden sequences when you've intentionally changed the execution flow. If a test fails due to sequence mismatch, first investigate whether the change is expected before regenerating.

## What Are Golden Sequences?

Golden sequences are the expected execution order of phases for each node in a topology. Each sequence is a list of `(node_name, phase)` tuples representing the exact order of operations.

Example sequence for a simple 4-node linear chain in FORWARD mode with BFS:
```python
('linear_chain', 'FORWARD', 'bfs'): [
    ('Node-1', 'INIT'),        # Input node initialization
    ('Node-2', 'RELAX'),       # Build relaxation
    ('Node-2', 'PROPAGATE'),   # Forward propagation
    ('Node-2', 'CACHE'),       # Cache symbolic bounds
    ('Node-2', 'COMPUTE'),     # Compute concrete bounds
    ('Node-2', 'CACHE'),       # Cache computed bounds
    ('Node-3', 'RELAX'),       # Next node...
    # ... continues
]
```

## Test Coverage

The golden sequences cover:

**17 Topologies:**
- Basic: linear_chain, y_shape, sequential_branches
- Skip Connections: single_skip_connection, multiple_skip_connections, nested_skip_connections
- Multi-Input: same_source_twice, diamond_pattern, wide_merge
- Complex: wide_broadcast, asymmetric_tree, multiple_merge_points
- Boundary: minimal_dag, long_chain (15 nodes), deep_branching
- Realistic: inception_like_module (10 nodes), densenet_like_connection (5 nodes)

**2 Propagation Modes:**
- FORWARD: Standard forward propagation
- BACKWARD: Backward substitution with node revisitation

**2 Sort Strategies:**
- BFS: Breadth-first topological sort
- DFS: Depth-first topological sort

## Why Exact Sequences?

Previous tests only verified node execution order (checking that nodes execute after predecessors). This was too weak and missed bugs in:
- Phase transition order
- Node revisitation in BACKWARD mode
- Complete execution flow

Golden sequences verify the **complete execution correctness** by comparing against deterministic reference outputs.
