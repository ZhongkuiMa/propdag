# Golden Sequences for T2 Tests

This directory contains golden sequence files for verifying exact execution order in T2 (Toy2 model) tests with reversed graph execution.

## Files

- **`golden_sequences_t2.py`**: Contains 34 golden sequences for all T2 topology test cases
  - 17 topologies × 2 sort strategies (BFS/DFS)
  - Generated automatically - do not edit manually

- **`capture_golden_sequences_t2.py`**: Script to regenerate golden sequences
  - Use this when you need to update golden sequences after code changes

## Regenerating Golden Sequences

When you make changes to the T2 template logic or reversed graph execution, regenerate the golden sequences:

```bash
cd tests/test_t2
python capture_golden_sequences_t2.py > golden_sequences_t2.py
```

**Important:** Only regenerate golden sequences when you've intentionally changed the execution flow. If a test fails due to sequence mismatch, first investigate whether the change is expected before regenerating.

## What Are Golden Sequences?

Golden sequences are the expected execution order of phases for each node in a topology after graph reversal. Each sequence is a list of `(node_name, phase)` tuples representing the exact order of operations.

Example sequence for a simple 4-node linear chain with BFS (reversed execution):
```python
('linear_chain', 'bfs'): [
    ('Node-4', 'INIT'),        # Output node becomes input after reversal
    ('Node-3', 'RELAX'),       # Build relaxation
    ('Node-3', 'CACHE'),       # Cache operation
    ('Node-3', 'PROPAGATE'),   # Backward propagation
    ('Node-3', 'CACHE'),       # Cache symbolic bounds
    ('Node-3', 'COMPUTE'),     # Compute bounds
    ('Node-3', 'CACHE'),       # Cache computed bounds
    ('Node-2', 'RELAX'),       # Previous node...
    # ... continues backward
]
```

## T2 Template Differences

T2 (Template2) differs from T1 in several key ways:

1. **Graph Reversal**: The computation graph is reversed before execution
   - Original output node becomes the input (INIT phase)
   - Original input node becomes the output

2. **Execution Direction**: Processes backward through the reversed graph
   - In original graph: Node-1 → Node-2 → Node-3 → Node-4
   - In reversed graph: Node-4 → Node-3 → Node-2 → Node-1

3. **Bound Intersection**: Uses forward bounds (fwd_bnds) for intersection
   - Original input maintains fwd_bnds for backward bound computation

## Test Coverage

The golden sequences cover the same 17 topologies as T1, but with reversed execution:

**17 Topologies:**
- Basic: linear_chain, y_shape, sequential_branches
- Skip Connections: single_skip_connection, multiple_skip_connections, nested_skip_connections
- Multi-Input: same_source_twice, diamond_pattern, wide_merge
- Complex: wide_broadcast, asymmetric_tree, multiple_merge_points
- Boundary: minimal_dag, long_chain (15 nodes), deep_branching
- Realistic: inception_like_module (10 nodes), densenet_like_connection (5 nodes)

**2 Sort Strategies:**
- BFS: Breadth-first topological sort (on reversed graph)
- DFS: Depth-first topological sort (on reversed graph)

## Why Exact Sequences?

Golden sequences verify the **complete execution correctness** of the reversed graph execution model by comparing against deterministic reference outputs. This ensures that:
- Graph reversal is correct
- Topological order is maintained in reversed graph
- All phases execute in the correct order
- Bounds propagate correctly through reversed edges
