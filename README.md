# PropDAG: Bound Propagation for DAG-Structured Neural Networks

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ZhongkuiMa/propdag/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/propdag/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/ZhongkuiMa/propdag/branch/main/graph/badge.svg)](https://codecov.io/gh/ZhongkuiMa/propdag)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Tests](https://img.shields.io/badge/tests-119%20passed-success)](https://github.com/ZhongkuiMa/propdag/actions/workflows/unit-tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](https://github.com/ZhongkuiMa/propdag)
[![Version](https://img.shields.io/badge/version-2026.1.1-blue.svg)](https://github.com/ZhongkuiMa/propdag/releases)

Lightweight framework for neural network verification via bound propagation on directed acyclic graphs (DAGs).

> **Background:** Bound propagation dominates neural network verification research: [ReLUVal (USENIX Security'18)](https://www.usenix.org/conference/usenixsecurity18/presentation/wang-shiqi), [DeepZ (NeurIPS'18)](https://proceedings.neurips.cc/paper_files/paper/2018/hash/f2f446980d8e971ef3da97af089481c3-Abstract.html), [Fast-Lin (ICML'18)](https://mlr.press/v80/weng18a.html), [CROWN (NeurIPS'18)](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html), [DeepPoly (POPL'19)](https://dl.acm.org/doi/abs/10.1145/3290354). See [our blog](https://zhongkuima.github.io/blogs/bound_prop.html) for details.

## Why PropDAG?

Rapid prototyping of bound propagation algorithms without building infrastructure from scratch.

## Features

- **Zero dependencies** - Pure Python, no PyTorch/TensorFlow required
- **Customizable** - Define propagation rules per layer type
- **DAG-native** - Handles residual/skip connections and branching
- **Flexible ordering** - BFS or DFS topological traversal
- **Bidirectional** - Forward and backward bound propagation
- **Dual templates** - Choose template (original) or template2 (reversed graph with cleaner semantics)
- **Extensible** - Abstract base classes for custom algorithms

## Quality Metrics

- **Test Suite**: 119 comprehensive tests with 100% pass rate
- **Code Coverage**: 95% statement coverage (377 statements, 20 uncovered)
- **Type Safety**: Fully typed with mypy type checking
- **Code Quality**: Enforced with ruff linter (comprehensive ruleset)
- **Lines of Code**: ~1,100 lines (source only, excluding tests)
- **Zero Dependencies**: Pure Python 3.11+, no runtime dependencies

### Test Coverage Details

**Test Breakdown (119 total tests):**

| Category | Tests | Coverage |
|----------|-------|----------|
| **DAG Topologies** | 56 tests | Basic structures, skip connections, branching, realistic patterns |
| **Sorting Algorithms** | 18 tests | BFS/DFS validity, topological ordering, edge cases |
| **Cache Management** | 10 tests | Progressive cleanup, reference counting, memory efficiency |
| **Error Handling** | 15 tests | Input/output constraints, cycle detection, error messages |
| **Coverage Gaps** | 10 tests | Verbose mode, sort strategies, model properties, cache clearing |
| **Benchmark Tests** | 6 tests | Long chains and memory leak detection (marked with `@pytest.mark.benchmark`) |

**Module Coverage:**

```
src/propdag/__init__.py                  100% (5/5 statements)
src/propdag/custom_types.py              100% (8/8 statements)
src/propdag/template/__init__.py         100% (6/6 statements)
src/propdag/template/_arguments.py       100% (7/7 statements)
src/propdag/template/_cache.py           100% (5/5 statements)
src/propdag/template/_model.py            97% (97/97 statements, 3 uncovered: lines 104, 114, 178)
src/propdag/template/_node.py             83% (63/63 statements, 11 uncovered: abstract method branches)
src/propdag/template/_sort.py             99% (67/67 statements, 1 uncovered: line 50)
src/propdag/toy/__init__.py              100% (6/6 statements)
src/propdag/toy/_arguments.py            100% (4/4 statements)
src/propdag/toy/_backward_node.py         98% (43/43 statements, 1 uncovered: line 97)
src/propdag/toy/_cache.py                100% (11/11 statements)
src/propdag/toy/_forward_node.py          90% (41/41 statements, 4 uncovered: lines 62, 86, 110, 125)
src/propdag/toy/_model.py                100% (7/7 statements)
src/propdag/utils.py                     100% (7/7 statements)
```

**TOTAL: 377 statements, 20 uncovered (5% uncovered) = 95% coverage**

### Running All Tests

To run the complete test suite including benchmark tests:

```bash
# Run all tests (119 total)
pytest tests/ -v

# Run only benchmark tests (6 tests)
pytest tests/ -m benchmark -v

# Run with coverage report
pytest tests/ --cov=src/propdag --cov-report=term-missing -v

# Run specific test categories
pytest tests/test_dag_topologies.py -v       # 56 tests: DAG structure
pytest tests/test_sorting_algorithms.py -v   # 18 tests: Topological sorting
pytest tests/test_cache_management.py -v     # 10 tests: Cache lifecycle
pytest tests/test_error_handling.py -v       # 15 tests: Error detection
pytest tests/test_coverage_gaps.py -v        # 10 tests: Additional coverage
```

## Test Results Summary

### Overall Results

**✅ ALL TESTS PASSED: 119/119 (100% success rate)**

```
============================= test session starts ==============================
Platform: linux, Python 3.11.6, pytest-9.0.2
Test execution time: 0.38 seconds
Coverage: 95% (377 statements analyzed, 20 uncovered)
============================= 119 passed in 0.38s ==============================
```

### Test Results by Category

#### 1. Cache Management (10 tests) ✅ ALL PASSED
**File:** `tests/test_cache_management.py`

| Test Class | Tests | Status | Purpose |
|-----------|-------|--------|---------|
| `TestCacheProgressiveCleanup` | 3 | ✅ PASS | Verify cache cleanup in linear chain, Y-shape, and skip connections |
| `TestCacheInputOutputPreservation` | 2 | ✅ PASS | Ensure input/output bounds preserved throughout execution |
| `TestCacheReferenceCountingLogic` | 2 | ✅ PASS | Validate reference counting in wide merge and broadcast patterns |
| `TestCacheMemoryEfficiency` | 2 | ✅ PASS | Verify no memory leaks in long chains and broadcast-merge patterns |
| `TestSymbolicBoundsCleanup` | 1 | ✅ PASS | Verify symbolic bounds cleanup in linear chain |

**Key Validations:**
- Cache state tracked after each propagation step
- Intermediate caches cleared when no longer needed
- Input and output bounds always preserved
- Memory efficiency through progressive cleanup

#### 2. Coverage Gaps (10 tests) ✅ ALL PASSED
**File:** `tests/test_coverage_gaps.py`

| Test Class | Tests | Status | Purpose |
|-----------|-------|--------|---------|
| `TestVerboseMode` | 3 | ✅ PASS | Verbose output during forward/backward passes |
| `TestInvalidSortStrategy` | 2 | ✅ PASS | Error handling for invalid sort strategies |
| `TestInputNodeHandling` | 1 | ✅ PASS | Input node forward pass handling |
| `TestModelProperties` | 3 | ✅ PASS | Property getters in TModel (sort_strategy, cache, arguments) |
| `TestCacheClearingVerbose` | 1 | ✅ PASS | Cache clearing with verbose logging |

**Key Validations:**
- Verbose mode prints expected messages
- Invalid sort strategies raise ValueError
- Both 'bfs' and 'dfs' strategies work correctly
- All model properties accessible and working

#### 3. DAG Topologies (56 tests) ✅ ALL PASSED
**File:** `tests/test_dag_topologies.py`

| Test Category | Count | Status | DAG Patterns Tested |
|--------------|-------|--------|-------------------|
| `TestBasicStructures` | 12 | ✅ PASS | Linear chains (4), Y-shape (4), Sequential branches (4) |
| `TestSkipConnections` | 12 | ✅ PASS | Single skip (4), Multiple skips (4), Nested skips (4) |
| `TestMultiInputCases` | 12 | ✅ PASS | Same source twice (4), Diamond (4), Wide merge (4) |
| `TestComplexBranching` | 12 | ✅ PASS | Wide broadcast (4), Asymmetric tree (4), Multiple merges (4) |
| `TestBoundaryAndEdgeCases` | 4 | ✅ PASS | Minimal DAG, Long chain (3 variants), Deep branching |
| `TestRealisticNetworkPatterns` | 4 | ✅ PASS | Inception-like module, DenseNet-like connections |

**Parametrization:** Each topology tested with:
- 2 sort strategies: BFS and DFS
- 2 propagation modes: Forward (1) and Backward (2)

**Key Validations:**
- All DAG topologies execute correctly
- BFS and DFS produce valid topological orderings
- Forward and backward propagation modes work
- Skip connections and residual patterns supported
- Realistic neural network patterns (Inception, DenseNet) validated

#### 4. Error Handling (15 tests) ✅ ALL PASSED
**File:** `tests/test_error_handling.py`

| Test Class | Tests | Status | Error Conditions Validated |
|-----------|-------|--------|---------------------------|
| `TestInputOutputConstraints` | 4 | ✅ PASS | Multiple inputs, multiple outputs, no input, no output |
| `TestCycleDetection` | 4 | ✅ PASS | Two-node cycle, three-node cycle, self-loop, cycle in larger DAG |
| `TestBFSAndDFSErrorDetection` | 2 | ✅ PASS | Both sort strategies detect multiple inputs and cycles |
| `TestValidDAGsAccepted` | 3 | ✅ PASS | Valid DAGs: linear chain, diamond, skip connection |
| `TestErrorMessageQuality` | 2 | ✅ PASS | Error messages are informative for multiple inputs and cycles |

**Key Validations:**
- Exactly one input node required
- Exactly one output node required
- No cycles allowed (acyclic property enforced)
- Both BFS and DFS detect violations
- Error messages provide clear information

#### 5. Topological Sorting (18 tests) ✅ ALL PASSED
**File:** `tests/test_sorting_algorithms.py`

| Test Class | Tests | Status | Sorting Validation |
|-----------|-------|--------|-------------------|
| `TestTopologicalSortValidity` | 11 | ✅ PASS | Linear chains (8 variants), Diamond, Skip connection, Wide merge |
| `TestSortingStrategies` | 2 | ✅ PASS | BFS vs DFS comparison, Model works with both |
| `TestEdgeCases` | 5 | ✅ PASS | Minimal DAG, Long chain order maintenance, Repeated predecessors |

**Chain Length Variants Tested:** 2, 5, 10, 15 node chains

**Key Validations:**
- Topological ordering is valid for all DAGs
- BFS produces breadth-first ordering
- DFS produces depth-first ordering
- Both strategies produce valid (though different) orderings
- Edge cases handled correctly

### Execution Details

**Test Run Information:**
- Date: 2026-01-03
- Platform: Linux 5.15.0-161-generic, Python 3.11.6
- Pytest Version: 9.0.2
- Execution Time: 0.38 seconds
- Timeout: 10 seconds per test

**Coverage Report (Full Details):**

```
Name                                 Stmts   Miss  Cover   Missing
------------------------------------------------------------------
src/propdag/__init__.py                  5      0   100%
src/propdag/custom_types.py              8      0   100%
src/propdag/template/__init__.py         6      0   100%
src/propdag/template/_arguments.py       7      0   100%
src/propdag/template/_cache.py           5      0   100%
src/propdag/template/_model.py          97      3    97%   104, 114, 178
src/propdag/template/_node.py           63     11    83%   66, 77, 88, 99, 110, 128, 145, 162, 173, 200, 218
src/propdag/template/_sort.py           67      1    99%   50
src/propdag/toy/__init__.py              6      0   100%
src/propdag/toy/_arguments.py            4      0   100%
src/propdag/toy/_backward_node.py       43      1    98%   97
src/propdag/toy/_cache.py               11      0   100%
src/propdag/toy/_forward_node.py        41      4    90%   62, 86, 110, 125
src/propdag/toy/_model.py                7      0   100%
src/propdag/utils.py                    7      0   100%
------------------------------------------------------------------
TOTAL                                  377     20    95%
```

**Coverage Analysis:**
- 9 modules with 100% coverage (100% perfect)
- 5 modules with ≥90% coverage (very good)
- 1 module with 83% coverage (TNode - abstract methods hard to fully cover)
- Total: **95% overall coverage** (20 uncovered statements out of 377)

**Uncovered Code Analysis:**
- **TNode._node.py (11 uncovered):** Abstract method branches for different propagation modes
- **ForwardToyNode._forward_node.py (4 uncovered):** NotImplementedError paths for backward operations
- **TModel._model.py (3 uncovered):** Edge cases in back-substitution logic (lines 104, 114, 178)
- **Sort._sort.py (1 uncovered):** Edge case in topological sort (line 50)
- **BackwardToyNode._backward_node.py (1 uncovered):** NotImplementedError path (line 97)

### Quality Metrics Summary

| Metric | Result | Status |
|--------|--------|--------|
| **Tests Passed** | 119/119 | ✅ 100% |
| **Code Coverage** | 377/377 statements | ✅ 95% |
| **Type Checking** | 0 errors | ✅ Clean |
| **Linting (Ruff)** | 0 violations | ✅ Clean |
| **Format Compliance** | All files | ✅ Compliant |
| **Execution Time** | 0.38 seconds | ✅ Fast |

## Core Concepts

### TNode - Network Layers/Operations

`TNode` is the fundamental abstraction representing a single layer or operation in your neural network DAG. Each node:

- Maintains references to predecessor (`pre_nodes`) and successor (`next_nodes`) nodes
- Implements computation via `forward()` and `backward()` methods
- Manages bounds via `build_rlx()`, `init_symbnd()`, `fwdprop_symbnd()`, `bwdprop_symbnd()`
- Calculates concrete bounds via `cal_and_update_cur_node_bnd()`

**Node Types:**
- **Input nodes**: No predecessors (graph entry points)
- **Hidden nodes**: Have both predecessors and successors
- **Output nodes**: No successors (graph exit points)

### TCache - Intermediate Results

`TCache` stores computation results during graph traversal:

- Shared across all nodes in the graph (memory efficient)
- Stores relaxations, symbolic bounds, and concrete bounds
- Cleared intelligently as computation progresses (reference counting)
- Customizable structure via dataclass inheritance

### TArgument - Configuration

`TArgument` controls node behavior:

- Immutable configuration (frozen dataclass)
- Shared across all nodes
- Primary field: `prop_mode` (FORWARD or BACKWARD)
- Extend with custom parameters (learning rates, epsilon values, etc.)

### TModel - Graph Execution

`TModel` orchestrates entire graph computation:

- Performs topological sorting (BFS or DFS)
- Executes nodes in correct order
- Manages cache lifecycle (clearing when no longer needed)
- Supports both forward propagation and backward substitution

### Propagation Modes

**Forward Propagation (`PropMode.FORWARD`)**
- Propagates bounds from input → output
- Traditional bound propagation
- Efficient for feedforward networks

**Backward Propagation (`PropMode.BACKWARD`)**
- Symbolic back-substitution for tighter bounds
- Substitutes intermediate symbolic expressions back to inputs
- More accurate but computationally intensive
- Used in algorithms like CROWN, DeepPoly

## Choosing Between Template and Template2

PropDAG provides two complementary template systems for different use cases:

| Aspect | Template (Original) | Template2 (New in v2026.1.1) |
|--------|---------------------|------------------------------|
| **Graph Structure** | User builds Input → Output | User builds Input → Output |
| **Execution Model** | Multiple modes (FORWARD/BACKWARD) | Single-purpose (backward propagation only) |
| **Semantics** | `backward_bound()` goes Output → Input | `forward()` on reversed graph |
| **Clarity** | Semantic mismatch possible | Clear, intuitive semantics |
| **Configuration** | `prop_mode` required | No mode switching |
| **Use Case** | Forward propagation, research prototyping | Backward bound propagation (recommended) |

**Recommendation for new projects:** Use **Template2** for backward bound propagation. It provides cleaner semantics and a more intuitive API.

## Template2: Reversed Graph Semantics

### The Problem with Template's Backward Propagation

In the original `template/` module:
- Users build graphs in the **natural direction**: Input → Hidden Layers → Output
- But `backward_bound()` propagates **backward** through this **forward graph**
- This creates semantic confusion: "going backward" through "forward edges"

### Template2's Solution: Automatic Graph Reversal

Template2 elegantly solves this with automatic graph reversal:

1. **User builds graph normally** (Input → Output)
2. **T2Model automatically reverses the graph** (swaps predecessor/successor relationships)
3. **Now propagation flows forward** through the reversed graph (Output → Input)
4. **Clear semantics**: `forward()` method on reversed graph achieves backward propagation naturally

This approach aligns with how algorithms like CROWN and DeepPoly conceptually work: propagating bounds backward from output to input.

### Key Advantages

- **Intuitive API**: No semantic mismatch between method name and behavior
- **Single purpose**: Focused on one task (backward bound propagation)
- **No mode switching**: No need for `prop_mode` configuration
- **Framework agnostic**: Can work with NumPy, PyTorch, or other frameworks
- **Cleaner cache structure**: Simplified field naming (bnds, rlxs, fwd_bnds, symbnds)

### Example Concept

```
# Graph as built by user (Input → Output)
Input → ReLU → Linear → Output

# T2Model automatically reverses to (Output → Input)
Output → Linear → ReLU → Input

# forward() propagation on reversed graph:
# Bounds flow Output → Input (backward semantics achieved!)
```

## Installation

PropDAG is a standalone framework **not available on PyPI**. Install from source:

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/ZhongkuiMa/propdag.git
cd propdag

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check version
python -c "import propdag; print(propdag.__version__)"
# Expected: 2026.1.0

# Run test suite
pytest tests/ -v
# Expected: 119 passed

# Run linting
ruff check src/propdag tests
```

### Installation Options

**Development mode (recommended for contributors):**
```bash
pip install -e ".[dev]"  # Includes pytest, ruff, mypy
```

**Minimal install (runtime only):**
```bash
pip install -e .  # No additional dependencies (zero-dependency framework)
```

### Requirements

- **Python**: 3.11 or higher
- **Runtime dependencies**: None (pure Python, zero dependencies)
- **Development dependencies**: pytest, pytest-cov, pytest-timeout, ruff, mypy

## Quick Start

### Step 1: Study the Template Classes

Review the abstract base classes in `propdag/template/`:

1. **`_node.py`** - Understand the TNode interface and lifecycle
2. **`_model.py`** - See how TModel orchestrates execution
3. **`_cache.py`** and **`_arguments.py`** - Learn the data structures

### Step 2: Review Toy Implementations

Study concrete implementations in `propdag/toy/`:

- **`_forward_node.py`** - Forward-only propagation pattern
- **`_backward_node.py`** - Backward substitution pattern
- **`_cache.py`** - Example cache structure
- **`_arguments.py`** - Example arguments
- **`_model.py`** - Simple model wrapper

### Step 3: Implement Your Custom Algorithm

1. Define your cache structure (extend `TCache`)
2. Define your arguments (extend `TArgument`)
3. Implement your nodes (extend `TNode`)
4. Create a model (extend `TModel` or use directly)
5. Build your graph and run!

See "Usage Guide" below for detailed examples.

### Quick Start with Template2 (Recommended for Backward Propagation)

If you're implementing backward bound propagation, Template2 provides a cleaner approach:

**Step 1: Study Template2 Classes**

Review the template2 module in `propdag/template2/`:

1. **`_node.py`** - T2Node interface
2. **`_model.py`** - T2Model with automatic graph reversal
3. **`_cache.py`** - Simplified cache structure
4. **`_arguments.py`** - Configuration (no prop_mode)

**Step 2: Review Toy2 Implementations**

Study concrete examples in `propdag/toy2/`:

- **`_node.py`** - Example T2Node implementation
- **`_cache.py`** - Example cache with bnds, rlxs, fwd_bnds, symbnds
- **`_arguments.py`** - Example arguments
- **`_model.py`** - Example model wrapper

**Step 3: Simple Template2 Example**

```python
from propdag.toy2 import Toy2Node, Toy2Cache, Toy2Argument, Toy2Model

# Create cache and arguments
cache = Toy2Cache()
args = Toy2Argument()

# Build graph normally (Input → Output)
input_node = Toy2Node("input", cache, args)
hidden_node = Toy2Node("hidden", cache, args)
output_node = Toy2Node("output", cache, args)

# Connect nodes in forward direction
input_node.next_nodes = [hidden_node]
hidden_node.pre_nodes = [input_node]
hidden_node.next_nodes = [output_node]
output_node.pre_nodes = [hidden_node]

# T2Model automatically reverses graph and executes
model = Toy2Model([input_node, hidden_node, output_node])
model.run()

# Bounds are propagated from output to input (backward semantics)
print(f"Input bounds: {cache.bnds.get('input')}")
print(f"Output bounds: {cache.bnds.get('output')}")
```

**Key Differences from Template:**
- Build graph the same way (Input → Output)
- T2Model automatically reverses internally
- No `prop_mode` configuration needed
- `forward()` method achieves backward propagation
- Cleaner semantics overall

See "Usage Guide" below for detailed Template2 examples.

## API Overview

### Template Classes (`propdag.template`)

All template classes are abstract base classes (ABCs) that you extend:

**`TNode[CacheType, ArgumentType]`**
- Abstract methods to implement:
  - `forward()` - Forward computation
  - `backward()` - Backward computation
  - `build_rlx()` - Build relaxations for non-linear ops
  - `init_symbnd()` - Initialize symbolic bounds
  - `fwdprop_symbnd()` - Forward propagate symbolic bounds
  - `bwdprop_symbnd()` - Backward substitute symbolic bounds
  - `cal_and_update_cur_node_bnd()` - Calculate concrete bounds

**`TModel[CacheType, ArgumentType, NodeType]`**
- Methods:
  - `run()` - Execute entire graph
  - `backsub(node)` - Perform back-substitution from a node

**`TCache`**
- Extend with your own fields as a dataclass
- Use `slots=True` for memory efficiency

**`TArgument`**
- Extend with your own configuration parameters
- Must be frozen (`frozen=True`)

### Template2 Classes (`propdag.template2`)

Template2 provides reversed graph semantics for intuitive backward bound propagation:

**`T2Node[CacheType, ArgumentType]`**
- Abstract node for reversed graph execution
- Implement `forward()` for backward propagation semantics
- Same interface as TNode but works on automatically-reversed graph
- Abstract methods:
  - `forward()` - Forward computation on reversed graph (achieves backward semantics)
  - `backward()` - Not typically used in backward propagation mode
  - `build_rlx()`, `init_symbnd()`, `fwdprop_symbnd()`, `bwdprop_symbnd()`, `cal_and_update_cur_node_bnd()`

**`T2Model[CacheType, ArgumentType, NodeType]`**
- Model with automatic graph reversal
- Methods:
  - `run()` - Automatically reverses graph, then executes
  - No back-substitution mode needed (semantic is already backward)

**`T2Cache`**
- Simplified cache for reversed graph execution
- Typical fields: `bnds`, `rlxs`, `fwd_bnds`, `symbnds`
- Extend with your own fields as a dataclass

**`T2Argument`**
- Configuration for template2 (no `prop_mode` needed)
- Single-purpose: backward bound propagation
- Extend with custom parameters

**`reverse_dag(nodes)`**
- Helper function to manually reverse graph edges
- Used internally by T2Model but available for custom use
- Swaps `pre_nodes` ↔ `next_nodes` for all nodes

**Template2 Sorting Functions**
- `topo_sort_forward_bfs_t2(nodes, verbose=False)` - BFS on reversed graph
- `topo_sort_forward_dfs_t2(nodes, verbose=False)` - DFS on reversed graph

### Topological Sorting Functions

**`topo_sort_forward_bfs(nodes, verbose=False)`**
- Breadth-first traversal
- Use when: Input has high dimensionality (avoid caching early layers)

**`topo_sort_forward_dfs(nodes, verbose=False)`**
- Depth-first traversal
- Use when: Input has low dimensionality (cache early layers for reuse)

**`topo_sort_backward(nodes, verbose=False)`**
- Generates backward order for each node
- Returns `Dict[TNode, List[TNode]]`

## Usage Guide

### Creating a Simple Custom Node

Here's a minimal example implementing a ReLU node with interval bounds:

```python
from dataclasses import dataclass
from propdag import TNode, TCache, TArgument, PropMode

@dataclass
class IntervalCache(TCache):
    """Cache storing interval bounds [lower, upper]."""
    bounds: dict[str, tuple[float, float]] = None

    def __post_init__(self):
        if self.bounds is None:
            self.bounds = {}

class ReLUNode(TNode[IntervalCache, TArgument]):
    """ReLU node with interval arithmetic."""

    def forward(self):
        """Forward pass: propagate concrete bounds."""
        # Get input bounds from predecessor
        pre_node = self.pre_nodes[0]
        lower, upper = self.cache.bounds[pre_node.name]

        # Apply ReLU: max(0, x)
        relu_lower = max(0.0, lower)
        relu_upper = max(0.0, upper)

        # Store in cache
        self.cache.bounds[self.name] = (relu_lower, relu_upper)

    def backward(self):
        """Backward pass - not needed for forward-only."""
        pass

    # ... implement other abstract methods as needed
```

### Building a Graph

```python
# Create shared cache and arguments
cache = IntervalCache()
args = TArgument(prop_mode=PropMode.FORWARD)

# Create nodes
input_node = InputNode("input", cache, args)
relu1 = ReLUNode("relu1", cache, args)
relu2 = ReLUNode("relu2", cache, args)
output = LinearNode("output", cache, args)

# Connect nodes
input_node.next_nodes = [relu1]
relu1.pre_nodes = [input_node]
relu1.next_nodes = [relu2]
relu2.pre_nodes = [relu1]
relu2.next_nodes = [output]
output.pre_nodes = [relu2]

# Create model and run
from propdag import TModel
model = TModel([input_node, relu1, relu2, output])
model.run()

# Access results
final_bounds = cache.bounds["output"]
print(f"Output bounds: {final_bounds}")
```

### Choosing BFS vs DFS

**Use BFS** when:
- Input has many dimensions (e.g., image inputs)
- You want to avoid caching large early-layer tensors
- Memory is limited

**Use DFS** when:
- Input has few dimensions (e.g., scalar or small vector)
- Early layers are reused by many paths (benefit from caching)
- You have sufficient memory

```python
# Specify sort method in TModel
model = TModel(nodes, sort_method="bfs")  # or "dfs"
```

## Examples and Tests

Run the comprehensive test suite to see PropDAG in action. See [Quality Metrics](#quality-metrics) above for detailed test coverage information (119 tests, 95% coverage).

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories (see Quality Metrics for breakdown)
pytest tests/test_dag_topologies.py -v       # 56 DAG structure tests
pytest tests/test_sorting_algorithms.py -v   # 18 Topological sorting tests
pytest tests/test_cache_management.py -v     # 10 Cache lifecycle tests
pytest tests/test_error_handling.py -v       # 15 Error detection tests
pytest tests/test_coverage_gaps.py -v        # 10 Additional coverage tests

# Run only benchmark tests (marked with @pytest.mark.benchmark)
pytest tests/ -m benchmark -v
```

The tests demonstrate:
- **DAG Topologies**: Linear chains, skip connections, branching, realistic patterns (Inception, DenseNet)
- **Propagation Modes**: Forward and backward propagation with both BFS and DFS traversal
- **Topological Sorting**: BFS vs DFS strategies with validity verification
- **Cache Management**: Progressive cleanup, reference counting, memory efficiency
- **Error Detection**: Cycle detection, input/output constraints, informative error messages
- **Benchmark Tests**: Performance under deep chains and memory efficiency verification

**Sample DAG structure tested:**
```text
    The DAG is:

        Node-1
        /    \
     Node-2  Node-3
        \    /    \
        Node-4    Node-5
            \    /
            Node-6
```

See `tests/conftest.py` for fixture examples showing how to build and run propagation on custom DAGs.

### Test Metrics Summary

**119 tests pass successfully:**
- All 56 DAG topology tests pass with 100% success rate
- All 18 sorting algorithm tests verify topological order validity
- All 10 cache management tests verify no memory leaks
- All 15 error handling tests verify constraint validation
- All 10 coverage gap tests verify edge cases
- All 6 benchmark tests verify performance under stress

**Code Quality:**
- **Coverage**: 95% statement coverage (377 statements)
- **Type Checking**: 0 errors with mypy
- **Linting**: All ruff checks pass (0 violations)
- **Formatting**: All files properly formatted with ruff

## Project Structure

```
propdag/
├── propdag/
│   ├── template/           # Abstract base classes
│   │   ├── __init__.py
│   │   ├── _node.py        # TNode - layer/operation abstraction
│   │   ├── _model.py       # TModel - graph execution engine
│   │   ├── _cache.py       # TCache - intermediate results storage
│   │   ├── _arguments.py   # TArgument - configuration
│   │   └── _sort.py        # Topological sorting (BFS/DFS)
│   ├── toy/                # Example implementations for template/
│   │   ├── __init__.py
│   │   ├── _forward_node.py   # Forward propagation example
│   │   ├── _backward_node.py  # Backward propagation example
│   │   ├── _cache.py          # Example cache structure
│   │   ├── _arguments.py      # Example arguments
│   │   └── _model.py          # Example model wrapper
│   ├── template2/          # Abstract base classes (reversed graph)
│   │   ├── __init__.py
│   │   ├── _node.py        # T2Node - reversed graph nodes
│   │   ├── _model.py       # T2Model - with automatic graph reversal
│   │   ├── _cache.py       # T2Cache - simplified cache structure
│   │   ├── _arguments.py   # T2Argument - configuration
│   │   └── _sort.py        # Topological sorting for template2
│   ├── toy2/               # Example implementations for template2/
│   │   ├── __init__.py
│   │   ├── _node.py        # Example T2Node for reversed graph
│   │   ├── _cache.py       # Example T2Cache structure
│   │   ├── _arguments.py   # Example T2Argument
│   │   └── _model.py       # Example T2Model wrapper
│   ├── __init__.py         # Module exports
│   ├── custom_types.py     # Type definitions (Generic types)
│   └── utils.py            # PropMode enum (FORWARD/BACKWARD)
├── tests/                  # Working examples
│   ├── example_forward.py    # Forward propagation demo
│   └── example_backward.py   # Backward propagation demo
├── pyproject.toml          # Package configuration
├── LICENSE                 # MIT License
└── README.md               # This file
```

## Contributing Guidelines

We welcome contributions! Please follow these guidelines to ensure smooth collaboration.

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/propdag.git
cd propdag

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify setup
pytest tests/ -v
ruff check src/propdag tests
```

### Branch Naming Conventions

Use descriptive branch names with prefixes:
- `feature/` - New features (e.g., `feature/add-multivariate-bounds`)
- `fix/` - Bug fixes (e.g., `fix/cache-clearing-issue`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-sorting`)
- `docs/` - Documentation updates (e.g., `docs/improve-api-examples`)
- `test/` - Test improvements (e.g., `test/add-edge-cases`)

### Commit Message Format

Write clear, concise commit messages:

```
<type>: <short summary in present tense>

<optional detailed description>

<optional footer with issue references>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring (no behavior change)
- `test:` - Add or update tests
- `docs:` - Documentation changes
- `style:` - Code style/formatting (ruff, whitespace)
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks (dependencies, config)

**Examples:**
```
feat: Add support for multi-input DAG nodes

Implements handling for nodes with multiple input branches,
enabling more complex graph topologies like inception modules.

Fixes #42
```

```
fix: Correct cache clearing in backward propagation

The backward cache wasn't being cleared for leaf nodes,
causing memory accumulation in deep networks.
```

### Pull Request Guidelines

1. **Before creating a PR:**
   ```bash
   # Run all tests
   pytest tests/ -v

   # Run linting
   ruff check src/propdag tests
   ruff format src/propdag tests

   # Run type checking
   python -m mypy
   ```

2. **Create PR with:**
   - **Clear title**: Follow commit message format
   - **Description**: Explain what changes and why
   - **Tests**: Add tests for new features/fixes
   - **Documentation**: Update README/docstrings if needed
   - **Changelog**: Note breaking changes if any

3. **PR template:**
   ```markdown
   ## Summary
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update

   ## Testing
   - [ ] All tests pass (`pytest tests/ -v`)
   - [ ] Linting passes (`ruff check src/propdag tests`)
   - [ ] Type checking passes (`mypy`)
   - [ ] Added tests for new features/fixes

   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Self-reviewed my own code
   - [ ] Commented complex logic
   - [ ] Updated documentation
   - [ ] No new warnings introduced
   ```

4. **Review process:**
   - Maintainers will review within 48-72 hours
   - Address feedback by pushing to your PR branch
   - Once approved, maintainers will merge

### Push Workflow

```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes and commit
git add <files>
git commit -m "feat: Add my new feature"

# Run pre-push checks (recommended)
pytest tests/ -v
ruff check src/propdag tests
python -m mypy

# Push to your fork
git push origin feature/my-new-feature

# Create PR on GitHub
```

### Running CI Locally

Before pushing, run the same checks as GitHub Actions:

```bash
# Full CI simulation
pytest tests/ --cov=src/propdag --cov-report=term-missing --cov-report=xml -v
ruff check src/propdag tests
ruff format --check src/propdag tests
python -m mypy
```

### Code Style

PropDAG follows strict code quality standards:

- **Formatter**: `ruff format` (100 char line length)
- **Linter**: `ruff check` (comprehensive ruleset)
- **Type checker**: `mypy`
- **Docstrings**: PEP 257 style with type hints

### Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_dag_topologies.py -v

# Run with coverage
pytest tests/ --cov=src/propdag --cov-report=term-missing -v
```

### Documentation

- All public classes/functions must have docstrings
- Use reStructuredText format (Sphinx-compatible)
- Include type hints for all parameters and return values

### Pre-commit Hooks (Optional)

Install pre-commit hooks to automatically check code before commits:

```bash
pip install pre-commit
pre-commit install

# Hooks will run automatically on git commit
# Or run manually:
pre-commit run --all-files
```

### Release Process

Releases are managed by maintainers:
1. Version bump in `pyproject.toml`
2. Update `__version__` in `src/propdag/__init__.py`
3. Create annotated git tag: `git tag -a v2026.1.0 -m "Release v2026.1.0"`
4. Push tag: `git push origin v2026.1.0`
5. GitHub Actions will run tests and create release

### Getting Help

- **Questions**: Open a GitHub Discussion
- **Bug reports**: Open an Issue with reproducible example
- **Feature requests**: Open an Issue with use case description

Thank you for contributing to PropDAG!

## License

MIT License - see [LICENSE](LICENSE)
