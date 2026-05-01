# PropDAG: Bound Propagation for DAG-Structured Neural Networks

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ZhongkuiMa/propdag/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/propdag/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/ZhongkuiMa/propdag/branch/main/graph/badge.svg)](https://codecov.io/gh/ZhongkuiMa/propdag)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

Propagate bounds through DAG-structured neural networks for verification. Implements the graph execution engine behind algorithms like [CROWN](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html), [DeepPoly](https://dl.acm.org/doi/abs/10.1145/3290354), and [DeepZ](https://proceedings.neurips.cc/paper_files/paper/2018/hash/f2f446980d8e971ef3da97af089481c3-Abstract.html) -- pure Python, zero dependencies.

## Installation

```bash
git clone https://github.com/ZhongkuiMa/propdag.git
cd propdag
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+. No runtime dependencies.

## Quick Start

Build a DAG, connect nodes, and propagate bounds. This example uses the `toy2` module (reversed-graph semantics for backward bound propagation):

```python
from propdag.toy2 import Toy2Node, Toy2Cache, Toy2Argument, Toy2Model

# Shared cache and arguments across all nodes
cache = Toy2Cache()
args = Toy2Argument(verbose=False)

# Build graph in forward direction: Input -> Hidden -> Output
input_node = Toy2Node("Input", cache, args)
hidden_node = Toy2Node("Hidden", cache, args)
output_node = Toy2Node("Output", cache, args)

input_node.next_nodes = [hidden_node]
hidden_node.pre_nodes = [input_node]
hidden_node.next_nodes = [output_node]
output_node.pre_nodes = [hidden_node]

# T2Model reverses the graph internally, then propagates Output -> Input
model = Toy2Model([input_node, hidden_node, output_node])
model.run()

# Bounds propagated backward through the DAG
print(cache.bnds)  # {'Output': ..., 'Hidden': ..., 'Input': ...}
```

## Core Abstractions

PropDAG provides abstract base classes you extend to implement your algorithm:

| Class | Role |
|-------|------|
| `TNode` / `T2Node` | One layer or operation. Implements `forward()`, `backward()`, `build_rlx()`, `init_symbnd()`, `fwdprop_symbnd()`, `cal_and_update_cur_node_bnd()` |
| `TModel` / `T2Model` | Topologically sorts the DAG and executes nodes in order |
| `TCache` / `T2Cache` | Shared storage for bounds, relaxations, and symbolic expressions |
| `TArgument` / `T2Argument` | Frozen dataclass holding propagation configuration |

Nodes are connected via `pre_nodes` (predecessors) and `next_nodes` (successors). The model handles topological sorting (BFS or DFS) and cache lifecycle.

## Two Template Systems

| Aspect | `template` | `template2` (recommended) |
|--------|------------|---------------------------|
| Graph direction | Forward (Input -> Output) | Auto-reversed (Output -> Input internally) |
| Propagation | `PropMode.FORWARD` or `PropMode.BACKWARD` | Single-purpose backward propagation |
| Semantics | `backward()` traverses forward edges backward | `forward()` traverses reversed edges forward |
| Configuration | Requires `prop_mode` | No mode switching needed |

Use **template2** for backward bound propagation (CROWN, DeepPoly). Use **template** when you need forward propagation or both directions.

## Implementing a Custom Node

Extend `T2Node` (or `TNode`) and implement the abstract methods:

```python
from propdag import T2Node, T2Cache, T2Argument

class MyReLUNode(T2Node[T2Cache, T2Argument]):
    def forward(self):
        if not self.pre_nodes:  # Output node in user's view
            self.cache.bnds[self.name] = initial_output_bounds
            return
        self.build_rlx()
        self.propagate_bounds()
        self.intersect_and_update_bnd()

    def build_rlx(self):
        lb, ub = self.cache.fwd_bnds[self.name]
        # Compute ReLU relaxation: max(0, x)
        self.cache.rlxs[self.name] = compute_relu_relaxation(lb, ub)

    # ... implement remaining abstract methods
```

Then build your graph, wrap in `T2Model`, and call `run()`.

## BFS vs DFS Sorting

| Strategy | Best when |
|----------|-----------|
| `bfs` (default) | High-dimensional inputs -- avoids caching large early-layer tensors |
| `dfs` | Low-dimensional inputs -- reuses cached early layers across paths |

```python
model = T2Model(nodes, sort_strategy="dfs")  # or "bfs"
```

## Project Structure

```
propdag/
├── src/propdag/
│   ├── template/      # Abstract base classes (TNode, TModel, TCache, TArgument)
│   ├── template2/     # Reversed-graph ABCs (T2Node, T2Model, T2Cache, T2Argument)
│   ├── toy/           # Example implementation for template/
│   ├── toy2/          # Example implementation for template2/
│   ├── custom_types.py
│   └── utils.py       # PropMode enum (FORWARD/BACKWARD)
└── tests/             # 119 tests, 95% coverage
```

## Tests

```bash
pytest tests/ -v                              # all 119 tests
pytest tests/ -m benchmark -v                 # benchmark tests only
pytest tests/ --cov=src/propdag -v            # with coverage report
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License -- see [LICENSE](LICENSE)
