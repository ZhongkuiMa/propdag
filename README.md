# PropDAG

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

```python
from propdag.toy2 import Toy2Node, Toy2Cache, Toy2Argument, Toy2Model

cache = Toy2Cache()
args = Toy2Argument(verbose=False)

# Build graph: Input -> Hidden -> Output
input_node = Toy2Node("Input", cache, args)
hidden_node = Toy2Node("Hidden", cache, args)
output_node = Toy2Node("Output", cache, args)

input_node.next_nodes = [hidden_node]
hidden_node.pre_nodes = [input_node]
hidden_node.next_nodes = [output_node]
output_node.pre_nodes = [hidden_node]

model = Toy2Model([input_node, hidden_node, output_node])
model.run()

print(list(cache.bnds.keys()))  # ['Output', 'Hidden', 'Input']
```

## Usage Guide

### Template Systems

Two abstract base families serve different propagation directions:

| | `template` | `template2` |
|--|------------|-------------|
| Graph direction | Forward (Input -> Output) | Auto-reversed (Output -> Input internally) |
| Propagation modes | `PropMode.FORWARD` or `PropMode.BACKWARD` | Backward only |
| Use when | Forward propagation or both directions needed | CROWN, DeepPoly-style backward bound propagation |

### Core Abstractions

| Class | Role |
|-------|------|
| `TNode` / `T2Node` | One layer or operation; override `forward()`, `build_rlx()`, `cal_and_update_cur_node_bnd()` |
| `TModel` / `T2Model` | Topologically sorts the DAG and runs nodes in order |
| `TCache` / `T2Cache` | Shared storage for bounds (`bnds`), relaxations (`rlxs`), symbolic bounds (`symbnds`) |
| `TArgument` / `T2Argument` | Frozen dataclass holding propagation configuration |

Nodes connect via `pre_nodes` and `next_nodes`. The model handles topological sorting and cache lifecycle.

### Implementing a Node

Extend `T2Node` and override the methods that raise `RuntimeError`:

```python
from propdag.template2 import T2Node, T2Cache, T2Argument

class MyNode(T2Node[T2Cache, T2Argument]):
    def forward(self):
        if not self.pre_nodes:
            self.cache.bnds[self.name] = initial_bounds
            return
        self.build_rlx()
        self.cal_and_update_cur_node_bnd()

    def build_rlx(self): ...
    def cal_and_update_cur_node_bnd(self): ...
    def clear_fwd_cache(self): ...
    def clear_bwd_cache(self): ...
```

### Sorting Strategy

| Strategy | Best when |
|----------|-----------|
| `"bfs"` (default) | High-dimensional inputs -- avoids caching large early-layer tensors |
| `"dfs"` | Low-dimensional inputs -- reuses cached early layers across paths |

```python
model = T2Model(nodes, sort_strategy="dfs")
```

## Project Structure

```
src/propdag/
    template/       Abstract base classes (TNode, TModel, TCache, TArgument)
    template2/      Reversed-graph ABCs (T2Node, T2Model, T2Cache, T2Argument)
    toy/            Example implementation for template/
    toy2/           Example implementation for template2/
    custom_types.py Type aliases (CacheType, ArgumentType, NodeType)
    utils.py        PropMode enum (FORWARD/BACKWARD)
```

## Tests

```bash
pytest tests/ -v                           # all tests
pytest tests/ -m benchmark -v              # benchmark tests only
pytest tests/ --cov=src/propdag -v         # with coverage report
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License -- see [LICENSE](LICENSE).
