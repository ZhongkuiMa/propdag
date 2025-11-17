# PropDAG: Bound Propagation for DAG-Structured Neural Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Lightweight framework for neural network verification via bound propagation on directed acyclic graphs (DAGs).

> **Background:** Bound propagation dominates neural network verification research: [ReLUVal (USENIX Security'18)](https://www.usenix.org/conference/usenixsecurity18/presentation/wang-shiqi), [DeepZ (NeurIPS'18)](https://proceedings.neurips.cc/paper_files/paper/2018/hash/f2f446980d8e971ef3da97af089481c3-Abstract.html), [Fast-Lin (ICML'18)](https://proceedings.mlr.press/v80/weng18a.html), [CROWN (NeurIPS'18)](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html), [DeepPoly (POPL'19)](https://dl.acm.org/doi/abs/10.1145/3290354). See [our blog](https://zhongkuima.github.io/blogs/bound_prop.html) for details.

## Why PropDAG?

Rapid prototyping of bound propagation algorithms without building infrastructure from scratch.

## Features

- **Zero dependencies** - Pure Python, no PyTorch/TensorFlow required
- **Customizable** - Define propagation rules per layer type
- **DAG-native** - Handles residual/skip connections and branching
- **Flexible ordering** - BFS or DFS topological traversal
- **Bidirectional** - Forward and backward bound propagation
- **Extensible** - Abstract base classes for custom algorithms

## Installation

```bash
git clone https://github.com/ZhongkuiMa/propdag.git
cd propdag
```

**Requirements:** Python 3.10+ (no additional libraries)

## Structure

```
propdag/
├── template/     # Abstract base classes for custom implementations
├── toy/          # Example implementations with verbose logging
└── utils.py      # Propagation mode enum (FORWARD/BACKWARD)
```

## Quick Start

1. Review examples in `test/` folder
2. Study abstract templates in `propdag/template/`
3. Implement custom propagation logic
4. Test your implementation

## Examples

Run examples to see propagation in action:

```bash
python test/example_forward.py   # Forward propagation
python test/example_backward.py  # Backward propagation with substitution
```

**Sample DAG structure:**
```text
    The DAG is:

        Node-1
        /    \
     Node-2  Node-3
        \    /    \
        Node-4    Node-5
            \    /
            Node-6
    
Running ToyModel...
Forward pass Node-1
Node-1: Skip input node
Forward pass Node-2
Node-2: Calculate relaxation if this is a non-linear node
Node-2: Build symbolic bounds if this is a linear node
	Back-substitute Node-2
Node-2: Prepare symbolic bounds of Node-2
Node-2: Cache substitution
Node-2: Calculate scalar bounds of Node-2
Node-2: Cache scalar bounds
	Back-substitute Node-1
Node-1: Backsubstitute symbolic bounds of Node-2
Node-1: Cache substitution
Node-1: Calculate scalar bounds of Node-2
Node-1: Cache scalar bounds
Node-2: Clear backforward cache of symbolic bounds
...
```

Output shows step-by-step bound propagation, relaxation building, and cache management.

## License

MIT License - see [LICENSE](LICENSE)
