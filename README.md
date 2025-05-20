# PropDAG: Bound Propagation of Directed Acyclic Computation Graphs for Neural Network Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> Bound propagation has dominated the field of neural network verification over the past years. 🏆 Many pioneering works have been proposed based on this approach, including:
> [ReLUVal (USENIX Security'18)](https://www.usenix.org/conference/usenixsecurity18/presentation/wang-shiqi)
> [DeepZ (NeurIPS'18)](https://proceedings.neurips.cc/paper_files/paper/2018/hash/f2f446980d8e971ef3da97af089481c3-Abstract.html)
> [Fast-Lin (ICML'18)](https://proceedings.mlr.press/v80/weng18a.html)
> [CROWN (NeurIPS'18)](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html)
> [DeepPoly (POPL'19)](https://dl.acm.org/doi/abs/10.1145/3290354)
> For more detailed discussion, see our blog: [Bound Propagation Approaches in Neural Network Verification](https://zhongkuima.github.io/blogs/bound_prop.html).

**propdag** is a flexible and research-oriented framework for developing **bound propagation** methods in neural network verification. 🧠🛡️

## 🎯 Why This Framework?

The goal of this repository is to provide a **lightweight yet powerful base** for researchers to rapidly prototype and test their own bound propagation algorithms.

Instead of spending time building the entire framework from scratch, you can now focus on your innovations.

### 🔍 Key Features

- ✅ **Lightweight**: No heavy dependencies like PyTorch or TensorFlow. Just clone the repo and start coding! 🚀
- ✅ **Full Customization**: Easily customize propagation rules for each layer type or define your own. Perfect for experimenting with new approaches. 🛠️
- ✅ **High-level Abstraction**: Focus on the algorithmic aspects of your work without getting bogged down in implementation details. 🧩
- ✅ **DAG Support**: Handles complex computation graphs with residual connections, skip connections, and branching structures. 🔄

Modern neural networks often involve complex structures beyond simple stacks of layers. **propdag** treats the network as a **Directed Acyclic Graph (DAG)** and propagates bounds through it in a clean, modular fashion.

### 🔑 Key Implementations

- **Topological ordering with BFS (Breadth-first Search) and DFS (Depth-first Search)**: Ensures proper processing order while minimizing cached intermediate results, important for networks with different dimensional layers. You can choose between BFS and DFS based on your needs.
- **Forward and Backward Propagation**: Supports both forward and backward bound propagation with clean, modular implementation for DAG structures.
- **Abstract Base Classes**: Provides template classes that can be easily extended for custom implementations.

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/ZhongkuiMa/propdag.git
cd propdag
```

**Requirements:**
- Python >= 3.10 (We use Python 3.12 in development)
- No additional third-party libraries needed

## 📦 Usage

This framework provides core logic and abstraction classes for building your own bound propagation algorithms, especially for DAG-structured networks.

### 📁 Folder Structure

- `propdag/template`: Abstract and template classes for defining custom propagation rules
- `propdag/toy`: Simple, illustrative example implementations

### 🧪 Getting Started

1. Study the provided examples in the `test` folder
2. Understand the template classes in `propdag/template`
3. Create your own implementation based on the templates
4. Run tests to verify your implementation

## 📊 Examples

The `test` folder contains examples demonstrating framework usage:

- **Basic examples**:
  - `test/example_forward.py`: Forward propagation through a DAG
  - `test/example_backward.py`: Backward propagation through a DAG
- Simple DAG structures to understand cross-layer propagation
- Starting points for developing your own methods 🔧

### Example Output: Backward Propagation

Below is a sample output from running `test/example_backward.py`:

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

## 🤝 Contributing

Contributions are welcome! Whether it's fixing bugs 🐞, adding features ✨, improving documentation 📚, or sharing ideas 💡—your input is appreciated!

**Contribution Guidelines:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

📌 **NOTE:** Direct pushes to the `main` branch are restricted.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
