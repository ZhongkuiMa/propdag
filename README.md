# propdag: Bound Propagation of Directed Acyclic Computation Graphs for Neural Network Verification

**propdag** is a flexible and research-oriented framework for developing **bound propagation** methods in neural network verification. 🧠🛡️

As we all know, **bound propagation has dominated** the field of neural network verification over the past years. 🏆 Many pioneering works have been proposed based on this approach, including
[ReLUVal (USENIX Security'18)](https://www.usenix.org/conference/usenixsecurity18/presentation/wang-shiqi),
[DeepZ (NeurIPS'18)](https://proceedings.neurips.cc/paper_files/paper/2018/hash/f2f446980d8e971ef3da97af089481c3-Abstract.html),
[Fast-Lin (ICML'18)](https://proceedings.mlr.press/v80/weng18a.html),
[CROWN (NeurIPS'18)](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html),
[DeepPoly (POPL'19)](https://dl.acm.org/doi/abs/10.1145/3290354), and so on and so on. 🔥

See more discussion in my blog [Bound Propagation Approaches in Neural Network Verification](https://zhongkuima.github.io/blogs/bound_prop.html).

## 🎯 Why This Framework?

The goal of this repository is to provide a **lightweight yet powerful base** for researchers to rapidly prototype and test their own bound propagation algorithms.

Instead of spending time building the entire framework from scratch, you can now focus on:

### 🔍 Key Features

- ✅ **Lightweight**: No need to install heavy dependencies like PyTorch or TensorFlow. Open and easy to learn and use. Just clone the repo and start coding! 🚀
- ✅ **Full Customization**: You can easily customize the propagation rules for each layer type, and even define your own layer types. This is especially useful for experimenting with new ideas and approaches. 🛠️
- ✅ **High-level Abstraction**: The framework provides a high-level abstraction for defining propagation rules, making it easy to understand and extend. You can focus on the algorithmic aspects of your work without getting bogged down in complicated computational. 🧩

Modern neural networks often involve complex computation graphs such as **residual connections**, **skip connections**, and **branching structures**, beyond the simple stack of fully connected or convolutional layers.

**propdag** is designed to handle this generality by treating the network as a **Directed Acyclic Graph (DAG)** and propagating bounds through it in a clean, modular fashion. 🔄

Thanks to its **thoughtful and modular design** 🧩, you only need to worry about how each individual layer propagates bounds — without worrying about how layers are connected or how the overall graph executes. It abstracts away the complexity, so you can innovate with clarity and focus. 💡✨

### 🔑 Key Implementations

- **Breadth-First Search (BFS) for topological ordering**. We use BFS to traverse the DAG and ensure that all nodes are processed in the correct order. This is crucial for bound propagation, as it ensures that all dependencies are resolved before processing a node. Further, the breadth-first search reduce the cached intermediate results, which is important for large networks because the layer closer to the input may have more dimensions than the layer closer to the output.
- **Forward and Backward Propagation**. The framework supports both forward and backward propagation of bounds, which are the two main types of bound propagation used in neural network verification. The backward propagation of DAG is not easy to implement from scratch, but we have implemented it in a clean and modular way.

## 📦 Usage

This is a framework — not a plug-and-play verifier — intended for extending and building your own bound propagation algorithms. It provides core logic and abstraction classes, especially for propagating bounds through DAG-structured networks.

### 📁 Folder Structure

- `progdag/template`: Contains abstract and template classes for defining custom propagation rules.
- `propdag/toy`: A simple, illustrative example of how bound propagation can be implemented using the framework.

### ⚙️ Install Dependencies

You only need:

- Python **>= 3.10** (We are using Python 3.12.) 🐍 for supporting some typing features.

There are **no additional third-party library requirements**. Just clone and go!

### 🧪 Run Examples

Check out the `test` folder for examples demonstrating how to use the framework. These include:

- Basic toy examples to get started quickly 🎮
    - `test/example_forward.py`: A simple example of forward propagation of bounds through a DAG.
    - `test/example_backward.py`: A simple example of backward propagation of bounds through a DAG.
- Simple DAG structures to help understand how bound propagation works across layers
- A good starting point for developing and testing your own methods 🔧

### Example Outputs: Backward Propagation

The following shows the console output of running the `test/example_backward.py` example. It demonstrates how the backward propagation works in the framework.

```text
    The DAG is:

        Node-1
        /    \
     Node-2  Node-3
        \    /    \
        Node-4    Node-5
            \    /
            Node-6
    
Preparing ToyModel...
Running ToyModel...
=================================FORWARD Node-1=================================
Node-1: Calculate relaxation if this is non-linear node
--------------------------------BACKWARD Node-1---------------------------------
Node-1: Prepare symbolic bounds of Node-1
Node-1: Cache substitution
Node-1: Calculate scalar bounds of Node-1
Node-1: Cache scalar bounds
Node-1: Clear backforward cache of symbolic bounds
=================================FORWARD Node-2=================================
Node-2: Calculate relaxation if this is non-linear node
--------------------------------BACKWARD Node-2---------------------------------
Node-2: Prepare symbolic bounds of Node-2
Node-2: Cache substitution
Node-2: Calculate scalar bounds of Node-2
Node-2: Cache scalar bounds
--------------------------------BACKWARD Node-1---------------------------------
Node-1: Backsubstitute symbolic bounds of ['Node-2', 'Node-3']
Node-1: Cache substitution
Node-1: Calculate scalar bounds of Node-2
Node-1: Cache scalar bounds
Node-2: Clear backforward cache of symbolic bounds
Node-1: Clear backforward cache of symbolic bounds
=================================FORWARD Node-3=================================
Node-3: Calculate relaxation if this is non-linear node
--------------------------------BACKWARD Node-3---------------------------------
Node-3: Prepare symbolic bounds of Node-3
Node-3: Cache substitution
Node-3: Calculate scalar bounds of Node-3
Node-3: Cache scalar bounds
--------------------------------BACKWARD Node-1---------------------------------
Node-1: Backsubstitute symbolic bounds of ['Node-2', 'Node-3']
Node-1: Cache substitution
Node-1: Calculate scalar bounds of Node-3
Node-1: Cache scalar bounds
Node-3: Clear backforward cache of symbolic bounds
Node-1: Clear backforward cache of symbolic bounds
Node-1: Clear forward cache of bounds
=================================FORWARD Node-4=================================
Node-4: Calculate relaxation if this is non-linear node
--------------------------------BACKWARD Node-4---------------------------------
Node-4: Prepare symbolic bounds of Node-4
Node-4: Cache substitution
Node-4: Calculate scalar bounds of Node-4
Node-4: Cache scalar bounds
--------------------------------BACKWARD Node-2---------------------------------
Node-2: Backsubstitute symbolic bounds of ['Node-4']
Node-2: Cache substitution
Node-2: Calculate scalar bounds of Node-4
Node-2: Cache scalar bounds
--------------------------------BACKWARD Node-3---------------------------------
Node-3: Backsubstitute symbolic bounds of ['Node-4', 'Node-5']
Node-3: Cache substitution
Node-3: Calculate scalar bounds of Node-4
Node-3: Cache scalar bounds
Node-4: Clear backforward cache of symbolic bounds
--------------------------------BACKWARD Node-1---------------------------------
Node-1: Backsubstitute symbolic bounds of ['Node-2', 'Node-3']
Node-1: Cache substitution
Node-1: Calculate scalar bounds of Node-4
Node-1: Cache scalar bounds
Node-2: Clear backforward cache of symbolic bounds
Node-3: Clear backforward cache of symbolic bounds
Node-1: Clear backforward cache of symbolic bounds
Node-2: Clear forward cache of bounds
=================================FORWARD Node-5=================================
Node-5: Calculate relaxation if this is non-linear node
--------------------------------BACKWARD Node-5---------------------------------
Node-5: Prepare symbolic bounds of Node-5
Node-5: Cache substitution
Node-5: Calculate scalar bounds of Node-5
Node-5: Cache scalar bounds
--------------------------------BACKWARD Node-4---------------------------------
Node-4: Backsubstitute symbolic bounds of ['Node-5']
Node-4: Cache substitution
Node-4: Calculate scalar bounds of Node-5
Node-4: Cache scalar bounds
--------------------------------BACKWARD Node-3---------------------------------
Node-3: Backsubstitute symbolic bounds of ['Node-4', 'Node-5']
Node-3: Cache substitution
Node-3: Calculate scalar bounds of Node-5
Node-3: Cache scalar bounds
Node-5: Clear backforward cache of symbolic bounds
--------------------------------BACKWARD Node-2---------------------------------
Node-2: Backsubstitute symbolic bounds of ['Node-4']
Node-2: Cache substitution
Node-2: Calculate scalar bounds of Node-5
Node-2: Cache scalar bounds
Node-4: Clear backforward cache of symbolic bounds
--------------------------------BACKWARD Node-1---------------------------------
Node-1: Backsubstitute symbolic bounds of ['Node-2', 'Node-3']
Node-1: Cache substitution
Node-1: Calculate scalar bounds of Node-5
Node-1: Cache scalar bounds
Node-2: Clear backforward cache of symbolic bounds
Node-3: Clear backforward cache of symbolic bounds
Node-1: Clear backforward cache of symbolic bounds
Node-4: Clear forward cache of bounds
Node-3: Clear forward cache of bounds
=================================FORWARD Node-6=================================
Node-6: Calculate relaxation if this is non-linear node
--------------------------------BACKWARD Node-6---------------------------------
Node-6: Prepare symbolic bounds of Node-6
Node-6: Cache substitution
Node-6: Calculate scalar bounds of Node-6
Node-6: Cache scalar bounds
--------------------------------BACKWARD Node-5---------------------------------
Node-5: Backsubstitute symbolic bounds of ['Node-6']
Node-5: Cache substitution
Node-5: Calculate scalar bounds of Node-6
Node-5: Cache scalar bounds
Node-6: Clear backforward cache of symbolic bounds
--------------------------------BACKWARD Node-4---------------------------------
Node-4: Backsubstitute symbolic bounds of ['Node-5']
Node-4: Cache substitution
Node-4: Calculate scalar bounds of Node-6
Node-4: Cache scalar bounds
--------------------------------BACKWARD Node-3---------------------------------
Node-3: Backsubstitute symbolic bounds of ['Node-4', 'Node-5']
Node-3: Cache substitution
Node-3: Calculate scalar bounds of Node-6
Node-3: Cache scalar bounds
Node-5: Clear backforward cache of symbolic bounds
--------------------------------BACKWARD Node-2---------------------------------
Node-2: Backsubstitute symbolic bounds of ['Node-4']
Node-2: Cache substitution
Node-2: Calculate scalar bounds of Node-6
Node-2: Cache scalar bounds
Node-4: Clear backforward cache of symbolic bounds
--------------------------------BACKWARD Node-1---------------------------------
Node-1: Backsubstitute symbolic bounds of ['Node-2', 'Node-3']
Node-1: Cache substitution
Node-1: Calculate scalar bounds of Node-6
Node-1: Cache scalar bounds
Node-2: Clear backforward cache of symbolic bounds
Node-3: Clear backforward cache of symbolic bounds
Node-1: Clear backforward cache of symbolic bounds
Node-5: Clear forward cache of bounds
```

## 🤝 Contributing

We warmly welcome contributions from everyone! Whether it's fixing bugs 🐞, adding features ✨, improving documentation 📚, or just sharing ideas 💡—your input is appreciated!

📌 NOTE: Direct pushes to the `main` branch are restricted. Make sure to fork the repository and submit a Pull Request for any changes!
