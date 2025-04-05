# propdag: Bound Propagation of Directed Acyclic Computation Graphs for Neural Network Verification

**propdag** is a flexible and research-oriented framework for developing **bound propagation** methods in neural network verification. 🧠🛡️

As we all know, **bound propagation has dominated** the field of neural network verification over the past years. 🏆 Many pioneering works have been proposed based on this approach, including
[ReLUVal (USENIX Security'18)](https://www.usenix.org/conference/usenixsecurity18/presentation/wang-shiqi)
[DeepZ (NeurIPS'18)](https://proceedings.neurips.cc/paper_files/paper/2018/hash/f2f446980d8e971ef3da97af089481c3-Abstract.html)[Fast-Lin (ICML'18)](https://proceedings.mlr.press/v80/weng18a.html)[CROWN (NeurIPS'18)](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html)[DeepPoly (POPL'19)](https://dl.acm.org/doi/abs/10.1145/3290354), and so on and so on. 🔥

---

## 🎯 Why This Framework?

The goal of this repository is to provide a **lightweight yet powerful base** for researchers to rapidly prototype and test their own bound propagation algorithms.

Instead of spending time building the entire framework from scratch, you can now focus on:

✅ Implementing **layer-wise propagation rules**  
✅ Customizing your own methods easily  
✅ Abstracting away from the complex computation graph details

Modern neural networks often involve complex computation graphs such as **residual connections**, **skip connections**, and **branching structures**, beyond the simple stack of fully connected or convolutional layers.

**propdag** is designed to handle this generality by treating the network as a **Directed Acyclic Graph (DAG)** and propagating bounds through it in a clean, modular fashion. 🔄

Thanks to its **thoughtful and modular design** 🧩, you only need to worry about how each individual layer propagates bounds — without worrying about how layers are connected or how the overall graph executes. It abstracts away the complexity, so you can innovate with clarity and focus. 💡✨

---

## 📦 Usage

This is a framework — not a plug-and-play verifier — intended for extending and building your own bound propagation algorithms. It provides core logic and abstraction classes, especially for propagating bounds through DAG-structured networks.

### 📁 Folder Structure

- `progdag/template`: Contains abstract and template classes for defining custom propagation rules.
- `propdag/toy`: A simple, illustrative example of how bound propagation can be implemented using the framework.

---

## ⚙️ Install Dependencies

You only need:

- Python **>= 3.10** 🐍

There are **no additional third-party library requirements**. Just clone and go!

---

## 🧪 Run Examples

Check out the `test` folder for examples demonstrating how to use the framework. These include:

- Basic toy examples to get started quickly 🎮
- Simple DAG structures to help understand how bound propagation works across layers
- A good starting point for developing and testing your own methods 🔧
