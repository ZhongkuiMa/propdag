"""
Toy2: Example implementation for reversed graph semantics.

This module provides concrete implementations demonstrating backward bound
propagation with reversed graph semantics. It serves as an educational
reference for building real implementations.

**Purpose**:
- Demonstrate template2 usage with verbose logging
- Show how backward bound propagation works through reversed graphs
- Provide a reference implementation pattern

**Comparison to toy/ module**:
- toy/: Uses template/ with forward graph + backward_bound() method (confusing!)
- toy2/: Uses template2/ with reversed graph + forward() method (clear!)

Example Usage::

    from propdag.toy2 import Toy2Node, Toy2Cache, Toy2Argument, Toy2Model

    # Create cache and arguments
    cache = Toy2Cache()
    cache.fwd_bnds["Input"] = ("initial forward bounds",)
    args = Toy2Argument(verbose=True)

    # Build graph normally (Input → Output)
    input_node = Toy2Node("Input", cache, args)
    output_node = Toy2Node("Output", cache, args)

    input_node.next_nodes = [output_node]
    output_node.pre_nodes = [input_node]

    # Model automatically reverses graph and runs
    model = Toy2Model([input_node, output_node], verbose=True)
    model.run()  # Prints execution trace

    # Verify results
    assert "Input" in cache.bnds
    assert "Output" in cache.bnds

Components:
- Toy2Argument: Arguments with verbose flag
- Toy2Cache: Cache inheriting from T2Cache
- Toy2Model: Model with verbose logging
- Toy2Node: Node with verbose bound propagation logging
"""

from propdag.toy2._arguments import Toy2Argument
from propdag.toy2._cache import Toy2Cache
from propdag.toy2._model import Toy2Model
from propdag.toy2._node import Toy2Node

__all__ = [
    "Toy2Argument",
    "Toy2Cache",
    "Toy2Model",
    "Toy2Node",
]
