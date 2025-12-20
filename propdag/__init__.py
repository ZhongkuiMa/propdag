"""
PropDAG: Bound propagation for DAG-structured neural networks.

Provides abstract templates and example implementations for bound propagation
algorithms in neural network verification. Supports both forward and backward
propagation through directed acyclic graphs with residual/skip connections.

Main components:
- template: Abstract base classes (TNode, TModel, TCache, TArgument)
- toy: Example implementations with verbose logging
- utils: PropMode enum (FORWARD/BACKWARD)
"""

from propdag.propdag.template import *
from propdag.propdag.toy import *
from propdag.propdag.utils import *
