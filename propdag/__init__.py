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

from .template import *
from .toy import *
from .utils import *
