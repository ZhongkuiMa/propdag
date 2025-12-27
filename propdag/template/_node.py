__docformat__ = "restructuredtext"
__all__ = ["TNode"]

from abc import ABC
from collections.abc import Sequence
from typing import Generic

from propdag.custom_types import ArgumentType, CacheType


class TNode(ABC, Generic[CacheType, ArgumentType]):
    """
    Abstract base class for computational graph nodes.

    Each node represents a layer or operation in a neural network DAG.
    Nodes must implement forward/backward propagation, bound calculation,
    and cache management.

    **Key responsibilities:**
    - Build relaxations for non-linear operations (e.g., ReLU)
    - Propagate symbolic bounds through the graph
    - Calculate concrete numerical bounds
    - Manage caches for intermediate results

    **Graph structure:**
    - Input nodes: no predecessors, bounds provided externally
    - Hidden nodes: have predecessors and successors
    - Output nodes: no successors, final bounds computed here

    :ivar _name: Unique identifier for this node
    :ivar _cache: Shared cache for bounds and symbolic expressions
    :ivar _argument: Shared arguments (e.g., verification parameters)
    :ivar _pre_nodes: Predecessor nodes (inputs to this operation)
    :ivar _next_nodes: Successor nodes (consumers of this operation)
    """

    _name: str
    _cache: CacheType
    _argument: ArgumentType
    _pre_nodes: list["TNode[CacheType, ArgumentType]"]
    _next_nodes: list["TNode[CacheType, ArgumentType]"]

    def __init__(self, name: str, cache: CacheType, argument: ArgumentType):
        """
        Initialize a node in the computational graph.

        :param name: Name of the node
        :param cache: Shared cache instance
        :param argument: Shared arguments instance
        """
        self._name = name
        self._cache = cache
        self._argument = argument
        self._pre_nodes = []
        self._next_nodes = []

    def forward(self):
        """
        Perform forward computation for this node.

        This method should be overridden by subclasses to implement
        specific forward computation logic.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def backward(self):
        """
        Perform backward computation for this node.

        This method should be overridden by subclasses to implement
        specific backward computation logic.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def clear_fwd_cache(self):
        """
        Clear forward computation cache.

        This method should be overridden by subclasses to implement
        specific cache clearing for forward computation.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def clear_bwd_cache(self):
        """
        Clear backward computation cache.

        This method should be overridden by subclasses to implement
        specific cache clearing for backward computation.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def init_symbnd(self):
        """
        Initialize symbolic bounds.

        This method should be overridden by subclasses to implement
        initialization of symbolic bounds.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def build_rlx(self):
        """
        Build relaxation for non-linear operations.

        For non-linear activations (ReLU, sigmoid, etc.), compute linear
        relaxations (upper/lower bounds) that over-approximate the activation.

        Example for ReLU::

            # For x in [l, u]:
            if l >= 0: relaxation is identity
            elif u <= 0: relaxation is zero
            else: compute triangle relaxation

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def fwdprop_symbnd(self):
        """
        Forward propagate symbolic bounds.

        Combines symbolic bounds from predecessor nodes according to this
        node's operation. For linear ops (Conv, FC), composes affine expressions.
        For non-linear ops, applies relaxations.

        Example::

            # For y = Wx + b:
            symbolic_y = W * symbolic_x + b

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def bwdprop_symbnd(self):
        """
        Backward propagate symbolic bounds via substitution.

        Back-substitutes symbolic expressions from successor nodes to
        eliminate intermediate variables, yielding tighter bounds directly
        in terms of input variables.

        Example::

            # If z = f(y) and y = g(x), substitute:
            z_symbolic = f(g(x))  # Eliminates y

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def cal_and_update_cur_node_bnd(self):
        """
        Calculate and update bounds for current node.

        This method should be overridden by subclasses to implement
        bound calculation and updating.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    @property
    def name(self):
        """
        Get node name.

        :returns: Name of this node
        """
        return self._name

    @property
    def cache(self) -> CacheType:
        """
        Get shared cache instance.

        :returns: Cache instance shared across nodes
        """
        return self._cache

    @cache.setter
    def cache(self, value: CacheType):
        """
        Set shared cache instance.

        :param value: Cache instance to use
        """
        self._cache = value

    @property
    def argument(self) -> ArgumentType:
        """
        Get shared arguments instance.

        :returns: Arguments instance shared across nodes
        """
        return self._argument

    @argument.setter
    def argument(self, value: ArgumentType):
        """
        Set shared arguments instance.

        :param value: Arguments instance to use
        """
        self._argument = value

    @property
    def pre_nodes(self) -> Sequence["TNode[CacheType, ArgumentType]"]:
        """
        Get predecessor nodes.

        :returns: Sequence of nodes that precede this node
        """
        return self._pre_nodes

    @pre_nodes.setter
    def pre_nodes(self, value: list["TNode[CacheType, ArgumentType]"]):
        """
        Set predecessor nodes.

        :param value: List of nodes that precede this node
        """
        self._pre_nodes = value

    @property
    def next_nodes(self) -> Sequence["TNode[CacheType, ArgumentType]"]:
        """
        Get successor nodes.

        :returns: Sequence of nodes that succeed this node
        """
        return self._next_nodes

    @next_nodes.setter
    def next_nodes(self, value: list["TNode[CacheType, ArgumentType]"]):
        """
        Set successor nodes.

        :param value: List of nodes that succeed this node
        """
        self._next_nodes = value
