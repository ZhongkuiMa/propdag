__docformat__ = "restructuredtext"
__all__ = ["TNode"]

from abc import ABC

from ._arguments import TArgument
from ._cache import TCache


class TNode(ABC):
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
    _cache: TCache
    _argument: TArgument
    _pre_nodes: list["TNode"]
    _next_nodes: list["TNode"]

    def __init__(self, name: str, cache: TCache, argument: TArgument):
        """
        Initialize a node in the computational graph.

        :param name: Name of the node
        :type name: str
        :param cache: Shared cache instance
        :type cache: TCache
        :param argument: Shared arguments instance
        :type argument: TArgument
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
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def backward(self):
        """
        Perform backward computation for this node.

        This method should be overridden by subclasses to implement
        specific backward computation logic.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def clear_fwd_cache(self):
        """
        Clear forward computation cache.

        This method should be overridden by subclasses to implement
        specific cache clearing for forward computation.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def clear_bwd_cache(self):
        """
        Clear backward computation cache.

        This method should be overridden by subclasses to implement
        specific cache clearing for backward computation.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _init_symbnd(self):
        """
        Initialize symbolic bounds.

        This method should be overridden by subclasses to implement
        initialization of symbolic bounds.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _build_rlx(self):
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
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _fwdprop_symbnd(self):
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
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _bwdprop_symbnd(self):
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
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _cal_and_update_cur_node_bnd(self):
        """
        Calculate and update bounds for current node.

        This method should be overridden by subclasses to implement
        bound calculation and updating.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    @property
    def name(self):
        """
        Get node name.

        :returns: Name of this node
        :rtype: str
        """
        return self._name

    @property
    def cache(self) -> TCache:
        """
        Get shared cache instance.

        :returns: Cache instance shared across nodes
        :rtype: TCache
        """
        return self._cache

    @cache.setter
    def cache(self, value: TCache):
        """
        Set shared cache instance.

        :param value: Cache instance to use
        :type value: TCache
        """
        self._cache = value

    @property
    def argument(self) -> TArgument:
        """
        Get shared arguments instance.

        :returns: Arguments instance shared across nodes
        :rtype: TArgument
        """
        return self._argument

    @argument.setter
    def argument(self, value: TArgument):
        """
        Set shared arguments instance.

        :param value: Arguments instance to use
        :type value: TArgument
        """
        self._argument = value

    @property
    def pre_nodes(self) -> list["TNode"]:
        """
        Get predecessor nodes.

        :returns: List of nodes that precede this node
        :rtype: list[TNode]
        """
        return self._pre_nodes

    @pre_nodes.setter
    def pre_nodes(self, value: list["TNode"]):
        """
        Set predecessor nodes.

        :param value: List of nodes that precede this node
        :type value: list[TNode]
        """
        self._pre_nodes = value

    @property
    def next_nodes(self) -> list["TNode"]:
        """
        Get successor nodes.

        :returns: List of nodes that succeed this node
        :rtype: list[TNode]
        """
        return self._next_nodes

    @next_nodes.setter
    def next_nodes(self, value: list["TNode"]):
        """
        Set successor nodes.

        :param value: List of nodes that succeed this node
        :type value: list[TNode]
        """
        self._next_nodes = value
