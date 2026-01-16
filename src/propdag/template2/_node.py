__docformat__ = "restructuredtext"
__all__ = ["T2ArgumentType", "T2CacheType", "T2Node"]

from abc import ABC
from collections.abc import Sequence
from typing import Generic, TypeVar

from propdag.template2._arguments import T2Argument
from propdag.template2._cache import T2Cache

# Type variables for template2 components
T2CacheType = TypeVar("T2CacheType", bound=T2Cache)
T2ArgumentType = TypeVar("T2ArgumentType", bound=T2Argument)


class T2Node(ABC, Generic[T2CacheType, T2ArgumentType]):
    """
    Abstract base class for reversed graph computational nodes.

    **REVERSED GRAPH SEMANTICS**:
    Template2 uses reversed graph structure where "forward" propagation
    achieves backward bound propagation without semantic confusion.

    **Graph Structure After Reversal**:
    - User builds: Input → Hidden → Output
    - T2Model reverses to: Output → Hidden → Input
    - T2Node "input" (pre_nodes=[]): User's OUTPUT (constraint origin)
    - T2Node "output" (next_nodes=[]): User's INPUT (final bounds)

    **Method Semantics (changed by graph reversal)**:
    - forward(): Propagates bounds backward (output → input in user's view)
    - build_rlx(): Builds inverse relaxations (was build_inverse_rlx in template/)

    **Key Responsibilities**:
    - Build inverse relaxations for non-linear operations
    - Propagate bound constraints backward through reversed graph
    - Calculate concrete numerical bounds
    - Manage caches for intermediate results
    - Intersect backward-propagated bounds with forward bounds

    :ivar _name: Unique identifier for this node
    :ivar _cache: Shared cache for bounds and relaxations
    :ivar _argument: Shared arguments
    :ivar _pre_nodes: Predecessor nodes in REVERSED graph (user's successors)
    :ivar _next_nodes: Successor nodes in REVERSED graph (user's predecessors)
    """

    _name: str
    _cache: T2CacheType
    _argument: T2ArgumentType
    _pre_nodes: list["T2Node[T2CacheType, T2ArgumentType]"]
    _next_nodes: list["T2Node[T2CacheType, T2ArgumentType]"]

    def __init__(self, name: str, cache: T2CacheType, argument: T2ArgumentType):
        """
        Initialize a node in the reversed computational graph.

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
        Forward propagation through reversed graph.

        **SEMANTIC SHIFT**: Because the graph is reversed, this method
        actually propagates bounds backward (from user's output to input).
        This eliminates the semantic confusion of "backward_bound()" method.

        Process (at each node from output to input in user's view):
        1. Build inverse relaxations
        2. Propagate bounds from successors in reversed graph
        3. Intersect with forward bounds from initial pass

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def backward(self):
        """
        Backward propagation for BACKWARD mode.

        Template2 supports backward mode same as template, just with reversed
        graph semantics. This method is used when prop_mode is BACKWARD.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def clear_fwd_cache(self):
        """
        Clear forward computation cache.

        In template2 context, this clears caches from the "forward" pass
        through the reversed graph (which is backward propagation in user's view).

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def clear_bwd_cache(self):
        """
        Clear backward propagation cache.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def init_symbnd(self):
        """
        Initialize symbolic bounds.

        May or may not be used depending on implementation. Some template2
        implementations might compute bounds directly without symbolic expressions.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def build_rlx(self):
        """
        Build relaxation (INVERSE relaxation in template2 semantics).

        **SEMANTIC SHIFT**: In template/, build_rlx() builds forward relaxations.
        In template2/, build_rlx() builds inverse relaxations because the graph
        is reversed. This makes the method name match its actual usage.

        For non-linear activations (ReLU, sigmoid, etc.), compute inverse
        linear relaxations that propagate constraints backward.

        Example for ReLU inverse::

            # For y = ReLU(x), given bounds on y in [y_l, y_u]:
            # Compute bounds on x that could produce these y values
            if y_l > 0: x >= y_l  # ReLU was active
            if y_u == 0: x <= 0   # ReLU was inactive
            else: compute inverse triangle relaxation

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def fwdprop_symbnd(self):
        """
        Forward propagate symbolic bounds through reversed graph.

        **SEMANTIC SHIFT**: In reversed graph, "forward" means output→input.
        This might be used for bound calculation if symbolic expressions are needed.

        :raises RuntimeError: If not implemented by subclass
        """
        raise RuntimeError(f"This method should be instantiated in {type(self).__name__}.")

    def bwdprop_symbnd(self):
        """
        Backward propagate symbolic bounds.

        Template2 doesn't support symbolic back-substitution.

        :raises RuntimeError: Always
        """
        raise RuntimeError("bwdprop_symbnd() is not supported in template2.")

    def cal_and_update_cur_node_bnd(self):
        """
        Calculate and update bounds for current node.

        Computes concrete bounds and stores them in cache. In template2,
        this typically involves intersecting backward-propagated bounds
        with forward bounds.

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
    def cache(self) -> T2CacheType:
        """
        Get shared cache instance.

        :returns: Cache instance shared across nodes
        """
        return self._cache

    @cache.setter
    def cache(self, value: T2CacheType):
        """
        Set shared cache instance.

        :param value: Cache instance to use
        """
        self._cache = value

    @property
    def argument(self) -> T2ArgumentType:
        """
        Get shared arguments instance.

        :returns: Arguments instance shared across nodes
        """
        return self._argument

    @argument.setter
    def argument(self, value: T2ArgumentType):
        """
        Set shared arguments instance.

        :param value: Arguments instance to use
        """
        self._argument = value

    @property
    def pre_nodes(self) -> Sequence["T2Node[T2CacheType, T2ArgumentType]"]:
        """
        Get predecessor nodes in REVERSED graph.

        **IMPORTANT**: After graph reversal, pre_nodes are the user's successor nodes!
        Example: If user built A→B, after reversal B.pre_nodes = [] and A.pre_nodes = [B]

        :returns: Sequence of nodes that precede this node in reversed graph
        """
        return self._pre_nodes

    @pre_nodes.setter
    def pre_nodes(self, value: list["T2Node[T2CacheType, T2ArgumentType]"]):
        """
        Set predecessor nodes in REVERSED graph.

        :param value: List of nodes that precede this node in reversed graph
        """
        self._pre_nodes = value

    @property
    def next_nodes(self) -> Sequence["T2Node[T2CacheType, T2ArgumentType]"]:
        """
        Get successor nodes in REVERSED graph.

        **IMPORTANT**: After graph reversal, next_nodes are the user's predecessor nodes!
        Example: If user built A→B, after reversal A.next_nodes = [B] and B.next_nodes = []

        :returns: Sequence of nodes that succeed this node in reversed graph
        """
        return self._next_nodes

    @next_nodes.setter
    def next_nodes(self, value: list["T2Node[T2CacheType, T2ArgumentType]"]):
        """
        Set successor nodes in REVERSED graph.

        :param value: List of nodes that succeed this node in reversed graph
        """
        self._next_nodes = value
