__docformat__ = "restructuredtext"
__all__ = ["BackwardToyNode"]

from ._arguments import *
from ._cache import *
from ..template import *


class BackwardToyNode(TNode):
    """
    Node implementation for backward propagation in toy models.

    This class implements backward propagation logic with symbolic bound calculation
    and substitution, providing verbose output for educational purposes.

    :ivar _name: Name identifier of the node
    :type _name: str
    :ivar _cache: Shared toy cache instance
    :type _cache: ToyCache
    :ivar _argument: Shared toy argument instance
    :type _argument: ToyArgument
    :ivar _pre_nodes: List of predecessor nodes
    :type _pre_nodes: list[BackwardToyNode]
    :ivar _next_nodes: List of successor nodes
    :type _next_nodes: list[BackwardToyNode]
    """

    _name: str
    _cache: ToyCache
    _argument: ToyArgument
    _pre_nodes: list["BackwardToyNode"]
    _next_nodes: list["BackwardToyNode"]

    def forward(self):
        """
        Perform forward initialization for backward propagation.

        For input nodes, validates that bounds are already set.
        For other nodes, builds relaxations and initializes symbolic bounds.
        """
        if len(self._pre_nodes) == 0:
            # For the input node
            assert self.name in self.cache.bnds
            print(f"{self.name}: Skip input node")
            return

        self.cache.cur_node = self

        self._build_rlx()
        self._init_symbnd()

    def backward(self):
        """
        Perform backward propagation for this node.

        Propagates symbolic bounds backward and calculates/updates scalar bounds for
        the current node.
        """
        self._bwdprop_symbnd()
        self._cal_and_update_cur_node_bnd()  # This may bot be valid for all nodes.

    def clear_fwd_cache(self):
        """
        Clear forward computation caches for this node.

        Removes bounds for internal nodes while preserving input and output node bounds.
        """
        if len(self.next_nodes) > 0 and len(self.pre_nodes) > 0:
            # We need keep the bounds of the input and output nodes
            print(f"{self.name}: Clear forward cache of bounds")
            del self.cache.bnds[self.name]

    def clear_bwd_cache(self):
        """
        Clear backward computation caches for this node.

        Removes symbolic bounds used during back-substitution.
        """
        print(f"{self.name}: Clear backforward cache of symbolic bounds")
        del self.cache.symbnds[self.name]

    @property
    def cache(self) -> ToyCache:
        """
        Get the toy cache instance.

        :returns: The shared toy cache
        :rtype: ToyCache
        """
        return self._cache

    @property
    def argument(self) -> ToyArgument:
        """
        Get the toy argument instance.

        :returns: The shared toy arguments
        :rtype: ToyArgument
        """
        return self._argument

    def _init_symbnd(self):
        """
        Initialize symbolic bounds for linear nodes.

        Builds initial symbolic bound representation for this node.
        """
        print(f"{self.name}: Build symbolic bounds if this is a linear node")

    def _build_rlx(self):
        """
        Build relaxations for non-linear operations.

        Prints a descriptive message about relaxation calculation.
        """
        print(f"{self.name}: Calculate relaxation if this is a non-linear node")

    def _fwdprop_symbnd(self):
        """
        Not supported in backward propagation mode.

        :raises RuntimeError: Always, as forward propagation is not supported
        """
        raise RuntimeError("Forward pass is not supported in backward mode. ")

    def _bwdprop_symbnd(self):
        """
        Backward propagate symbolic bounds.

        For the current node, initializes symbolic bounds.
        For other nodes, performs back-substitution to propagate bounds.
        Caches the resulting symbolic expressions.
        """
        if self == self.cache.cur_node:
            print(f"{self.name}: Prepare symbolic bounds of {self.name}")
        else:
            print(
                f"{self.name}: Backsubstitute symbolic bounds of {self.cache.cur_node.name}"
            )
        print(f"{self.name}: Cache substitution")
        self.cache.symbnds[self.name] = (f"substitution of {self.name}",)

    def _cal_and_update_cur_node_bnd(self):
        """
        Calculate and cache scalar bounds for the current node.

        Computes concrete numerical bounds based on symbolic bounds from backward
        propagation and updates the cache.
        """
        cur_name = self.cache.cur_node.name
        print(f"{self.name}: Calculate scalar bounds of {cur_name}")
        print(f"{self.name}: Cache scalar bounds")
        self.cache.bnds[cur_name] = ("scalar bounds",)
