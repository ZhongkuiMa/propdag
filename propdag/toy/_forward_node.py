__docformat__ = "restructuredtext"
__all__ = ["ForwardToyNode"]

from ._arguments import *
from ._cache import *
from ..template import *


class ForwardToyNode(TNode):
    """
    Node implementation for forward propagation in toy models.

    This class implements forward-only propagation logic with simple bound
    calculation and caching behaviors, providing verbose output for educational
    purposes.

    :ivar _name: Name identifier of the node
    :type _name: str
    :ivar _cache: Shared toy cache instance
    :type _cache: ToyCache
    :ivar _argument: Shared toy argument instance
    :type _argument: ToyArgument
    :ivar _pre_nodes: List of predecessor nodes
    :type _pre_nodes: list[ForwardToyNode]
    :ivar _next_nodes: List of successor nodes
    :type _next_nodes: list[ForwardToyNode]
    """

    _name: str
    _cache: ToyCache
    _argument: ToyArgument
    _pre_nodes: list["ForwardToyNode"]
    _next_nodes: list["ForwardToyNode"]

    def forward(self):
        """
        Perform forward computation for this node.

        For input nodes, validates that bounds are already set.
        For other nodes, builds relaxations, propagates symbolic bounds,
        and calculates bounds.
        """
        if len(self._pre_nodes) == 0:
            # For the input node
            assert self.name in self.cache.bnds
            print(f"{self.name}: Skip input node")
            return

        self.cache.cur_node = self

        self._build_rlx()
        self._fwdprop_symbnd()
        self._cal_and_update_cur_node_bnd()

    def backward(self):
        """
        Not supported in forward propagation mode.

        :raises RuntimeError: Always, as backward pass is not supported
        """
        raise RuntimeError("Backward pass is not supported in forward mode")

    def clear_fwd_cache(self):
        """
        Clear forward computation caches for this node.

        Removes bounds for internal nodes (preserving input and output nodes) and
        symbolic bounds for all nodes.
        """
        if len(self.next_nodes) > 0 and len(self.pre_nodes) > 0:
            # We need keep the bounds of the input and output nodes
            print(f"{self.name}: Clear forward cache of bounds")
            del self.cache.bnds[self.name]
        print(f"{self.name}: Clear forward cache of symbolic bounds")
        del self.cache.symbnds[self.name]

    def clear_bwd_cache(self):
        """
        Not supported in forward propagation mode.

        :raises RuntimeError: Always, as backward pass is not supported
        """
        raise RuntimeError("Backward pass is not supported in forward mode")

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

    def _build_rlx(self):
        """
        Build relaxations for non-linear operations.

        Prints a descriptive message about relaxation calculation.
        """
        print(f"{self.name}: Calculate relaxation if this is non-linear node")

    def _fwdprop_symbnd(self):
        """
        Forward propagate symbolic bounds from predecessor nodes.

        For input nodes, initializes symbolic bounds.
        For other nodes, propagates bounds from predecessors.
        Caches the resulting symbolic bounds.
        """
        if len(self.pre_nodes) == 0:  # For the input node
            print(f"{self.name}: Prepare symbolic bounds of {self.name}")
        else:
            pre_names = [pre_node.name for pre_node in self.pre_nodes]
            print(f"{self.name}: Forward propagate symbolic bounds of {pre_names}")
        print(f"{self.name}: Cache symbolic bounds")
        self.cache.symbnds[self.name] = ("symbolic bounds",)

    def _bwdprop_symbnd(self):
        """
        Not supported in forward propagation mode.

        :raises RuntimeError: Always, as backward propagation is not supported
        """
        raise RuntimeError("Backward pass is not supported in forward mode")

    def _cal_and_update_cur_node_bnd(self):
        """
        Calculate and cache scalar bounds for this node.

        Computes concrete numerical bounds based on symbolic bounds
        and updates the cache.
        """
        print(f"{self.name}: Calculate scalar bounds")
        print(f"{self.name}: Cache scalar bounds")
        self.cache.bnds[self.name] = ("scalar bounds",)
