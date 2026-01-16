__docformat__ = "restructuredtext"
__all__ = ["BackwardToyNode"]

from propdag.template import TNode
from propdag.toy._arguments import ToyArgument
from propdag.toy._cache import ToyCache


class BackwardToyNode(TNode[ToyCache, ToyArgument]):
    """
    Node implementation for backward propagation in toy models.

    This class implements backward propagation logic with symbolic bound calculation
    and substitution, providing verbose output for educational purposes.

    :ivar _name: Name identifier of the node
    :ivar _cache: Shared toy cache instance
    :ivar _argument: Shared toy argument instance
    :ivar _pre_nodes: List of predecessor nodes
    :ivar _next_nodes: List of successor nodes
    """

    # Inherited from TNode[ToyCache, ToyArgument]

    def forward(self):
        """
        Perform forward initialization for backward propagation.

        For input nodes, validates that bounds are already set.
        For other nodes, builds relaxations and initializes symbolic bounds.
        """
        if len(self._pre_nodes) == 0:
            # For the input node
            assert self.name in self.cache.bnds
            print(f"[INIT] {self.name}.forward() | skip → input_node [no_predecessors]")
            return

        self.cache.cur_node = self

        self.build_rlx()
        self.init_symbnd()

    def backward(self):
        """
        Perform backward propagation for this node.

        Propagates symbolic bounds backward and calculates/updates scalar bounds for
        the current node.
        """
        self.bwdprop_symbnd()
        self.cal_and_update_cur_node_bnd()  # This may bot be valid for all nodes.

    def clear_fwd_cache(self):
        """
        Clear forward computation caches for this node.

        Removes bounds for internal nodes while preserving input and output node bounds.
        """
        if len(self.next_nodes) > 0 and len(self.pre_nodes) > 0:
            # We need keep the bounds of the input and output nodes
            print(f"[CLEAR] {self.name}.clear_fwd_cache() | clear fwd_cache → bnds[{self.name}]")
            del self.cache.bnds[self.name]

    def clear_bwd_cache(self):
        """
        Clear backward computation caches for this node.

        Removes symbolic bounds used during back-substitution.
        """
        print(f"[CLEAR] {self.name}.clear_bwd_cache() | clear bwd_cache → symbnds[{self.name}]")
        del self.cache.symbnds[self.name]

    # Inherited properties: cache, argument (avoid override issues)

    def init_symbnd(self):
        """
        Initialize symbolic bounds for linear nodes.

        Builds initial symbolic bound representation for this node.
        """
        is_cur_node = self.cache.cur_node == self

        if is_cur_node:
            print(
                f"[PROPAGATE] {self.name}.forward() | init_symbnds → symbnds[{self.name}] [cur_node]"
            )
        else:
            cur_name = self.cache.cur_node.name if self.cache.cur_node else "unknown"
            print(
                f"[PROPAGATE] {self.name}.forward() | prepare_symbnds → symbnds[{self.name}] [for: {cur_name}]"
            )

        print(f"[CACHE] {self.name}.forward() | store symbnds → cache.symbnds[{self.name}]")
        self.cache.symbnds[self.name] = ("symbolic bounds",)

    def build_rlx(self):
        """
        Build relaxations for non-linear operations.

        Prints a descriptive message about relaxation calculation.
        """
        print(f"[RELAX] {self.name}.build_rlx() | compute relaxation → rlxs[{self.name}]")

    def fwdprop_symbnd(self):
        """
        Not supported in backward propagation mode.

        :raises RuntimeError: Always, as forward propagation is not supported
        """
        raise RuntimeError("Forward pass is not supported in backward mode. ")

    def bwdprop_symbnd(self):
        """
        Backward propagate symbolic bounds.

        For the current node, initializes symbolic bounds.
        For other nodes, performs back-substitution to propagate bounds.
        Caches the resulting symbolic expressions.
        """
        assert self.cache.cur_node is not None, "cur_node must be set before backward propagation"
        cur_name = self.cache.cur_node.name

        if self == self.cache.cur_node:
            print(
                f"[PROPAGATE] {self.name}.backward() | init_symbnds → symbnds[{self.name}] [cur_node]"
            )
        else:
            print(
                f"[PROPAGATE] {self.name}.backward() | bwd_substitute → symbnds[{self.name}] [from: {cur_name}]"
            )

        print(f"[CACHE] {self.name}.backward() | store substitution → cache.symbnds[{self.name}]")
        self.cache.symbnds[self.name] = (f"substitution of {self.name}",)

    def cal_and_update_cur_node_bnd(self):
        """
        Calculate and cache scalar bounds for the current node.

        Computes concrete numerical bounds based on symbolic bounds from backward
        propagation and updates the cache.
        """
        assert self.cache.cur_node is not None, "cur_node must be set before bound calculation"
        cur_name = self.cache.cur_node.name
        print(f"[COMPUTE] {self.name}.backward() | calculate bounds → bnds[{cur_name}]")
        print(f"[CACHE] {self.name}.backward() | store bnds → cache.bnds[{cur_name}]")
        self.cache.bnds[cur_name] = ("scalar bounds",)
