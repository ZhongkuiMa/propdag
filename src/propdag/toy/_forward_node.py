__docformat__ = "restructuredtext"
__all__ = ["ForwardToyNode"]

from propdag.template import TNode
from propdag.toy._arguments import ToyArgument
from propdag.toy._cache import ToyCache


class ForwardToyNode(TNode[ToyCache, ToyArgument]):
    """
    Example node for forward bound propagation.

    Demonstrates forward-only propagation with verbose logging for
    educational purposes. Shows the flow of:
    1. Building relaxations (for non-linear layers)
    2. Forward propagating symbolic bounds
    3. Calculating concrete bounds
    4. Cache management

    Use this as a template for implementing your own forward propagation
    algorithms (e.g., CROWN, DeepPoly forward pass).

    :ivar _name: Node identifier (e.g., "Conv1", "ReLU2")
    :ivar _cache: Shared cache storing bounds and symbolic expressions
    :ivar _argument: Shared arguments (verification parameters)
    :ivar _pre_nodes: Input nodes to this operation
    :ivar _next_nodes: Output nodes consuming this result
    """

    # Inherited from TNode[ToyCache, ToyArgument]

    def forward(self):
        """
        Execute forward pass for this node.

        **Process:**
        1. Input nodes: Verify initial bounds already set (provided externally)
        2. Hidden/output nodes:
           a. Build relaxation (if non-linear layer like ReLU)
           b. Forward propagate symbolic bounds from predecessors
           c. Calculate concrete numerical bounds
        """
        # Input node: bounds provided externally (e.g., input specification)
        if len(self._pre_nodes) == 0:
            assert self.name in self.cache.bnds, f"Input bounds not set for {self.name}"
            print(f"{self.name}: Skip input node")
            return

        # Non-input node: compute bounds via propagation
        self.cache.cur_node = self

        self.build_rlx()  # Step 1: Relaxation for non-linear ops
        self.fwdprop_symbnd()  # Step 2: Symbolic bound propagation
        self.cal_and_update_cur_node_bnd()  # Step 3: Concrete bound calculation

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
        # Only clear symbnds if they exist (input nodes may not create symbnds)
        if self.name in self.cache.symbnds:
            print(f"{self.name}: Clear forward cache of symbolic bounds")
            del self.cache.symbnds[self.name]

    def clear_bwd_cache(self):
        """
        Not supported in forward propagation mode.

        :raises RuntimeError: Always, as backward pass is not supported
        """
        raise RuntimeError("Backward pass is not supported in forward mode")

    # Inherited properties: cache, argument (avoid override issues)

    def build_rlx(self):
        """
        Build relaxations for non-linear operations.

        Prints a descriptive message about relaxation calculation.
        """
        print(f"{self.name}: Calculate relaxation if this is non-linear node")

    def fwdprop_symbnd(self):
        """
        Forward propagate symbolic bounds from predecessors.

        **Logic:**
        - Input nodes: Initialize symbolic identity (e.g., x = x)
        - Hidden nodes: Combine predecessor symbolics via this operation
          Example: For y = Wx + b, symbolic_y = W * symbolic_x + b

        Resulting symbolic expression is cached for later use.
        """
        if len(self.pre_nodes) == 0:  # Input node
            print(f"{self.name}: Prepare symbolic bounds of {self.name}")
        else:  # Hidden/output node
            pre_names = [pre_node.name for pre_node in self.pre_nodes]
            print(f"{self.name}: Forward propagate symbolic bounds of {pre_names}")

        # Cache symbolic expression (tuple placeholder in toy example)
        print(f"{self.name}: Cache symbolic bounds")
        self.cache.symbnds[self.name] = ("symbolic bounds",)

    def bwdprop_symbnd(self):
        """
        Not supported in forward propagation mode.

        :raises RuntimeError: Always, as backward propagation is not supported
        """
        raise RuntimeError("Backward pass is not supported in forward mode")

    def cal_and_update_cur_node_bnd(self):
        """
        Calculate and cache scalar bounds for this node.

        Computes concrete numerical bounds based on symbolic bounds
        and updates the cache.
        """
        print(f"{self.name}: Calculate scalar bounds")
        print(f"{self.name}: Cache scalar bounds")
        self.cache.bnds[self.name] = ("scalar bounds",)
