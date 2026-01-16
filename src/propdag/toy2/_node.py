__docformat__ = "restructuredtext"
__all__ = ["Toy2Node"]

from propdag.template2 import T2Node
from propdag.toy2._arguments import Toy2Argument
from propdag.toy2._cache import Toy2Cache


class Toy2Node(T2Node[Toy2Cache, Toy2Argument]):
    """
    Toy node implementation for reversed graph semantics.

    **SEMANTIC SHIFT from BackwardBoundToyNode**:
    - BackwardBoundToyNode.forward(): Computes forward bounds
    - BackwardBoundToyNode.backward_bound(): Propagates backward bounds
    - Toy2Node.forward(): DOES BOTH (because graph is reversed!)

    **Method Mapping (Old → New)**:
    - backward_bound() → merged into forward()
    - build_inverse_rlx() → build_rlx() (semantic shift)
    - bwdprop_bounds() → propagate_bounds()
    - intersect_and_update_bnd() → unchanged

    **Cache Field Mapping (Old → New)**:
    - cache.back_bnds → cache.bnds (primary bounds)
    - cache.inv_rlxs → cache.rlxs (primary relaxations)
    - cache.bnds → cache.fwd_bnds (forward bounds for intersection)

    Demonstrates backward bound propagation through reversed graph with
    verbose logging for educational purposes.

    :ivar _name: Node identifier
    :ivar _cache: Shared Toy2Cache storing bounds and relaxations
    :ivar _argument: Shared Toy2Argument
    :ivar _pre_nodes: Predecessors in REVERSED graph (user's successors!)
    :ivar _next_nodes: Successors in REVERSED graph (user's predecessors!)
    """

    def forward(self):
        """
        Forward propagation through reversed graph.

        **IMPORTANT**: Because graph is reversed, this propagates bounds
        BACKWARD (from user's output to input).

        Process:
        - If pre_nodes=[] (user's output): Initialize bounds from output constraints
        - Otherwise: Build inverse relaxations, propagate, and intersect

        This combines what was forward() + backward_bound() in BackwardBoundToyNode.
        """
        # Input node in reversed graph = user's OUTPUT node
        if len(self._pre_nodes) == 0:
            print(
                f"[INIT] {self.name}.forward() | initialize output → bnds[{self.name}] [reversed_input]"
            )
            # In real implementation, these would come from output specification
            self.cache.bnds[self.name] = ("output constraint bounds",)
            return

        # Non-input nodes: propagate bounds from predecessors in reversed graph
        self.cache.cur_node = self

        # Step 1: Build inverse relaxations
        self.build_rlx()  # Now semantically correct (builds inverse)

        # Step 2: Propagate bounds from predecessors in reversed graph
        self.propagate_bounds()

        # Step 3: Intersect with forward bounds (if available)
        self.intersect_and_update_bnd()

    def backward(self):
        """
        Backward propagation.

        :raises RuntimeError: Always
        """
        raise RuntimeError("backward() is not implemented in Toy2Node.")

    def clear_fwd_cache(self):
        """
        Clear forward computation caches.

        In template2, "forward" means propagation through reversed graph
        (which is backward propagation in user's view).
        """
        # Only clear internal nodes (preserve input/output)
        if len(self.next_nodes) > 0 and len(self.pre_nodes) > 0:
            print(f"[CLEAR] {self.name}.clear_fwd_cache() | clear fwd_cache → bnds[{self.name}]")
            del self.cache.bnds[self.name]
        if self.name in self.cache.symbnds:
            print(f"[CLEAR] {self.name}.clear_fwd_cache() | clear fwd_cache → symbnds[{self.name}]")
            del self.cache.symbnds[self.name]
        if self.name in self.cache.rlxs:
            print(f"[CLEAR] {self.name}.clear_fwd_cache() | clear fwd_cache → rlxs[{self.name}]")
            del self.cache.rlxs[self.name]

    def clear_bwd_cache(self):
        """
        Clear backward propagation cache.

        :raises RuntimeError: Always
        """
        raise RuntimeError("clear_bwd_cache() is not implemented in Toy2Node.")

    def init_symbnd(self):
        """
        Initialize symbolic bounds.

        :raises RuntimeError: Always
        """
        raise RuntimeError("init_symbnd() is not implemented in Toy2Node.")

    # Core implementation methods

    def build_rlx(self):
        """
        Build inverse relaxations for backward propagation.

        **SEMANTIC SHIFT**: In template/, build_rlx() builds forward relaxations.
        In template2/, build_rlx() builds INVERSE relaxations because the graph
        is reversed. The method name now matches its usage.
        """
        print(f"[RELAX] {self.name}.build_rlx() | compute inverse_rlx → rlxs[{self.name}]")
        print(f"[CACHE] {self.name}.build_rlx() | store rlxs → cache.rlxs[{self.name}]")
        self.cache.rlxs[self.name] = ("inverse relaxation",)

    def fwdprop_symbnd(self):
        """
        Forward propagate symbolic bounds (may not be used in all implementations).

        In template2, this might be used for bound calculation if symbolic
        expressions are needed. Optional method.
        """
        if len(self.pre_nodes) == 0:
            print(
                f"[PROPAGATE] {self.name}.forward() | prepare symbnds → symbnds[{self.name}] [input]"
            )
        else:
            pre_names = [pre_node.name for pre_node in self.pre_nodes]
            pre_str = ", ".join(pre_names)
            print(
                f"[PROPAGATE] {self.name}.forward() | fwd_propagate → symbnds[{self.name}] [from: {pre_str}]"
            )

        print(f"[CACHE] {self.name}.forward() | store symbnds → cache.symbnds[{self.name}]")
        self.cache.symbnds[self.name] = ("symbolic bounds",)

    def cal_and_update_cur_node_bnd(self):
        """
        Calculate and update bounds for current node.

        In template2, this typically computes bounds and stores them in cache.bnds.
        """
        print(f"[COMPUTE] {self.name}.forward() | calculate bounds → bnds[{self.name}]")
        print(f"[CACHE] {self.name}.forward() | store bnds → cache.bnds[{self.name}]")
        self.cache.bnds[self.name] = ("computed bounds",)

    # New methods for template2

    def propagate_bounds(self):
        """
        Propagate bounds from predecessors in reversed graph.

        **RENAMED from bwdprop_bounds()** for clarity in template2 context.

        In reversed graph:
        - pre_nodes are the user's successor nodes
        - We propagate FROM pre_nodes TO this node
        """
        pre_names = [pre_node.name for pre_node in self.pre_nodes]
        pre_str = ", ".join(pre_names)
        print(
            f"[PROPAGATE] {self.name}.forward() | propagate bounds → bnds[{self.name}] [from: {pre_str}]"
        )
        print(f"[CACHE] {self.name}.forward() | store bnds → cache.bnds[{self.name}]")
        self.cache.bnds[self.name] = ("propagated bounds",)

    def intersect_and_update_bnd(self):
        """
        Intersect forward and backward bounds and update cache.

        In template2:
        - cache.fwd_bnds[name]: Forward bounds from initial pass
        - cache.bnds[name]: Backward-propagated bounds (via reversed graph)
        - Result: Tightened bounds = intersection

        In toy implementation, we just simulate this with a message.
        """
        print(f"[COMPUTE] {self.name}.forward() | intersect bounds → bnds[{self.name}] [fwd ∩ bwd]")
        print(f"[CACHE] {self.name}.forward() | update tightened → cache.bnds[{self.name}]")
        # In real implementation:
        # fwd = self.cache.fwd_bnds.get(self.name)
        # bwd = self.cache.bnds[self.name]
        # self.cache.bnds[self.name] = intersect(fwd, bwd)
        self.cache.bnds[self.name] = ("tightened bounds (forward ∩ backward)",)
