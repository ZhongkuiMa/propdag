__docformat__ = "restructuredtext"
__all__ = ["ForwardToyNode"]

from ._arguments import *
from ._cache import *
from ..template import *


class ForwardToyNode(TNode):
    _name: str
    _cache: ToyCache
    _argument: ToyArgument
    _pre_nodes: list["ForwardToyNode"]
    _next_nodes: list["ForwardToyNode"]

    def forward(self):
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
        raise RuntimeError("Backward pass is not supported in forward mode")

    def clear_fwd_cache(self):
        if len(self.next_nodes) > 0 and len(self.pre_nodes) > 0:
            # We need keep the bounds of the input and output nodes
            print(f"{self.name}: Clear forward cache of bounds")
            del self.cache.bnds[self.name]
        print(f"{self.name}: Clear forward cache of symbolic bounds")
        del self.cache.symbnds[self.name]

    def clear_bwd_cache(self):
        raise RuntimeError("Backward pass is not supported in forward mode")

    @property
    def cache(self) -> ToyCache:
        return self._cache

    @property
    def argument(self) -> ToyArgument:
        return self._argument

    def _build_rlx(self):
        print(f"{self.name}: Calculate relaxation if this is non-linear node")

    def _fwdprop_symbnd(self):
        if len(self.pre_nodes) == 0:  # For the input node
            print(f"{self.name}: Prepare symbolic bounds of {self.name}")
        else:
            pre_names = [pre_node.name for pre_node in self.pre_nodes]
            print(f"{self.name}: Forward propagate symbolic bounds of {pre_names}")
        print(f"{self.name}: Cache symbolic bounds")
        self.cache.symbnds[self.name] = ("symbolic bounds",)

    def _bwdprop_symbnd(self):
        raise RuntimeError("Backward pass is not supported in forward mode")

    def _cal_and_update_cur_node_bnd(self):
        print(f"{self.name}: Calculate scalar bounds")
        print(f"{self.name}: Cache scalar bounds")
        self.cache.bnds[self.name] = ("scalar bounds",)
