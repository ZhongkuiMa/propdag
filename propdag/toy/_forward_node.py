__docformat__ = "restructuredtext"
__all__ = ["ForwardToyNode"]

from ..template import *
from ._arguments import *
from ._cache import *


class ForwardToyNode(TNode):
    _name: str
    _cache: ToyCache
    _arguments: ToyArguments
    _pre_nodes: list["ForwardToyNode"]
    _next_nodes: list["ForwardToyNode"]

    def forward(self):
        print(f"FORWARD {self.name}".center(80, "="))
        self.cache.cur_node = self

        self._build_rlxs()
        self._fwdprop_symbnds()
        self._cal_bnds()

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
    def arguments(self) -> ToyArguments:
        return self._arguments

    def _build_rlxs(self):
        print(f"{self.name}: Calculate relaxation if this is non-linear node")

    def _fwdprop_symbnds(self):
        if len(self.pre_nodes) == 0:  # For the input node
            print(f"{self.name}: Prepare symbolic bounds of {self.name}")
        else:
            pre_names = [pre_node.name for pre_node in self.pre_nodes]
            print(f"{self.name}: Forward propagate symbolic bounds of {pre_names}")
        print(f"{self.name}: Cache symbolic bounds")
        self.cache.symbnds[self.name] = ("symbolic bounds",)

    def _bwdprop_symbnds(self):
        raise RuntimeError("Backward pass is not supported in forward mode")

    def _cal_bnds(self):
        print(f"{self.name}: Calculate scalar bounds")
        print(f"{self.name}: Cache scalar bounds")
        self.cache.bnds[self.name] = ("scalar bounds",)
