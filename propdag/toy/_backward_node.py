__docformat__ = "restructuredtext"
__all__ = ["BackwardToyNode"]

from ..template import *
from .arguments import *
from .cache import *


class BackwardToyNode(TNode):
    def forward(self):
        print(f"FORWARD {self.name}".center(80, "="))
        self.cache.cur_node = self

        self._build_rlxs()

    def backward(self):
        print(f"BACKWARD {self.name}".center(80, "-"))

        self._bwdprop_symbnds()
        self._cal_bnds()  # This may bot be valid for all nodes.

    def clear_fwd_cache(self):
        if len(self.next_nodes) > 0:
            # We need keep the bounds of the last node that has no next nodes.
            print(f"{self.name}: Clear forward cache of bounds")
            del self.cache.bnds[self.name]

    def clear_bwd_cache(self):
        print(f"{self.name}: Clear backforward cache of symbolic bounds")
        del self.cache.symbnds[self.name]

    @property
    def cache(self) -> ToyCache:
        return self._cache

    @property
    def arguments(self) -> ToyArguments:
        return self._arguments

    def _build_rlxs(self):
        print(f"{self.name}: Calculate relaxation if this is non-linear node")

    def _fwdprop_symbnds(self):
        raise RuntimeError("Forward pass is not supported in backward mode")

    def _bwdprop_symbnds(self):
        if self == self.cache.cur_node:
            print(f"{self.name}: Prepare symbolic bounds of {self.name}")
        else:
            next_names = [next_node.name for next_node in self.next_nodes]
            print(f"{self.name}: Backsubstitute symbolic bounds of {next_names}")
        print(f"{self.name}: Cache substitution")
        self.cache.symbnds[self.name] = (f"substitution of {self.name}",)

    def _cal_bnds(self):
        cur_name = self.cache.cur_node.name
        print(f"{self.name}: Calculate scalar bounds of {cur_name}")
        print(f"{self.name}: Cache scalar bounds")
        self.cache.bnds[cur_name] = ("scalar bounds",)
