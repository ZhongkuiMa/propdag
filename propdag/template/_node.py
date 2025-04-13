__docformat__ = "restructuredtext"
__all__ = ["TNode"]

from abc import ABC

from ._arguments import TArguments
from ._cache import TCache


class TNode(ABC):
    _name: str
    _cache: TCache | None
    _arguments: TArguments | None
    _pre_nodes: list["TNode"]
    _next_nodes: list["TNode"]

    def __init__(self, name: str):
        self._name = name
        self._cache = None
        self._arguments = None
        self._pre_nodes = []
        self._next_nodes = []

    def forward(self):
        raise RuntimeError("This method should be instantiated in the child class.")

    def backward(self):
        raise RuntimeError("This method should be instantiated in the child class.")

    def clear_fwd_cache(self):
        raise RuntimeError("This method should be instantiated in the child class.")

    def clear_bwd_cache(self):
        raise RuntimeError("This method should be instantiated in the child class.")

    def _build_rlxs(self):
        raise RuntimeError("This method should be instantiated in the child class.")

    def _fwdprop_symbnds(self):
        raise RuntimeError("This method should be instantiated in the child class.")

    def _bwdprop_symbnds(self):
        raise RuntimeError("This method should be instantiated in the child class.")

    def _cal_bnds(self):
        raise RuntimeError("This method should be instantiated in the child class.")

    @property
    def name(self):
        return self._name

    @property
    def cache(self) -> TCache:
        return self._cache

    @cache.setter
    def cache(self, value: TCache):
        self._cache = value

    @property
    def arguments(self) -> TArguments:
        return self._arguments

    @arguments.setter
    def arguments(self, value: TArguments):
        self._arguments = value

    @property
    def pre_nodes(self) -> list["TNode"]:
        return self._pre_nodes

    @pre_nodes.setter
    def pre_nodes(self, value: list["TNode"]):
        self._pre_nodes = value

    @property
    def next_nodes(self) -> list["TNode"]:
        return self._next_nodes

    @next_nodes.setter
    def next_nodes(self, value: list["TNode"]):
        self._next_nodes = value
