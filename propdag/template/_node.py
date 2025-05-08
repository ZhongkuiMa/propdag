__docformat__ = "restructuredtext"
__all__ = ["TNode"]

from abc import ABC

from ._arguments import TArgument
from ._cache import TCache


class TNode(ABC):
    _name: str
    _cache: TCache
    _argument: TArgument
    _pre_nodes: list["TNode"]
    _next_nodes: list["TNode"]

    def __init__(self, name: str, cache: TCache, argument: TArgument):
        self._name = name
        self._cache = cache
        self._argument = argument
        self._pre_nodes = []
        self._next_nodes = []

    def forward(self):
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def backward(self):
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def clear_fwd_cache(self):
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def clear_bwd_cache(self):
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _build_symbnds(self):
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _build_rlxs(self):
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _fwdprop_symbnds(self):
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _bwdprop_symbnds(self):
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _cal_bnds(self):
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

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
    def argument(self) -> TArgument:
        return self._argument

    @argument.setter
    def argument(self, value: TArgument):
        self._argument = value

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
