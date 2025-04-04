__docformat__ = "restructuredtext"
__all__ = ["TNode"]

from abc import ABC

from .arguments import TArguments
from .cache import TCache


class TNode(ABC):
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

    def clear_forward_cache(self):
        raise RuntimeError("This method should be instantiated in the child class.")

    def clear_backward_cache(self):
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
