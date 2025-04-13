__docformat__ = "restructured"
__all__ = ["ToyModel"]

from ..template import *
from ._cache import ToyCache
from ._arguments import ToyArgument


class ToyModel(TModel):
    _nodes: list[TNode]
    _cache: ToyCache
    _arguments: ToyArgument
    _all_backward_sorts: dict[TNode, list[TNode]]

    def run(self):
        print("Running ToyModel...")
        super().run()
