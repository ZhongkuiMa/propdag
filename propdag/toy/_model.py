__docformat__ = "restructured"
__all__ = ["ToyModel"]

from ._arguments import ToyArgument
from ._cache import ToyCache
from ..template import *


class ToyModel(TModel):
    _nodes: list[TNode]
    _cache: ToyCache
    _arguments: ToyArgument
    _all_backward_sorts: dict[TNode, list[TNode]]

    def run(self):
        print("Running ToyModel...")
        super().run()
