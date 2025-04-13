__docformat__ = "restructured"
__all__ = ["ToyModel"]

from ..template import *
from ._cache import ToyCache
from ._arguments import ToyArguments


class ToyModel(TModel):
    _nodes: list[TNode]
    _all_backward_sorts: dict[TNode, list[TNode]] | None
    _cache: ToyCache | None
    _arguments: ToyArguments | None

    def prepare(self, cache: ToyCache, arguments: ToyArguments):
        print("Preparing ToyModel...")
        super().prepare(cache, arguments)

    def run(self):
        print("Running ToyModel...")
        super().run()
