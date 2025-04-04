__docformat__ = "restructured"
__all__ = ["ToyModel"]

from propdag import TNode
from propdag.template import TModel, TCache, TArguments


class ToyModel(TModel):
    def prepare(self, cache: TCache, arguments: TArguments):
        print("Preparing...")
        super().prepare(cache, arguments)

    def run(self):
        print("Running...")
        super().run()

    def _backsub(self, node: TNode):
        print("Backsub...")
        super()._backsub(node)
