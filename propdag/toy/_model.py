__docformat__ = "restructured"
__all__ = ["ToyModel"]

from ..template import *


class ToyModel(TModel):
    def prepare(self, cache: TCache, arguments: TArguments):
        print("Preparing ToyModel...")
        super().prepare(cache, arguments)

    def run(self):
        print("Running ToyModel...")
        super().run()
