__docformat__ = "restructured"
__all__ = ["ToyModel"]

from ._arguments import ToyArgument
from ._cache import ToyCache
from ..template import *


class ToyModel(TModel):
    """
    A toy implementation of the template model.

    This class demonstrates a simple implementation of the TModel abstract class with
    toy versions of cache and arguments.

    :ivar _nodes: List of nodes in topological order
    :type _nodes: list[TNode]
    :ivar _cache: Toy cache instance shared among all nodes
    :type _cache: ToyCache
    :ivar _arguments: Toy arguments instance shared among all nodes
    :type _arguments: ToyArgument
    :ivar _all_backward_sorts: Mapping of nodes to their backward topological sorts
    :type _all_backward_sorts: dict[TNode, list[TNode]]
    """

    _nodes: list[TNode]
    _cache: ToyCache
    _arguments: ToyArgument
    _all_backward_sorts: dict[TNode, list[TNode]]

    def run(self):
        """
        Execute the toy model with visible logging.

        Prints a message before running the model and then delegates to the
        parent class implementation.
        """
        print("Running ToyModel...")
        super().run()
