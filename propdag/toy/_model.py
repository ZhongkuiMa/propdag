__docformat__ = "restructuredtext"
__all__ = ["ToyModel"]

from propdag.template import TModel


class ToyModel(TModel):
    """
    A toy implementation of the template model.

    This class demonstrates a simple implementation of the TModel abstract class with
    toy versions of cache and arguments.

    :ivar _nodes: List of nodes in topological order
    :ivar _cache: Toy cache instance shared among all nodes
    :ivar _arguments: Toy arguments instance shared among all nodes
    :ivar _all_backward_sorts: Mapping of nodes to their backward topological sorts
    """

    def run(self):
        """
        Execute the toy model with visible logging.

        Prints a message before running the model and then delegates to the
        parent class implementation.
        """
        print("Running ToyModel...")
        super().run()
