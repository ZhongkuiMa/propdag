__docformat__ = "restructuredtext"
__all__ = ["ToyArgument"]

from propdag.template import TArgument


class ToyArgument(TArgument):
    """
    Arguments class for toy models.

    A simple implementation of the TArgument abstract class that inherits the
    propagation mode property without adding additional parameters.
    """
