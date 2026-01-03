__docformat__ = "restructuredtext"
__all__ = ["TArgument"]

from dataclasses import dataclass

from propdag.utils import PropMode


@dataclass(frozen=True, slots=True)
class TArgument:
    """
    Base class for computational graph arguments.

    This immutable class defines arguments that control the behavior
    of nodes in computational graphs.

    :ivar prop_mode: Propagation mode to use (forward, backward, etc.)
    """

    prop_mode: PropMode = PropMode.BACKWARD
