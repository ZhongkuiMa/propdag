__docformat__ = "restructuredtext"
__all__ = ["TArgument"]

from abc import ABC
from dataclasses import dataclass

from ..utils import PropMode


@dataclass(frozen=True, slots=True)
class TArgument(ABC):
    """
    Abstract base class for computational graph arguments.

    This immutable class defines arguments that control the behavior
    of nodes in computational graphs.

    :ivar prop_mode: Propagation mode to use (forward, backward, etc.)
    :type prop_mode: PropMode
    """

    prop_mode: PropMode = PropMode.BACKWARD
