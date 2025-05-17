__docformat__ = "restructuredtext"
__all__ = ["PropMode"]

from enum import Enum, auto


class PropMode(Enum):
    """
    Enumeration of propagation modes.

    Defines the direction in which properties are propagated through a
    directed acyclic graph (DAG).

    :cvar FORWARD: Forward propagation mode - properties flow from inputs to outputs
    :cvar BACKWARD: Backward propagation mode - properties flow from outputs to inputs
    """

    FORWARD = auto()
    BACKWARD = auto()
