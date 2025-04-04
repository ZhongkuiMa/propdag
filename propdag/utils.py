__docformat__ = "restructuredtext"
__all__ = ["PropMode"]

from enum import Enum, auto


class PropMode(Enum):
    FORWARD = auto()
    BACKWARD = auto()
