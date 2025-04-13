__docformat__ = "restructuredtext"
__all__ = ["TArgument"]

from abc import ABC
from dataclasses import dataclass

from ..utils import PropMode


@dataclass(frozen=True, slots=True)
class TArgument(ABC):
    prop_mode: PropMode
