__docformat__ = "restructuredtext"
__all__ = ["TArguments"]

from abc import ABC
from dataclasses import dataclass

from propdag.utils import PropMode


@dataclass(frozen=True, slots=True)
class TArguments(ABC):
    prop_mode: PropMode
