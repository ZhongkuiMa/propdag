__docformat__ = "restructuredtext"
__all__ = ["TCache"]

from abc import ABC
from dataclasses import dataclass


@dataclass(slots=True)
class TCache(ABC):
    pass
