__docformat__ = "restructuredtext"
__all__ = ["TCache"]

from dataclasses import dataclass


@dataclass(slots=True)
class TCache:
    """
    Base class for caching in computational graphs.

    This class serves as a template for implementing caching functionality
    in computational graphs. Concrete implementations should define specific
    fields needed to cache computation results and intermediate values.
    """
