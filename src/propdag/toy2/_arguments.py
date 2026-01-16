__docformat__ = "restructuredtext"
__all__ = ["Toy2Argument"]

from dataclasses import dataclass

from propdag.template2 import T2Argument


@dataclass(frozen=True, slots=True)
class Toy2Argument(T2Argument):
    """
    Toy arguments for reversed graph models.

    Extends T2Argument with toy-specific parameters for demonstration purposes.

    :ivar verbose: Whether to enable verbose logging (defaults to False)
    """

    verbose: bool = False
