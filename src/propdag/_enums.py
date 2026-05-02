"""Enumerations for propdag."""

__docformat__ = "restructuredtext"
__all__ = ["PropMode"]

from enum import IntEnum, unique


@unique
class PropMode(IntEnum):
    """Propagation direction through computation graph."""

    FORWARD = 1
    """Forward propagation from inputs to outputs."""
    BACKWARD = 2
    """Backward propagation with substitution from outputs to inputs."""

    def __repr__(self):
        """Return string representation."""
        return f"{self.name}"
