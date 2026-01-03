"""Utility types and enumerations for propdag."""

__docformat__ = "restructuredtext"
__all__ = ["PropMode"]

from enum import IntEnum, unique


@unique
class PropMode(IntEnum):
    """
    Propagation direction through computation graph.

    Defines how bounds propagate through the DAG:
    - FORWARD: Input bounds -> intermediate layers -> output bounds
    - BACKWARD: Output bounds + symbolic back-substitution -> tighter input bounds

    :cvar FORWARD: Forward propagation (inputs to outputs)
    :cvar BACKWARD: Backward propagation with substitution (outputs to inputs)
    """

    FORWARD = 1
    BACKWARD = 2
