__docformat__ = "restructuredtext"
__all__ = ["T2Argument"]

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class T2Argument:
    """
    Arguments for reversed graph models (template2).

    Template2 is single-purpose (backward bound propagation only), so there's
    no need for a propagation mode enum like in template/.

    **Simplified vs Template**:

    - template/TArgument has ``prop_mode`` field (FORWARD/BACKWARD)
    - template2/T2Argument has no mode field (always backward bound propagation)

    This dataclass is intentionally minimal. Future fields could include:
    - bound_tightening_iterations: int
    - intersection_strategy: Literal["min", "convex_hull"]
    - timeout: float
    - etc.

    For now, it serves as a type marker and provides a place for future
    backward-bound-specific parameters.
    """

    # No fields initially - may add bound propagation specific params later
