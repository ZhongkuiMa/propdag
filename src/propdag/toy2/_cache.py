__docformat__ = "restructuredtext"
__all__ = ["Toy2Cache"]

from dataclasses import dataclass

from propdag.template2 import T2Cache


@dataclass(slots=True)
class Toy2Cache(T2Cache):
    """
    Toy cache implementation for reversed graph semantics.

    Inherits all fields from T2Cache:
    - cur_node: Currently executing node
    - bnds: Primary bounds (backward-propagated)
    - rlxs: Inverse relaxations
    - fwd_bnds: Forward bounds (for intersection)
    - symbnds: Symbolic bounds (if needed)

    This toy implementation uses the exact same cache structure as T2Cache
    without adding any additional fields. Real implementations might extend
    with additional caching for performance or debugging.
    """

    # No additional fields for toy implementation
