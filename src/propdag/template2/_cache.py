__docformat__ = "restructuredtext"
__all__ = ["T2Cache"]

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from propdag.template2._node import T2Node


@dataclass(slots=True)
class T2Cache:
    """
    Cache for reversed graph (backward bound propagation).

    Template2 uses reversed graph semantics where "forward" propagation
    through the reversed graph achieves backward bound propagation. This
    cache structure reflects this simplified semantic model.

    **Simplified Naming vs Template**:

    In template/, we had confusing field names:
    - ``bnds``: forward bounds
    - ``back_bnds``: backward bounds
    - ``rlxs``: forward relaxations
    - ``inv_rlxs``: inverse relaxations

    In template2/, the graph is reversed so:
    - ``bnds``: primary bounds (backward-propagated, but "forward" in reversed graph)
    - ``rlxs``: primary relaxations (inverse relaxations, but "forward" in reversed graph)
    - ``fwd_bnds``: forward bounds from initial pass (for intersection)
    - ``symbnds``: symbolic bounds (if needed for bound calculation)

    :ivar cur_node: Reference to the currently active node during execution
    :ivar bnds: Primary bounds computed via backward propagation (dict: node_name -> bounds)
    :ivar rlxs: Inverse relaxations for backward propagation (dict: node_name -> relaxation)
    :ivar fwd_bnds: Forward bounds from initial forward pass (dict: node_name -> forward_bounds)
    :ivar symbnds: Symbolic bounds for bound calculation (dict: node_name -> symbolic_expression)
    """

    cur_node: "T2Node | None" = None
    bnds: dict[str, tuple] = field(default_factory=OrderedDict)
    rlxs: dict[str, tuple] = field(default_factory=OrderedDict)
    fwd_bnds: dict[str, tuple] = field(default_factory=OrderedDict)
    symbnds: dict[str, tuple] = field(default_factory=OrderedDict)
