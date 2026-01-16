__docformat__ = "restructuredtext"
__all__ = ["ToyCache"]

from collections import OrderedDict
from dataclasses import dataclass, field

from propdag.template import TCache, TNode


@dataclass(slots=True)
class ToyCache(TCache):
    """
    Cache implementation for the toy model.

    Stores computation results including symbolic bounds, scalar bounds,
    and relaxations for nodes in a toy model.

    :ivar cur_node: Reference to the currently active node
    :ivar symbnds: Mapping from node names to their symbolic bounds
    :ivar bnds: Mapping from node names to their scalar bounds
    :ivar rlxs: Mapping from node names to their forward relaxations
    """

    cur_node: TNode | None = None
    symbnds: dict[str, tuple] = field(default_factory=OrderedDict)
    bnds: dict[str, tuple] = field(default_factory=OrderedDict)
    rlxs: dict[str, tuple] = field(default_factory=OrderedDict)
