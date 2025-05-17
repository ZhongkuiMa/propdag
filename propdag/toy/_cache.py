__docformat__ = "restructuredtext"
__all__ = ["ToyCache"]

from collections import OrderedDict
from dataclasses import dataclass, field

from ..template import *


@dataclass(slots=True)
class ToyCache(TCache):
    """
    Cache implementation for the toy model.

    Stores computation results including symbolic bounds, scalar bounds,
    and relaxations for the nodes in a toy model.

    :ivar cur_node: Reference to the currently active node
    :type cur_node: TNode | None
    :ivar symbnds: Mapping from node names to their symbolic bounds
    :type symbnds: dict[str, tuple]
    :ivar bnds: Mapping from node names to their scalar bounds
    :type bnds: dict[str, tuple]
    :ivar rlxs: Mapping from node names to their relaxations
    :type rlxs: dict[str, tuple]
    """

    cur_node: TNode | None = None
    symbnds: dict[str, tuple] = field(default_factory=OrderedDict)
    bnds: dict[str, tuple] = field(default_factory=OrderedDict)
    rlxs: dict[str, tuple] = field(default_factory=OrderedDict)
