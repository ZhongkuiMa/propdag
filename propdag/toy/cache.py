__docformat__ = "restructuredtext"
__all__ = ["ToyCache"]

from collections import OrderedDict
from dataclasses import dataclass, field

from propdag.template import *


@dataclass(slots=True)
class ToyCache(TCache):
    cur_node: TNode | None = None
    symbnds: dict[str, tuple] = field(default_factory=OrderedDict)
    bnds: dict[str, tuple] = field(default_factory=OrderedDict)
    rlxs: dict[str, tuple] = field(default_factory=OrderedDict)
