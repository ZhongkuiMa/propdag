"""Type definitions for the propdag package."""

__docformat__ = "restructuredtext"
__all__ = [
    "CacheType",
    "ArgumentType",
    "NodeType",
]

from typing import TYPE_CHECKING, TypeAlias, TypeVar

from propdag.propdag.template._arguments import TArgument
from propdag.propdag.template._cache import TCache

if TYPE_CHECKING:
    from propdag.propdag.template._node import TNode

# Type variables for generic components
CacheType = TypeVar("CacheType", bound=TCache)
ArgumentType = TypeVar("ArgumentType", bound=TArgument)

# Type alias for nodes with specific cache and argument types
NodeType: TypeAlias = "TNode[CacheType, ArgumentType]"
