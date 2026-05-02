"""Constants for the propdag package."""

__docformat__ = "restructuredtext"
__all__ = [
    "CYCLE_ERROR_MSG",
    "LOG_CACHE",
    "LOG_CLEAR",
    "LOG_COMPUTE",
    "LOG_INIT",
    "LOG_PROPAGATE",
    "LOG_RELAX",
    "SEPARATOR_LINE",
]

# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------

CYCLE_ERROR_MSG = "Graph has a cycle, cannot perform topological sort"

# ---------------------------------------------------------------------------
# Log prefixes for toy/toy2 verbose output
# ---------------------------------------------------------------------------

LOG_INIT = "[INIT]"
LOG_PROPAGATE = "[PROPAGATE]"
LOG_RELAX = "[RELAX]"
LOG_CACHE = "[CACHE]"
LOG_COMPUTE = "[COMPUTE]"
LOG_CLEAR = "[CLEAR]"

# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------

SEPARATOR_LINE = "=" * 60
