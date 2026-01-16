"""
Pytest configuration and fixtures for propdag tests.

Provides reusable fixtures for creating test DAGs with forward and backward
propagation modes, along with utility classes for cache tracking.
"""

import pytest

from propdag import (
    BackwardToyNode,
    ForwardToyNode,
    PropMode,
    ToyArgument,
    ToyCache,
)


class CacheTracker:
    """
    Utility class to track cache state after each node propagation.

    Records which cache keys (bounds, symbolic bounds) exist after each node
    completes its forward pass.
    """

    def __init__(self) -> None:
        """Initialize the tracker."""
        self.cache_history: list[dict[str, object]] = []

    def record(self, cache: ToyCache, node_name: str) -> None:
        """
        Record current cache state after a node executes.

        :param cache: The ToyCache instance
        :param node_name: Name of the node that just executed
        """
        state: dict[str, object] = {
            "node": node_name,
            "bnds_keys": set(cache.bnds.keys()),
            "symbnds_keys": set(cache.symbnds.keys()),
        }
        self.cache_history.append(state)

    def get_bounds_at_step(self, step_index: int) -> set[str]:
        """
        Get the set of keys in bnds after a specific step.

        :param step_index: Index of the step (0-based)
        :return: Set of bound keys at that step
        """
        if step_index < len(self.cache_history):
            return self.cache_history[step_index]["bnds_keys"]  # type: ignore[return-value]
        return set()

    def get_all_bounds_keys(self) -> set[str]:
        """Get all unique bound keys seen across all steps."""
        all_keys: set[str] = set()
        for state in self.cache_history:
            all_keys.update(state["bnds_keys"])  # type: ignore[arg-type]
        return all_keys


@pytest.fixture
def toy_cache():
    """Provide a fresh ToyCache instance for each test."""
    return ToyCache()


@pytest.fixture
def toy_arguments(request):
    """
    Provide ToyArgument instances for both forward and backward modes.

    Can be parametrized with PropMode to test different modes.
    """
    # Default to forward mode, but can be overridden
    prop_mode = getattr(request, "param", PropMode.FORWARD)
    return ToyArgument(prop_mode=prop_mode)


@pytest.fixture
def forward_arguments():
    """Provide forward propagation arguments."""
    return ToyArgument(prop_mode=PropMode.FORWARD)


@pytest.fixture
def backward_arguments():
    """Provide backward propagation arguments."""
    return ToyArgument(prop_mode=PropMode.BACKWARD)


@pytest.fixture
def cache_tracker():
    """Provide a cache tracking utility for tracking cache state during execution."""
    return CacheTracker()


@pytest.fixture
def instrumented_node_factory(toy_cache, forward_arguments, cache_tracker):
    """Create ToyNode factory with cache tracking instrumentation."""

    class InstrumentedForwardToyNode(ForwardToyNode):
        """ForwardToyNode that tracks cache state after execution."""

        def forward(self):
            """Execute forward pass and record cache state."""
            super().forward()
            cache_tracker.record(self.cache, self.name)

    class InstrumentedBackwardToyNode(BackwardToyNode):
        """BackwardToyNode that tracks cache state after execution."""

        def forward(self):
            """Execute forward pass and record cache state."""
            super().forward()
            cache_tracker.record(self.cache, self.name)

        def backward(self):
            """Execute backward pass and record cache state."""
            super().backward()
            cache_tracker.record(self.cache, self.name)

    def _create_node(name: str, node_class=None, prop_mode=PropMode.FORWARD):
        """
        Create a node with optional cache tracking.

        :param name: Node name
        :param node_class: Node class to use (auto-selected if None)
        :param prop_mode: Propagation mode (forward, backward, or backward_bound)
        :return: Configured node instance
        """
        arguments = ToyArgument(prop_mode=prop_mode)

        if node_class is None:
            if prop_mode == PropMode.FORWARD:
                node_class = InstrumentedForwardToyNode
            elif prop_mode == PropMode.BACKWARD:
                node_class = InstrumentedBackwardToyNode
            else:
                raise ValueError(f"Unknown prop_mode: {prop_mode}")

        return node_class(name, toy_cache, arguments)

    return _create_node


@pytest.fixture
def node_factory(toy_cache):
    """Create ToyNode factory without cache tracking."""

    def _create_node(name: str, prop_mode=PropMode.FORWARD):
        """
        Create a node.

        :param name: Node name
        :param prop_mode: Propagation mode (forward, backward, or backward_bound)
        :return: Configured node instance
        """
        arguments = ToyArgument(prop_mode=prop_mode)
        if prop_mode == PropMode.FORWARD:
            return ForwardToyNode(name, toy_cache, arguments)
        if prop_mode == PropMode.BACKWARD:
            return BackwardToyNode(name, toy_cache, arguments)
        raise ValueError(f"Unknown prop_mode: {prop_mode}")

    return _create_node
