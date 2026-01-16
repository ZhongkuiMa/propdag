"""Pytest configuration and fixtures for Template2/Toy2 (T2) tests."""

import pytest

from propdag import Toy2Argument, Toy2Cache, Toy2Node


class CacheTrackerT2:
    """
    Track cache state for T2 tests.

    Similar to CacheTracker but for T2Cache structure.
    """

    def __init__(self):
        """Initialize cache tracker with empty history."""
        self.cache_history = []

    def record(self, cache, node_name):
        """Record T2 cache state."""
        state = {
            "node": node_name,
            "bnds_keys": set(cache.bnds.keys()),
            "rlxs_keys": set(cache.rlxs.keys()),
            "fwd_bnds_keys": set(cache.fwd_bnds.keys()),
            "symbnds_keys": set(cache.symbnds.keys()),
        }
        self.cache_history.append(state)


@pytest.fixture
def toy2_cache():
    """Provide a fresh Toy2Cache instance for each test."""
    return Toy2Cache()


@pytest.fixture
def toy2_arguments():
    """Provide Toy2Argument instance."""
    return Toy2Argument()


@pytest.fixture
def cache_tracker_t2():
    """Provide cache tracker for T2 tests."""
    return CacheTrackerT2()


@pytest.fixture
def toy2_node_factory(toy2_cache, toy2_arguments):
    """
    Create Toy2Node instances.

    Usage:
        node = toy2_node_factory("NodeName")
    """

    def _create_node(name: str):
        return Toy2Node(name, toy2_cache, toy2_arguments)

    return _create_node
