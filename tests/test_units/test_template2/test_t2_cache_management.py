"""
Test suite for cache management in T2 (reversed graph semantics).

T2 has simpler cache management than T1:
- No PropMode (single-purpose: backward propagation)
- Uses fwd_bnds (from forward pass) and bnds (backward propagated)
- Simpler clearing logic (not implemented in Toy2Node)
"""

import pytest

from propdag import Toy2Cache
from test_template2._helpers import build_chain_model_t2, build_diamond_model_t2


class TestCacheStructure:
    """T2Cache structure and fields."""

    def test_cache_has_required_fields(self):
        """T2Cache exposes the expected attribute slots."""
        cache = Toy2Cache()
        for field in ("cur_node", "bnds", "rlxs", "fwd_bnds", "symbnds"):
            assert hasattr(cache, field), f"Toy2Cache must expose {field}"

    def test_cache_fields_are_dicts(self):
        """The four mapping fields on T2Cache are dictionaries."""
        cache = Toy2Cache()
        for field in ("bnds", "rlxs", "fwd_bnds", "symbnds"):
            assert isinstance(getattr(cache, field), dict), f"{field} must be a dict"


class TestModelExecution:
    """T2Model executes without cache errors."""

    def test_simple_chain_executes(self):
        """A simple chain executes and populates bounds in the cache."""
        model, cache, _ = build_chain_model_t2(3)
        model.run()
        assert len(cache.bnds) > 0, "Execution must produce bounds"

    def test_diamond_executes(self):
        """A diamond DAG executes and populates bounds in the cache."""
        model, cache, _ = build_diamond_model_t2()
        model.run()
        assert len(cache.bnds) > 0, "Execution must produce bounds"


class TestCacheClearing:
    """Cache-clearing parameter behavior in T2."""

    @pytest.mark.parametrize("clear", [False, True])
    def test_clear_cache_during_running_parameter(self, clear):
        """The flag round-trips through model construction."""
        model, _, _ = build_chain_model_t2(2, clear_cache_during_running=clear)
        assert model.clear_cache_during_running is clear


class TestSharedCache:
    """All nodes share the same cache instance."""

    def test_all_nodes_share_cache(self):
        """``model.cache`` and every node's cache attribute are the same object."""
        model, cache, nodes = build_chain_model_t2(3)
        assert model.cache is cache
        for node in nodes:
            assert node.cache is cache


class TestForwardBounds:
    """Forward bounds initialization."""

    def test_seeded(self):
        """``cache.fwd_bnds`` retains its seed entry through model execution."""
        model, cache, _ = build_chain_model_t2(2)
        model.run()
        assert "Node-1" in cache.fwd_bnds
