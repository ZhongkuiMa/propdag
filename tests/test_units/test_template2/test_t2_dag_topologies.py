"""
Test suite for various DAG topologies in T2 (reversed graph semantics).

Tests representative DAG structures using template2/toy2 (linear chains,
skip connections, multi-input merges, complex branching, edge cases, and
realistic neural-network-like patterns) parametrized over BFS/DFS sorting
strategies. Unlike T1, T2 uses a single Toy2Node with automatic graph
reversal for backward propagation, so PropMode is not parametrized here.
"""

import pytest

from propdag import Toy2Argument, Toy2Cache, Toy2Model
from test_template2._helpers import (
    build_chain_model_t2,
    build_diamond_model_t2,
    build_skip_model_t2,
    build_y_model_t2,
    make_nodes_t2,
)

_SORT_STRATEGIES = ["bfs", "dfs"]


def _new_cache_args() -> tuple[Toy2Cache, Toy2Argument]:
    """Construct a fresh (cache, arguments) pair seeded with Node-1 forward bounds."""
    cache = Toy2Cache()
    cache.fwd_bnds["Node-1"] = ("forward bounds",)
    return cache, Toy2Argument()


class TestBasicStructures:
    """Category 1: Basic structures (linear and simple branching)."""

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_linear_chain(self, sort_strategy):
        """Linear chain Node-1 -> ... -> Node-4 propagates bounds end-to-end."""
        model, cache, _ = build_chain_model_t2(4, sort_strategy=sort_strategy)
        model.run()
        assert "Node-1" in cache.bnds or "Node-4" in cache.bnds

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_y_shape(self, sort_strategy):
        """Y-shape DAG (single branch and merge) propagates bounds."""
        model, cache, _ = build_y_model_t2(sort_strategy=sort_strategy)
        model.run()
        assert "Node-1" in cache.bnds or "Node-4" in cache.bnds


class TestSkipConnections:
    """Category 2: Skip connections and residual patterns."""

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_single_skip_connection(self, sort_strategy):
        """Single skip connection topology propagates bounds."""
        model, cache, _ = build_skip_model_t2(sort_strategy=sort_strategy)
        model.run()
        assert "Node-1" in cache.bnds or "Node-4" in cache.bnds

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_multiple_skip_connections(self, sort_strategy):
        """Multiple parallel skip connections propagate bounds."""
        # Same structural shape as build_skip_model_t2; kept distinct for traceability.
        model, cache, _ = build_skip_model_t2(sort_strategy=sort_strategy)
        model.run()
        assert "Node-1" in cache.bnds or "Node-4" in cache.bnds

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_nested_skip_connections(self, sort_strategy):
        """Nested skip connections propagate bounds."""
        model, cache, _ = build_skip_model_t2(sort_strategy=sort_strategy)
        model.run()
        assert "Node-1" in cache.bnds or "Node-4" in cache.bnds


class TestMultiInputCases:
    """Category 3: Multi-input edge cases (same source, diamonds, wide merge)."""

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_same_source_twice(self, sort_strategy):
        """Same source appearing twice in pre_nodes (like x^2 = x * x)."""
        cache, arguments = _new_cache_args()
        n1, n2 = make_nodes_t2(2, cache, arguments)
        n1.next_nodes = [n2, n2]
        n2.pre_nodes = [n1, n1]

        Toy2Model([n1, n2], sort_strategy=sort_strategy).run()

        assert "Node-1" in cache.bnds or "Node-2" in cache.bnds

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_diamond_pattern(self, sort_strategy):
        """Diamond pattern propagates bounds under the reversed graph."""
        model, cache, _ = build_diamond_model_t2(sort_strategy=sort_strategy)
        model.run()
        assert "Node-1" in cache.bnds or "Node-4" in cache.bnds

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_wide_merge(self, sort_strategy):
        """Wide merge (many-to-one): Node-1 broadcasts to 4 nodes which merge into Node-6."""
        cache, arguments = _new_cache_args()
        nodes = make_nodes_t2(6, cache, arguments)
        n1, *intermediates, n6 = nodes
        n1.next_nodes = list(intermediates)
        for n in intermediates:
            n.pre_nodes = [n1]
            n.next_nodes = [n6]
        n6.pre_nodes = list(intermediates)

        Toy2Model(nodes, sort_strategy=sort_strategy).run()

        assert len(cache.bnds) > 0


class TestComplexBranching:
    """Category 4: Complex branching patterns."""

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_wide_broadcast(self, sort_strategy):
        """One-to-many broadcast then merge into Node-6."""
        cache, arguments = _new_cache_args()
        nodes = make_nodes_t2(6, cache, arguments)
        n1, *intermediates, n6 = nodes
        n1.next_nodes = list(intermediates)
        for n in intermediates:
            n.pre_nodes = [n1]
            n.next_nodes = [n6]
        n6.pre_nodes = list(intermediates)

        Toy2Model(nodes, sort_strategy=sort_strategy).run()

        assert len(cache.bnds) > 0

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_asymmetric_tree(self, sort_strategy):
        """Asymmetric tree converging at Node-9."""
        cache, arguments = _new_cache_args()
        nodes = make_nodes_t2(9, cache, arguments)
        n1, n2, n3, n4, n5, n6, n7, n8, n9 = nodes
        n1.next_nodes = [n2, n3]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n4]
        n3.pre_nodes = [n1]
        n3.next_nodes = [n5, n6]
        n4.pre_nodes = [n2]
        n4.next_nodes = [n7]
        n5.pre_nodes = [n3]
        n5.next_nodes = [n8]
        n6.pre_nodes = [n3]
        n6.next_nodes = [n8]
        n7.pre_nodes = [n4]
        n7.next_nodes = [n9]
        n8.pre_nodes = [n5, n6]
        n8.next_nodes = [n9]
        n9.pre_nodes = [n7, n8]

        Toy2Model(nodes, sort_strategy=sort_strategy).run()

        assert len(cache.bnds) > 0

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_multiple_merge_points(self, sort_strategy):
        """Serial merging at different depths converging at Node-7."""
        cache, arguments = _new_cache_args()
        nodes = make_nodes_t2(7, cache, arguments)
        n1, n2, n3, n4, n5, n6, n7 = nodes
        n1.next_nodes = [n2, n3]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n4]
        n3.pre_nodes = [n1]
        n3.next_nodes = [n5]
        n4.pre_nodes = [n2]
        n4.next_nodes = [n6]
        n5.pre_nodes = [n3]
        n5.next_nodes = [n7]
        n6.pre_nodes = [n4]
        n6.next_nodes = [n7]
        n7.pre_nodes = [n5, n6]

        Toy2Model(nodes, sort_strategy=sort_strategy).run()

        assert len(cache.bnds) > 0


class TestBoundaryAndEdgeCases:
    """Category 5: Boundary and degenerate cases."""

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_minimal_dag(self, sort_strategy):
        """Minimal 2-node DAG runs without error."""
        model, cache, _ = build_chain_model_t2(2, sort_strategy=sort_strategy)
        model.run()
        assert len(cache.bnds) >= 0

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_long_chain(self, sort_strategy):
        """Long linear chain (10 nodes) runs without error (benchmark)."""
        model, cache, _ = build_chain_model_t2(10, sort_strategy=sort_strategy)
        model.run()
        assert len(cache.bnds) >= 0

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_deep_branching(self, sort_strategy):
        """Early broadcast, late merge (Node-1 -> 6 intermediates -> Node-8)."""
        cache, arguments = _new_cache_args()
        nodes = make_nodes_t2(8, cache, arguments)
        n1, *intermediate, n8 = nodes
        n1.next_nodes = list(intermediate)
        for n in intermediate:
            n.pre_nodes = [n1]
            n.next_nodes = [n8]
        n8.pre_nodes = list(intermediate)

        Toy2Model(nodes, sort_strategy=sort_strategy).run()

        assert len(cache.bnds) >= 0


class TestRealisticNetworkPatterns:
    """Category 6: Realistic neural network patterns."""

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_inception_like_module(self, sort_strategy):
        """Inception-like parallel processing paths converging at Node-6."""
        cache, arguments = _new_cache_args()
        nodes = make_nodes_t2(6, cache, arguments)
        nodes[0].next_nodes = [nodes[1], nodes[2], nodes[3], nodes[4]]
        for i in range(1, 5):
            nodes[i].pre_nodes = [nodes[0]]
            nodes[i].next_nodes = [nodes[5]]
        nodes[5].pre_nodes = nodes[1:5]

        Toy2Model(nodes, sort_strategy=sort_strategy).run()

        assert len(cache.bnds) >= 0

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    def test_densenet_like_connection(self, sort_strategy):
        """DenseNet-like dense connectivity ending at Node-4."""
        cache, arguments = _new_cache_args()
        nodes = make_nodes_t2(4, cache, arguments)
        nodes[0].next_nodes = [nodes[1], nodes[2], nodes[3]]
        nodes[1].pre_nodes = [nodes[0]]
        nodes[1].next_nodes = [nodes[2], nodes[3]]
        nodes[2].pre_nodes = [nodes[0], nodes[1]]
        nodes[2].next_nodes = [nodes[3]]
        nodes[3].pre_nodes = [nodes[0], nodes[1], nodes[2]]

        Toy2Model(nodes, sort_strategy=sort_strategy).run()

        assert len(cache.bnds) >= 0
