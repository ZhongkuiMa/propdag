"""
Test suite for various DAG topologies in propdag.

Tests 17 different directed acyclic graph structures including:
- Linear chains
- Skip connections (ResNet-like)
- Multi-input nodes (concatenation/merge)
- Complex branching patterns
- Edge cases and realistic neural network patterns

Each test is parametrized over both BFS/DFS sorting strategies and
forward/backward propagation modes, resulting in 68 total test cases
(17 topologies x 2 sort strategies x 2 propagation modes).
"""

import pytest

from propdag import PropMode, ToyArgument, ToyCache, ToyModel
from test_template._helpers import (
    build_chain_model,
    build_skip_model,
    build_y_model,
    make_nodes,
)

_SORT_STRATEGIES = ["bfs", "dfs"]
_PROP_MODES = [PropMode.FORWARD, PropMode.BACKWARD]


class TestBasicStructures:
    """Category 1: Basic structures (linear and simple branching)."""

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_linear_chain(self, sort_strategy, prop_mode):
        """Linear chain Node-1 -> Node-2 -> Node-3 -> Node-4 propagates bounds end-to-end."""
        model, cache, _ = build_chain_model(4, sort_strategy=sort_strategy, prop_mode=prop_mode)
        model.run()
        assert "Node-4" in cache.bnds, "Output node must have bounds"
        assert isinstance(cache.bnds["Node-4"], tuple), "Output bounds must be a tuple"
        assert len(cache.bnds["Node-4"]) > 0
        assert "Node-1" in cache.bnds, "Input node bounds must be preserved"

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_y_shape(self, sort_strategy, prop_mode):
        """Y-shape (Node-1 splits to Node-2 and Node-3, merge at Node-4) propagates bounds."""
        model, cache, _ = build_y_model(sort_strategy=sort_strategy, prop_mode=prop_mode)
        model.run()
        assert isinstance(cache.bnds["Node-4"], tuple), "Output bounds must be a tuple"
        assert "Node-1" in cache.bnds, "Input node bounds preserved"

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_sequential_branches(self, sort_strategy, prop_mode):
        r"""Late-stage branching and merging: Node-1 -> Node-2 -> {Node-3, Node-4}, Node-3 -> Node-4."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)
        n1, n2, n3, n4 = make_nodes(4, cache, arguments, prop_mode)
        n1.next_nodes = [n2]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n3, n4]
        n3.pre_nodes = [n2]
        n3.next_nodes = [n4]
        n4.pre_nodes = [n2, n3]

        ToyModel([n1, n2, n3, n4], sort_strategy=sort_strategy).run()

        assert isinstance(cache.bnds["Node-4"], tuple), "Output bounds must be a tuple"


class TestSkipConnections:
    """Category 2: Skip connections and residual patterns."""

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_single_skip_connection(self, sort_strategy, prop_mode):
        """Skip connection: Node-1 -> {Node-2, Node-4}; Node-2 -> Node-3 -> Node-4."""
        model, cache, _ = build_skip_model(sort_strategy=sort_strategy, prop_mode=prop_mode)
        model.run()
        assert isinstance(cache.bnds["Node-4"], tuple), "Output bounds must be a tuple"

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_multiple_skip_connections(self, sort_strategy, prop_mode):
        """Multiple parallel skip connections (ResNet-like) converging at Node-6."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)
        nodes = make_nodes(6, cache, arguments, prop_mode)
        n1, n2, n3, n4, n5, n6 = nodes
        n1.next_nodes = [n2, n5]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n3]
        n3.pre_nodes = [n2]
        n3.next_nodes = [n4, n6]
        n4.pre_nodes = [n3]
        n4.next_nodes = [n6]
        n5.pre_nodes = [n1]
        n5.next_nodes = [n6]
        n6.pre_nodes = [n3, n4, n5]

        ToyModel(nodes, sort_strategy=sort_strategy).run()

        assert isinstance(cache.bnds["Node-6"], tuple), "Output bounds must be a tuple"

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_nested_skip_connections(self, sort_strategy, prop_mode):
        """Overlapping skip connections from Node-1 and Node-2 onto Node-4."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)
        n1, n2, n3, n4 = make_nodes(4, cache, arguments, prop_mode)
        n1.next_nodes = [n2, n4]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n3, n4]
        n3.pre_nodes = [n2]
        n3.next_nodes = [n4]
        n4.pre_nodes = [n1, n2, n3]

        ToyModel([n1, n2, n3, n4], sort_strategy=sort_strategy).run()

        assert isinstance(cache.bnds["Node-4"], tuple), "Output bounds must be a tuple"


class TestMultiInputCases:
    """Category 3: Multi-input edge cases (same source to target, diamonds, etc.)."""

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_same_source_twice(self, sort_strategy, prop_mode):
        """Same source appearing twice in pre_nodes (like x^2 = x * x)."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)
        n1, n2, n3 = make_nodes(3, cache, arguments, prop_mode)
        n1.next_nodes = [n2]
        n2.pre_nodes = [n1, n1]  # Duplicate edge
        n2.next_nodes = [n3]
        n3.pre_nodes = [n2]

        ToyModel([n1, n2, n3], sort_strategy=sort_strategy).run()

        assert isinstance(cache.bnds["Node-3"], tuple), "Output bounds must be a tuple"

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_diamond_pattern(self, sort_strategy, prop_mode):
        """Standard diamond merge pattern with one extra Node-5 successor."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)
        nodes = make_nodes(5, cache, arguments, prop_mode)
        n1, n2, n3, n4, n5 = nodes
        n1.next_nodes = [n2, n3]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n4]
        n3.pre_nodes = [n1]
        n3.next_nodes = [n4]
        n4.pre_nodes = [n2, n3]
        n4.next_nodes = [n5]
        n5.pre_nodes = [n4]

        ToyModel(nodes, sort_strategy=sort_strategy).run()

        assert isinstance(cache.bnds["Node-5"], tuple), "Output bounds must be a tuple"

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_wide_merge(self, sort_strategy, prop_mode):
        """Wide merge (many-to-one): Node-1 broadcasts to 4 nodes which all merge into Node-6."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)
        nodes = make_nodes(6, cache, arguments, prop_mode)
        n1, *intermediates, n6 = nodes
        n1.next_nodes = list(intermediates)
        for n in intermediates:
            n.pre_nodes = [n1]
            n.next_nodes = [n6]
        n6.pre_nodes = list(intermediates)

        ToyModel(nodes, sort_strategy=sort_strategy).run()

        assert isinstance(cache.bnds["Node-6"], tuple), "Output bounds must be a tuple"


class TestComplexBranching:
    """Category 4: Complex branching patterns."""

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_wide_broadcast(self, sort_strategy, prop_mode):
        """One node broadcasting to many (one-to-many) and converging at Node-6."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)
        nodes = make_nodes(6, cache, arguments, prop_mode)
        n1, *intermediates, n6 = nodes
        n1.next_nodes = list(intermediates)
        for n in intermediates:
            n.pre_nodes = [n1]
            n.next_nodes = [n6]
        n6.pre_nodes = list(intermediates)

        ToyModel(nodes, sort_strategy=sort_strategy).run()

        assert isinstance(cache.bnds["Node-6"], tuple), "Output bounds must be a tuple"

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_asymmetric_tree(self, sort_strategy, prop_mode):
        """Unbalanced tree converging at Node-9."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)
        nodes = make_nodes(9, cache, arguments, prop_mode)
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

        ToyModel(nodes, sort_strategy=sort_strategy).run()

        assert isinstance(cache.bnds["Node-9"], tuple), "Output bounds must be a tuple"

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_multiple_merge_points(self, sort_strategy, prop_mode):
        """Serial merging at different depths with single output Node-7."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)
        nodes = make_nodes(7, cache, arguments, prop_mode)
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

        ToyModel(nodes, sort_strategy=sort_strategy).run()

        assert isinstance(cache.bnds["Node-7"], tuple), "Output bounds must be a tuple"


class TestBoundaryAndEdgeCases:
    """Category 5: Boundary and degenerate cases."""

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_minimal_dag(self, sort_strategy, prop_mode):
        """Minimal DAG with just 2 nodes preserves bounds for both."""
        model, cache, _ = build_chain_model(2, sort_strategy=sort_strategy, prop_mode=prop_mode)
        model.run()
        assert isinstance(cache.bnds["Node-2"], tuple), "Output bounds must be a tuple"
        assert isinstance(cache.bnds["Node-1"], tuple), "Input bounds must be preserved"

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_long_chain(self, sort_strategy, prop_mode):
        """Long linear chain of 15 nodes propagates bounds end-to-end (benchmark)."""
        model, cache, _ = build_chain_model(15, sort_strategy=sort_strategy, prop_mode=prop_mode)
        model.run()
        assert isinstance(cache.bnds["Node-15"], tuple), "Output bounds must be a tuple"

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_deep_branching(self, sort_strategy, prop_mode):
        """Early broadcast, late merge: Node-1 -> 6 intermediates -> Node-8."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)
        nodes = make_nodes(8, cache, arguments, prop_mode)
        n1, *intermediate, n8 = nodes
        n1.next_nodes = list(intermediate)
        for n in intermediate:
            n.pre_nodes = [n1]
            n.next_nodes = [n8]
        n8.pre_nodes = list(intermediate)

        ToyModel(nodes, sort_strategy=sort_strategy).run()

        assert isinstance(cache.bnds["Node-8"], tuple), "Output bounds must be a tuple"


class TestRealisticNetworkPatterns:
    """Category 6: Realistic neural network patterns."""

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_inception_like_module(self, sort_strategy, prop_mode):
        """Inception-like parallel processing paths converging at Node-10."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)
        nodes = make_nodes(10, cache, arguments, prop_mode)
        n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = nodes
        n1.next_nodes = [n2, n3, n4, n5]
        for src, mid in [(n2, n6), (n3, n7), (n4, n8), (n5, n9)]:
            src.pre_nodes = [n1]
            src.next_nodes = [mid]
            mid.pre_nodes = [src]
            mid.next_nodes = [n10]
        n10.pre_nodes = [n6, n7, n8, n9]

        ToyModel(nodes, sort_strategy=sort_strategy).run()

        assert isinstance(cache.bnds["Node-10"], tuple), "Output bounds must be a tuple"

    @pytest.mark.parametrize("sort_strategy", _SORT_STRATEGIES)
    @pytest.mark.parametrize("prop_mode", _PROP_MODES)
    def test_densenet_like_connection(self, sort_strategy, prop_mode):
        """DenseNet-like dense connectivity pattern: every node feeds the final Node-5."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=prop_mode)
        nodes = make_nodes(5, cache, arguments, prop_mode)
        n1, n2, n3, n4, n5 = nodes
        n1.next_nodes = [n2, n5]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n3, n5]
        n3.pre_nodes = [n2]
        n3.next_nodes = [n4, n5]
        n4.pre_nodes = [n3]
        n4.next_nodes = [n5]
        n5.pre_nodes = [n1, n2, n3, n4]

        ToyModel(nodes, sort_strategy=sort_strategy).run()

        assert isinstance(cache.bnds["Node-5"], tuple), "Output bounds must be a tuple"
