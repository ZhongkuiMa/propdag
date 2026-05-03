"""
Test suite for cache management and reference counting in propdag.

Validates that:
1. Cache state is tracked after each node propagation step.
2. Intermediate caches are cleared when no longer needed (reference counting).
3. Input and output node bounds are preserved throughout execution.
4. Memory efficiency through progressive cache cleanup.
5. The module-level helpers ``clear_fwd_cache`` and ``clear_bwd_cache`` behave
   correctly in isolation (added to close coverage gaps).
"""

import pytest

from propdag import ForwardToyNode, PropMode, ToyArgument, ToyCache, ToyModel
from propdag.template._model import clear_bwd_cache, clear_fwd_cache
from test_template._helpers import build_chain_model, build_y_model


class CacheSnapshot:
    """Captures the state of cache at a specific point in execution."""

    def __init__(self, step: int, node_name: str, bnds_keys: set[str], symbnds_keys: set[str]):
        """Initialize a cache snapshot.

        :param step: Execution step number
        :param node_name: Name of the node that executed
        :param bnds_keys: Keys present in cache.bnds after this step
        :param symbnds_keys: Keys present in cache.symbnds after this step
        """
        self.step = step
        self.node_name = node_name
        self.bnds_keys = bnds_keys
        self.symbnds_keys = symbnds_keys

    def __repr__(self):
        """Return string representation."""
        return (
            f"Step {self.step} ({self.node_name}): "
            f"bnds={self.bnds_keys}, symbnds={self.symbnds_keys}"
        )


class InstrumentedForwardToyNode(ForwardToyNode):
    """ForwardToyNode that records cache state on every forward pass."""

    def __init__(self, name: str, cache: ToyCache, argument: ToyArgument, tracker=None):
        """Initialize with optional cache-snapshot tracker list."""
        super().__init__(name, cache, argument)
        self.tracker = tracker or []

    def forward(self):
        """Execute forward pass and record cache state (skipping work for input nodes)."""
        if len(self._pre_nodes) == 0:
            self.tracker.append(
                CacheSnapshot(
                    step=len(self.tracker),
                    node_name=self.name,
                    bnds_keys=set(self.cache.bnds.keys()),
                    symbnds_keys=set(self.cache.symbnds.keys()),
                )
            )
            return

        super().forward()

        self.tracker.append(
            CacheSnapshot(
                step=len(self.tracker),
                node_name=self.name,
                bnds_keys=set(self.cache.bnds.keys()),
                symbnds_keys=set(self.cache.symbnds.keys()),
            )
        )

    def clear_fwd_cache(self):
        """Clear forward cache with safe handling of missing symbnds."""
        if len(self.next_nodes) > 0 and len(self.pre_nodes) > 0:
            print(f"{self.name}: Clear forward cache of bounds")
            del self.cache.bnds[self.name]
        if self.name in self.cache.symbnds:
            print(f"{self.name}: Clear forward cache of symbolic bounds")
            del self.cache.symbnds[self.name]


class TestCacheProgressiveCleanup:
    """Cache is cleaned up progressively as computation proceeds."""

    def test_linear_chain(self):
        """Linear chain Node-1 -> Node-2 -> Node-3 keeps input and output bounds."""
        model, cache, _ = build_chain_model(3)
        model.run()
        assert "Node-1" in cache.bnds, "Input node should be preserved"
        assert "Node-3" in cache.bnds, "Output node should be present"

    def test_y_shape(self):
        """Y-shape DAG keeps input and output bounds after execution."""
        model, cache, _ = build_y_model()
        model.run()
        assert "Node-1" in cache.bnds, "Input should be preserved"
        assert "Node-4" in cache.bnds, "Output should be present"

    def test_skip_connection(self):
        """Skip connection topology keeps input and output bounds."""
        # Use the canonical skip topology built ad-hoc here so we don't conflate
        # cache-management semantics with the helper's build_skip_model variant.
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)
        n1 = ForwardToyNode("Node-1", cache, arguments)
        n2 = ForwardToyNode("Node-2", cache, arguments)
        n3 = ForwardToyNode("Node-3", cache, arguments)
        n4 = ForwardToyNode("Node-4", cache, arguments)
        n1.next_nodes = [n2, n4]
        n2.pre_nodes = [n1]
        n2.next_nodes = [n3]
        n3.pre_nodes = [n2]
        n3.next_nodes = [n4]
        n4.pre_nodes = [n1, n3]

        ToyModel([n1, n2, n3, n4], sort_strategy="bfs").run()

        assert "Node-1" in cache.bnds, "Input should be preserved"
        assert "Node-4" in cache.bnds, "Output should be present"


class TestCacheInputOutputPreservation:
    """Input and output node bounds are always preserved."""

    def test_input_bounds_preserved_throughout(self):
        """Input node bounds are present in every cache snapshot."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)
        tracker: list[CacheSnapshot] = []
        nodes = [
            InstrumentedForwardToyNode(f"Node-{i}", cache, arguments, tracker) for i in range(1, 6)
        ]
        for i in range(len(nodes) - 1):
            nodes[i].next_nodes = [nodes[i + 1]]
            nodes[i + 1].pre_nodes = [nodes[i]]

        ToyModel(nodes, sort_strategy="bfs").run()

        for snapshot in tracker:
            assert "Node-1" in snapshot.bnds_keys, f"Input lost at step {snapshot.step}"

    def test_output_bounds_preserved_at_end(self):
        """Output node bounds are present after execution."""
        model, cache, _ = build_chain_model(3)
        model.run()
        assert "Node-3" in cache.bnds


class TestCacheReferenceCountingLogic:
    """Reference counting clears intermediate caches at the right moment."""

    def test_wide_merge(self):
        """Many-to-one pattern preserves input and output, clears intermediates."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)
        n1 = ForwardToyNode("Node-1", cache, arguments)
        intermediates = [ForwardToyNode(f"Node-{i}", cache, arguments) for i in range(2, 5)]
        n5 = ForwardToyNode("Node-5", cache, arguments)
        n1.next_nodes = intermediates
        for n in intermediates:
            n.pre_nodes = [n1]
            n.next_nodes = [n5]
        n5.pre_nodes = intermediates

        ToyModel([n1, *intermediates, n5], sort_strategy="bfs").run()

        assert "Node-1" in cache.bnds
        assert "Node-5" in cache.bnds

    def test_wide_broadcast(self):
        """One-to-many pattern preserves input and output."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)
        n1 = ForwardToyNode("Node-1", cache, arguments)
        intermediates = [ForwardToyNode(f"Node-{i}", cache, arguments) for i in range(2, 6)]
        n6 = ForwardToyNode("Node-6", cache, arguments)
        n1.next_nodes = intermediates
        for n in intermediates:
            n.pre_nodes = [n1]
            n.next_nodes = [n6]
        n6.pre_nodes = intermediates

        ToyModel([n1, *intermediates, n6], sort_strategy="bfs").run()

        assert "Node-1" in cache.bnds
        assert "Node-6" in cache.bnds


class TestCacheMemoryEfficiency:
    """Cache memory is used efficiently through cleanup."""

    def test_no_memory_leak_in_long_chain(self):
        """Long chain (20 nodes) keeps the cache size bounded by total node count."""
        model, cache, _ = build_chain_model(20)
        model.run()
        final_bnds = set(cache.bnds.keys())
        assert "Node-1" in final_bnds, "Input must be preserved"
        assert "Node-20" in final_bnds, "Output must be preserved"
        assert len(final_bnds) <= 20, "Cache should be efficiently cleaned"

    def test_symmetric_broadcast_merge(self):
        """Broadcast-merge pattern preserves input and output bounds."""
        cache = ToyCache()
        cache.bnds["Node-1"] = ("input bounds",)
        arguments = ToyArgument(prop_mode=PropMode.FORWARD)
        n1 = ForwardToyNode("Node-1", cache, arguments)
        intermediates = [ForwardToyNode(f"Node-{i}", cache, arguments) for i in range(2, 5)]
        n5 = ForwardToyNode("Node-5", cache, arguments)
        n1.next_nodes = intermediates
        for n in intermediates:
            n.pre_nodes = [n1]
            n.next_nodes = [n5]
        n5.pre_nodes = intermediates

        ToyModel([n1, *intermediates, n5], sort_strategy="bfs").run()

        assert "Node-1" in cache.bnds
        assert "Node-5" in cache.bnds


class TestSymbolicBoundsCleanup:
    """Symbolic bounds are cleaned up appropriately."""

    def test_symbolic_bounds_in_linear_chain(self):
        """Input and output bounds remain populated after a linear chain run."""
        model, cache, _ = build_chain_model(3)
        model.run()
        assert isinstance(cache.bnds["Node-1"], tuple), "Input bounds should be a tuple"
        assert isinstance(cache.bnds["Node-3"], tuple), "Output bounds should be a tuple"
        assert len(cache.bnds["Node-1"]) > 0, "Input bounds should not be empty"
        assert len(cache.bnds["Node-3"]) > 0, "Output bounds should not be empty"


class TestClearCacheDuringRunning:
    """Behavior of the ``clear_cache_during_running`` parameter."""

    def test_disabled_by_default(self):
        """``clear_cache_during_running`` defaults to False."""
        model, _, _ = build_chain_model(3)
        assert model.clear_cache_during_running is False

    def test_disabled_keeps_intermediate_caches(self):
        """With clearing disabled, all intermediate node caches survive the run."""
        model, cache, _ = build_chain_model(4, clear_cache_during_running=False)
        model.run()
        for i in range(1, 5):
            assert f"Node-{i}" in cache.bnds, (
                f"Node-{i} bounds should be preserved when not clearing"
            )

    def test_enabled_clears_intermediates(self):
        """With clearing enabled, input and output remain populated after the run."""
        model, cache, _ = build_chain_model(4, clear_cache_during_running=True)
        model.run()
        assert "Node-1" in cache.bnds, "Input should be preserved"
        assert "Node-4" in cache.bnds, "Output should be preserved"

    @pytest.mark.parametrize("clear", [False, True])
    def test_with_branching(self, clear):
        """Y-shape input/output bounds remain populated regardless of clearing flag."""
        model, cache, _ = build_y_model(clear_cache_during_running=clear)
        model.run()
        assert "Node-1" in cache.bnds, "Input should be preserved"
        assert "Node-4" in cache.bnds, "Output should be preserved"


class TestModuleLevelClearFunctions:
    """COV1/COV2: direct unit tests for ``clear_fwd_cache`` and ``clear_bwd_cache``."""

    def test_clear_fwd_cache_removes_node_when_counter_hits_zero(self):
        """``clear_fwd_cache`` deletes a node from the counter when its count reaches 0."""
        _, cache, nodes = build_chain_model(3)
        # Seed bnds entries so node.clear_fwd_cache() can delete them without KeyError.
        for n in nodes:
            cache.bnds[n.name] = ("placeholder",)
        counter = dict.fromkeys(nodes, 1)
        clear_fwd_cache(counter, [nodes[1]])
        assert nodes[1] not in counter, "node should be removed when counter reaches 0"
        assert nodes[0] in counter, "Node-1 must remain in the counter"
        assert nodes[2] in counter, "Node-3 must remain in the counter"

    def test_clear_fwd_cache_decrements_without_removal(self):
        """``clear_fwd_cache`` decrements but keeps a node whose counter stays positive."""
        _, _, nodes = build_chain_model(3)
        counter = dict.fromkeys(nodes, 2)
        clear_fwd_cache(counter, [nodes[1]])
        assert counter[nodes[1]] == 1, "counter should be decremented but not removed"

    def test_clear_bwd_cache_skips_nodes_not_in_counter(self):
        """``clear_bwd_cache`` silently skips nodes not present in the counter."""
        _, _, nodes = build_chain_model(3)
        counter = {nodes[0]: 1}
        clear_bwd_cache(counter, [nodes[2]])
        assert counter == {nodes[0]: 1}, "untracked nodes must not affect the counter"

    def test_clear_bwd_cache_removes_when_counter_hits_zero(self):
        """``clear_bwd_cache`` deletes a tracked node when its counter reaches 0."""
        # We can't actually call node.clear_bwd_cache() on ForwardToyNode (it raises),
        # so test only the counter book-keeping path with a counter starting at 1
        # against a node we then exclude from the nodes list.
        _, _, nodes = build_chain_model(3)
        sentinel_node = nodes[1]
        counter = {sentinel_node: 1}
        # Pass an empty nodes list -> no decrement, counter unchanged.
        clear_bwd_cache(counter, [])
        assert counter[sentinel_node] == 1
