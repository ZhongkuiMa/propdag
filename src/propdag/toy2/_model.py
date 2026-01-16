__docformat__ = "restructuredtext"
__all__ = ["Toy2Model"]

from propdag.template2 import T2Model


class Toy2Model(T2Model):
    """
    Toy model demonstrating reversed graph semantics.

    Extends T2Model with verbose logging for educational purposes.
    Shows the execution flow of backward bound propagation through
    a reversed graph.

    Example::

        cache = Toy2Cache()
        args = Toy2Argument(verbose=True)

        node1 = Toy2Node("Input", cache, args)
        node2 = Toy2Node("Output", cache, args)

        # User builds forward graph
        node1.next_nodes = [node2]
        node2.pre_nodes = [node1]

        # Model reverses and runs
        model = Toy2Model([node1, node2], verbose=True)
        model.run()  # Prints: Propagate bounds through Output, then Input
    """

    def run(self, *args, **kwargs):
        """
        Execute backward bound propagation with verbose logging.

        Prints a header before running to make the output clearer.

        :param args: Positional arguments (unused)
        :param kwargs: Keyword arguments (unused)
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("Running Toy2Model (Reversed Graph Semantics)")
            print("=" * 60)
            print(f"Nodes in topological order: {[n.name for n in self.nodes]}")
            print("=" * 60 + "\n")

        super().run(*args, **kwargs)

        if self.verbose:
            print("\n" + "=" * 60)
            print("Toy2Model execution complete")
            print("=" * 60 + "\n")
