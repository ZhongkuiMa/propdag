"""Cross-folder helpers shared by ``test_template`` and ``test_template2``.

Type-agnostic only: anything that references concrete backend classes
(``Toy*`` vs ``Toy2*``) belongs in the per-folder ``_helpers.py`` instead,
to avoid generic-factory abstraction that obscures the test setup.
"""

__docformat__ = "restructuredtext"


def verify_topological_order(sorted_nodes: list) -> None:
    """Assert every node appears after all of its predecessors in ``sorted_nodes``.

    Works on any node type that exposes ``name`` and ``pre_nodes`` attributes
    — both ``ForwardToyNode`` (template) and ``Toy2Node`` (template2) qualify.

    :param sorted_nodes: List of nodes in topological order.
    :raises AssertionError: If any predecessor appears after its successor.
    """
    position = {node.name: i for i, node in enumerate(sorted_nodes)}
    for node in sorted_nodes:
        for pre_node in node.pre_nodes:
            assert position[pre_node.name] < position[node.name], (
                f"{pre_node.name} should appear before {node.name}"
            )
