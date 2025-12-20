"""
Example script demonstrating forward propagation.

This script creates a directed acyclic graph (DAG) with six nodes
and runs a toy model with forward propagation mode to show how
properties flow through the graph from input to output.
"""

from propdag import BackwardToyNode, PropMode, ToyArgument, ToyCache, ToyModel

if __name__ == "__main__":
    cache = ToyCache()
    cache.bnds["Node-1"] = ("input bounds",)
    arguments = ToyArgument(prop_mode=PropMode.BACKWARD)
    node1 = BackwardToyNode("Node-1", cache, arguments)
    node2 = BackwardToyNode("Node-2", cache, arguments)
    node3 = BackwardToyNode("Node-3", cache, arguments)
    node4 = BackwardToyNode("Node-4", cache, arguments)
    node5 = BackwardToyNode("Node-5", cache, arguments)
    node6 = BackwardToyNode("Node-6", cache, arguments)
    node1.next_nodes = [node2, node3]
    node2.pre_nodes = [node1]
    node2.next_nodes = [node4]
    node3.pre_nodes = [node1]
    node3.next_nodes = [node4, node5]
    node4.pre_nodes = [node2, node3]
    node4.next_nodes = [node6]
    node5.pre_nodes = [node3]
    node5.next_nodes = [node6]
    node6.pre_nodes = [node4, node5]
    nodes_list = [node1, node2, node3, node4, node5, node6]
    # The DAG is:
    #
    #     Node-1
    #     /    \
    #  Node-2  Node-3
    #     \    /    \
    #     Node-4    Node-5
    #         \    /
    #         Node-6
    #
    dag_str = """
    The DAG is:

        Node-1
        /    \\
     Node-2  Node-3
        \\    /    \\
        Node-4    Node-5
            \\    /
            Node-6
    """
    print(dag_str)

    model = ToyModel(nodes_list, verbose=True)
    model.run()
