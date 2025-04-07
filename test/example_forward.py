from propdag import *

if __name__ == "__main__":
    node1 = ForwardToyNode("Node-1")
    node2 = ForwardToyNode("Node-2")
    node3 = ForwardToyNode("Node-3")
    node4 = ForwardToyNode("Node-4")
    node5 = ForwardToyNode("Node-5")
    node6 = ForwardToyNode("Node-6")
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

    model = ToyModel(nodes_list)

    cache = ToyCache()
    arguments = ToyArguments(prop_mode=PropMode.FORWARD)
    model.prepare(cache, arguments)
    model.run()
