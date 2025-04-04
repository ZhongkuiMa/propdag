from propdag import *

if __name__ == "__main__":
    node1 = ToyNode("Node-1")
    node2 = ToyNode("Node-2")
    node3 = ToyNode("Node-3")
    node4 = ToyNode("Node-4")
    node5 = ToyNode("Node-5")
    node6 = ToyNode("Node-6")
    node1.next_nodes = [node2, node3]
    node2.pre_nodes = [node1]
    node2.next_nodes = [node4]
    node3.pre_nodes = [node1]
    node3.next_nodes = [node4]
    node4.pre_nodes = [node2, node3]
    node4.next_nodes = [node5]
    node5.pre_nodes = [node4]
    node5.next_nodes = [node6]
    node6.pre_nodes = [node5]
    nodes_list = [node1, node2, node3, node4, node5, node6]
    model = ToyModel(nodes_list)

    cache = ToyCache()
    arguments = ToyArguments(prop_mode=PropMode.FORWARD)
    model.prepare(cache, arguments)
    model.run()

    cache = ToyCache()
    arguments = ToyArguments(prop_mode=PropMode.BACKWARD)
    model.prepare(cache, arguments)
    model.run()
