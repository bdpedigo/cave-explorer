from anytree import Node, PreOrderIter
import numpy as np


def create_lineage_tree(root_id, client, order="edits"):
    lineage = client.chunkedgraph.get_lineage_graph(root_id)
    links = lineage["links"]

    nodes = {}
    for link in links:
        source = link["source"]
        target = link["target"]
        if source not in nodes:
            nodes[source] = Node(source)
        if target not in nodes:
            nodes[target] = Node(target)

        # if nodes[source].parent is not None:
        #     raise ValueError(f"Node {source} has multiple parents!")

        nodes[source].parent = nodes[target]

    root = nodes[target].root

    for node in PreOrderIter(root):
        # make the left-hand side of the tree be whatever side has the most children
        if order == "edits":
            node._order_value = len(node.descendants)
        elif order == "l2_size":
            node._order_value = len(
                client.chunkedgraph.get_leaves(node.name, stop_layer=2)
            )
        else:
            raise ValueError(f"Unknown order: {order}")

    # reorder the children of each node by the values
    for node in PreOrderIter(root):
        children_order_value = np.array([child._order_value for child in node.children])
        inds = np.argsort(-children_order_value)
        node.children = [node.children[i] for i in inds]

    return root
