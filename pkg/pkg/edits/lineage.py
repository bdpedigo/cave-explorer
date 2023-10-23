from anytree import Node, PreOrderIter
import numpy as np
from datetime import datetime


def get_lineage_tree(
    root_id,
    client,
    flip=True,
    order=None,
    recurse=False,
    timestamps=False,
    labels=False,
):
    cg = client.chunkedgraph
    lineage_graph_dict = cg.get_lineage_graph(root_id)
    links = lineage_graph_dict["links"]

    node_map = {}
    for node in lineage_graph_dict["nodes"]:
        if "operation_id" not in node:
            operation_id = None
        else:
            operation_id = node["operation_id"]
        timestamp = node["timestamp"]
        timestamp = datetime.utcfromtimestamp(timestamp)
        timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        node_map[node["id"]] = {
            "timestamp": timestamp,
            "child_operation_id": operation_id,
        }

    lineage_nodes = {}
    for link in links:
        source = link["source"]
        target = link["target"]
        if source not in lineage_nodes:
            lineage_nodes[source] = Node(source, **node_map[source])
        if target not in lineage_nodes:
            lineage_nodes[target] = Node(target, **node_map[target])

        # flip means `root_id` is the root of this tree
        # not flip means `root_id` is the leaf of this tree, and parent is the one
        # or two objects that merged/split to form this leaf
        if flip:
            lineage_nodes[source].parent = lineage_nodes[target]
        else:
            lineage_nodes[target].parent = lineage_nodes[source]

    root = lineage_nodes[root_id].root

    for node in PreOrderIter(root):
        if labels:
            node.status = "internal"
        if timestamps:
            timestamp = cg.get_root_timestamps(node.name)[0]
            pretty_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            node.timestamp = pretty_time

    for node in root.leaves:
        if labels:
            node.status = "leaf"
        if labels or recurse:
            new_node = get_lineage_tree(
                node.name, client, flip=flip, order=order, recurse=False, labels=False
            )
            if not new_node.is_leaf:
                if labels:
                    node.status = "truncation"
                if recurse:
                    # may need to go deeper if this node is not a leaf
                    new_node = get_lineage_tree(
                        node.name,
                        client,
                        flip=flip,
                        order=order,
                        labels=labels,
                        recurse=True,
                    )
                    if flip:
                        # replace children of this node with the children of the new node
                        node.children = new_node.children
                    else:
                        # replace parent of this node with the parent of the new node
                        node.parent = new_node.parent

    if order is not None:
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
            children_order_value = np.array(
                [child._order_value for child in node.children]
            )
            inds = np.argsort(-children_order_value)
            node.children = [node.children[i] for i in inds]

    return lineage_nodes[root_id]
