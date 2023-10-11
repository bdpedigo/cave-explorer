import pandas as pd
import pcg_skel
import numpy as np


def get_level2_nodes_edges(root_id, client):
    edgelist = client.chunkedgraph.level2_chunk_graph(root_id)

    nodelist = set()
    for edge in edgelist:
        for node in edge:
            nodelist.add(node)
    nodelist = list(nodelist)

    l2stats = client.l2cache.get_l2data(nodelist, attributes=["rep_coord_nm"])
    nodes = pd.DataFrame(l2stats).T
    positions = pt_to_xyz(nodes["rep_coord_nm"])
    nodes = pd.concat([nodes, positions], axis=1)
    nodes.index = nodes.index.astype(int)
    nodes.index.name = "l2_id"

    edges = pd.DataFrame(edgelist)
    edges.columns = ["source", "target"]

    edges = edges.drop_duplicates(keep="first")

    return nodes, edges


def get_skeleton_nodes_edges(root_id, client):
    final_meshwork = pcg_skel.coord_space_meshwork(
        root_id,
        client=client,
        # synapses="all",
        # synapse_table=client.materialize.synapse_table,
    )
    skeleton_nodes = pd.DataFrame(
        final_meshwork.skeleton.vertices,
        index=np.arange(len(final_meshwork.skeleton.vertices)),
        columns=["x", "y", "z"],
    )
    skeleton_edges = pd.DataFrame(
        final_meshwork.skeleton.edges, columns=["source", "target"]
    )
    return skeleton_nodes, skeleton_edges


def pt_to_xyz(pts):
    name = pts.name
    idx_name = pts.index.name
    if idx_name is None:
        idx_name = "index"
    positions = pts.explode().reset_index()

    def to_xyz(order):
        if order % 3 == 0:
            return "x"
        elif order % 3 == 1:
            return "y"
        else:
            return "z"

    positions["axis"] = positions.index.map(to_xyz)
    positions = positions.pivot(index=idx_name, columns="axis", values=name)

    return positions
