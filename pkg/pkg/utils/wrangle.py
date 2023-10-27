from datetime import datetime
from time import sleep

import numpy as np
import pandas as pd
from anytree import Node, PreOrderIter
from requests import HTTPError

import pcg_skel


def get_positions(nodelist, client, n_retries=1, retry_delay=5):
    nodelist = list(nodelist)
    l2stats = client.l2cache.get_l2data(nodelist, attributes=["rep_coord_nm"])
    nodes = pd.DataFrame(l2stats).T
    if "rep_coord_nm" not in nodes.columns:
        nodes["rep_coord_nm"] = np.nan
    positions = pt_to_xyz(nodes["rep_coord_nm"])
    nodes = pd.concat([nodes, positions], axis=1)
    nodes.index = nodes.index.astype(int)
    nodes.index.name = "l2_id"

    if nodes.isna().any().any() and n_retries != 0:
        print(
            f"Missing positions for some L2 nodes, retrying ({n_retries} attempts left)"
        )
        sleep(retry_delay)
        return get_positions(
            nodelist, client, n_retries=n_retries - 1, retry_delay=retry_delay
        )

    if nodes.isna().any().any():
        missing = nodes.loc[nodes.isna().any(axis=1)]
        raise HTTPError(
            f"Missing positions for some L2 nodes, for instance: {missing.index[:5].to_list()}"
        )

    return nodes


def get_level2_nodes_edges(root_id, client, positions=True):
    try:
        edgelist = client.chunkedgraph.level2_chunk_graph(root_id)
        nodelist = set()
        for edge in edgelist:
            for node in edge:
                nodelist.add(node)
        nodelist = list(nodelist)
    except HTTPError:
        # REF: https://github.com/seung-lab/PyChunkedGraph/issues/404
        nodelist = client.chunkedgraph.get_leaves(root_id, stop_layer=2)
        if len(nodelist) != 1:
            raise HTTPError(
                f"HTTPError: level 2 chunk graph not found for root_id: {root_id}"
            )
        else:
            edgelist = np.empty((0, 2), dtype=int)

    if positions:
        if positions == "lazy":
            nodes = get_positions(nodelist, client, n_retries=0)
        else:
            nodes = get_positions(nodelist, client)
    else:
        nodes = pd.DataFrame(index=nodelist)

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


def get_all_nodes_edges(root_ids, client, positions=False):
    all_nodes = []
    all_edges = []
    for root_id in root_ids:
        nodes, edges = get_level2_nodes_edges(root_id, client, positions=positions)
        all_nodes.append(nodes)
        all_edges.append(edges)
    all_nodes = pd.concat(all_nodes, axis=0)
    all_edges = pd.concat(all_edges, axis=0, ignore_index=True)
    return all_nodes, all_edges


