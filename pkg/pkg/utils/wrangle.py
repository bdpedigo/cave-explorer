import pickle
from time import sleep

import numpy as np
import pandas as pd
import pcg_skel
from caveclient import CAVEclient
from requests import HTTPError
from sklearn.metrics import pairwise_distances_argmin

from pkg.paths import DATA_PATH


def get_positions(nodelist, client: CAVEclient, n_retries=2, retry_delay=10):
    nodelist = list(nodelist)
    chunk_size = 100_000
    if len(nodelist) > chunk_size:
        print(
            f"Warning: nodelist is too large ({len(nodelist)}), splitting into chunks of {chunk_size}"
        )
        chunks = [
            nodelist[i : i + chunk_size] for i in range(0, len(nodelist), chunk_size)
        ]
        nodes = []
        for chunk in chunks:
            nodes.append(
                get_positions(
                    chunk, client, n_retries=n_retries, retry_delay=retry_delay
                )
            )
        nodes = pd.concat(nodes, axis=0)
        return nodes
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


def get_level2_nodes_edges(root_id: int, client: CAVEclient, positions=True):
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


def get_skeleton_nodes_edges(root_id: int, client: CAVEclient):
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
    # name = pts.name
    # idx_name = pts.index.name
    # if idx_name is None:
    #     idx_name = "index"
    # positions = pts.explode().reset_index()

    # def to_xyz(order):
    #     if order % 3 == 0:
    #         return "x"
    #     elif order % 3 == 1:
    #         return "y"
    #     else:
    #         return "z"

    # positions["axis"] = positions.index.map(to_xyz)
    # positions = positions.pivot(index=idx_name, columns="axis", values=name)

    positions = pd.DataFrame(index=pts.index)

    positions["x"] = pts.apply(lambda x: x[0])
    positions["y"] = pts.apply(lambda x: x[1])
    positions["z"] = pts.apply(lambda x: x[2])

    return positions


def get_all_nodes_edges(root_ids, client: CAVEclient, positions=False):
    all_nodes = []
    all_edges = []
    for root_id in root_ids:
        nodes, edges = get_level2_nodes_edges(root_id, client, positions=positions)
        all_nodes.append(nodes)
        all_edges.append(edges)
    all_nodes = pd.concat(all_nodes, axis=0)
    all_edges = pd.concat(all_edges, axis=0, ignore_index=True)
    return all_nodes, all_edges


def integerize_dict_keys(dictionary):
    return {int(k): v for k, v in dictionary.items()}


def stringize_dict_keys(dictionary):
    return {str(k): v for k, v in dictionary.items()}


def get_nucleus_level2_id(root_id: int, client: CAVEclient):
    nuc = client.materialize.query_table(
        "nucleus_detection_v0",
        filter_equal_dict={"pt_root_id": root_id},
        select_columns=["pt_supervoxel_id", "pt_root_id", "pt_position"],
    ).set_index("pt_root_id")
    nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]
    current_nuc_level2 = client.chunkedgraph.get_roots([nuc_supervoxel], stop_layer=2)[
        0
    ]
    return current_nuc_level2


def get_nucleus_point_nm(root_id: int, client: CAVEclient, method="table"):
    nuc = client.materialize.query_table(
        "nucleus_detection_v0",
        filter_equal_dict={"pt_root_id": root_id},
        select_columns=["pt_supervoxel_id", "pt_root_id", "pt_position"],
    ).set_index("pt_root_id")
    if method == "l2cache":
        nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]
        current_nuc_level2 = client.chunkedgraph.get_roots(
            [nuc_supervoxel], stop_layer=2
        )[0]
        nuc_pt_nm = client.l2cache.get_l2data(
            [current_nuc_level2], attributes=["rep_coord_nm"]
        )[str(current_nuc_level2)]["rep_coord_nm"]
        nuc_pt_nm = np.array(nuc_pt_nm)
    elif method == "table":
        nuc_pt_nm = np.array(nuc.loc[root_id, "pt_position"])
        nuc_pt_nm *= np.array([4, 4, 40])
    return nuc_pt_nm


def find_closest_point(df, point):
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    X = df.loc[:, ["x", "y", "z"]].values
    min_iloc = pairwise_distances_argmin(point.reshape(1, -1), X)[0]
    return df.index[min_iloc]


def load_casey_palette():
    palette_file = DATA_PATH / "ctype_hues.pkl"

    with open(palette_file, "rb") as f:
        ctype_hues = pickle.load(f)

    ctype_hues = {ctype: tuple(ctype_hues[ctype]) for ctype in ctype_hues.keys()}
    return ctype_hues


def load_mtypes(client: CAVEclient):
    mtypes = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")
    root_id_counts = mtypes["pt_root_id"].value_counts()
    root_id_singles = root_id_counts[root_id_counts == 1].index
    mtypes = mtypes.query("pt_root_id in @root_id_singles")
    mtypes.set_index("pt_root_id", inplace=True)
    return mtypes
