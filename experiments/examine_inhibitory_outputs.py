# %%
import os

import caveclient as cc
from neuropull.graph import NetworkFrame
from tqdm.autonotebook import tqdm

from pkg.edits import (
    apply_edit,
    find_supervoxel_component,
    lazy_load_initial_network,
    lazy_load_network_edits,
)
from pkg.utils import get_level2_nodes_edges
import pandas as pd


def get_environment_variables():
    cloud = os.environ.get("SKEDITS_USE_CLOUD") == "True"
    recompute = os.environ.get("SKEDITS_RECOMPUTE") == "True"
    return cloud, recompute


get_environment_variables()
# %%


os.environ["SKEDITS_USE_CLOUD"] = "False"
os.environ["SKEDITS_RECOMPUTE"] = "False"
# from pkg.workers import extract_edit_info
client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")

nuc = client.materialize.query_table(
    "nucleus_detection_v0",  # select_columns=["pt_supervoxel_id", "pt_root_id"]
).set_index("pt_root_id")

# %%

root_id = query_neurons["pt_root_id"].iloc[2]

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

nf = lazy_load_initial_network(root_id, client=client)

networkdeltas_by_operation, networkdeltas_by_metaoperation = lazy_load_network_edits(
    root_id, client=client
)

for edit in tqdm(networkdeltas_by_operation.values()):
    apply_edit(nf, edit)

nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]

nuc_nf = find_supervoxel_component(nuc_supervoxel, nf, client)


nodes, edges = get_level2_nodes_edges(root_id, client=client)
final_nf = NetworkFrame(nodes, edges)

final_nf.nodes.drop(columns=["rep_coord_nm", "x", "y", "z"], inplace=True)
nuc_nf.nodes.drop(columns=["rep_coord_nm", "x", "y", "z"], inplace=True)

assert nuc_nf == final_nf


# %%


def find_relevant_merges(networkdeltas_by_metaoperation, final_nf):
    merge_metaedit_pool = []
    for metaoperation_id, networkdelta in networkdeltas_by_metaoperation.items():
        operation_ids = networkdelta.metadata["operation_ids"]
        is_merges = []
        for operation_id in operation_ids:
            is_merges.append(
                networkdeltas_by_operation[operation_id].metadata["is_merge"]
            )
        any_is_merges = any(is_merges)
        is_relevant = networkdelta.added_nodes.index.isin(final_nf.nodes.index).any()
        if any_is_merges and is_relevant:
            merge_metaedit_pool.append(metaoperation_id)
    
    merge_metaedit_pool = pd.Series(merge_metaedit_pool)



# %%

from requests.exceptions import HTTPError


def get_latest_network(root_id, client, positions=False, verbose=True):
    original_node_ids = client.chunkedgraph.get_original_roots(root_id)
    latest_node_ids = client.chunkedgraph.get_latest_roots(original_node_ids)

    all_nodes = []
    all_edges = []
    had_error = False
    for leaf_id in tqdm(
        latest_node_ids,
        desc="Finding L2 graphs for latest segmentation objects",
        disable=not verbose,
    ):
        try:
            nodes, edges = get_level2_nodes_edges(leaf_id, client, positions=positions)
            all_nodes.append(nodes)
            all_edges.append(edges)
        except HTTPError:
            if isinstance(positions, bool) and positions:
                raise ValueError(
                    f"HTTPError: no level 2 graph found for node ID: {leaf_id}"
                )
            else:
                had_error = True
    if had_error:
        print("HTTPError on at least one leaf node, continuing...")
    all_nodes = pd.concat(all_nodes, axis=0)
    all_edges = pd.concat(all_edges, axis=0, ignore_index=True)

    nf = NetworkFrame(all_nodes, all_edges)
    return nf


# %%
print(len(final_nf.nodes))


# %%


def reverse_edit(network_frame: NetworkFrame, network_delta):
    network_frame.add_nodes(network_delta.removed_nodes, inplace=True)
    network_frame.add_edges(network_delta.removed_edges, inplace=True)
    nodes_to_remove = network_delta.added_nodes.index.intersection(
        network_frame.nodes.index
    )
    added_edges = network_delta.added_edges.set_index(["source", "target"])
    added_edges_index = added_edges.index
    current_edges_index = network_frame.edges.set_index(["source", "target"]).index
    edges_to_remove_index = added_edges_index.intersection(current_edges_index)
    edges_to_remove = added_edges.loc[edges_to_remove_index].reset_index()
    if len(nodes_to_remove) > 0 or len(edges_to_remove) > 0:
        network_frame.remove_nodes(nodes_to_remove, inplace=True)
        network_frame.remove_edges(edges_to_remove, inplace=True)
    else:
        print("Skipping edit:", network_delta.metadata)





# %%
n_samples = 15
frac = 0.5
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 5, figsize=(15, 9))

interesting_pool = []
for i in tqdm(range(n_samples)):
    sampled_metaedit_pool = merge_metaedit_pool.sample(frac=frac).values

    partial_nf = final_nf.copy()

    for metaoperation_id in sampled_metaedit_pool:
        networkdelta = networkdeltas_by_metaoperation[metaoperation_id]
        reverse_edit(partial_nf, networkdelta)

    rooted_partial_nf = find_supervoxel_component(nuc_supervoxel, partial_nf, client)

    if len(rooted_partial_nf.nodes) < 1_000:
        interesting_pool += list(sampled_metaedit_pool)

    from pkg.plot import networkplot

    if i < 15:
        ax = axs.flat[i]
        networkplot(
            nodes=rooted_partial_nf.nodes,
            edges=rooted_partial_nf.edges,
            x="x",
            y="y",
            node_size=0.5,
            edge_linewidth=0.25,
            edge_alpha=0.5,
            edge_color="black",
            node_color="black",
            ax=ax,
        )

for ax in axs.flat:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

# %%

# TODO figure out a method for finding the operations that merge soma/nucleus

# %%

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

from pkg.edits import get_detailed_change_log

rows = []
all_modified_nodes = []
for root_id in tqdm(
    query_neurons["pt_root_id"].values, desc="Computing edit statistics"
):
    if root_id == 864691135279452833:
        continue
    (
        networkdeltas_by_operation,
        networkdeltas_by_metaoperation,
    ) = lazy_load_network_edits(root_id, client=client)

    nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]
    current_nuc_level2 = client.chunkedgraph.get_roots([nuc_supervoxel], stop_layer=2)[
        0
    ]
    nuc_pt_nm = client.l2cache.get_l2data(
        [current_nuc_level2], attributes=["rep_coord_nm"]
    )[str(current_nuc_level2)]["rep_coord_nm"]

    change_log = get_detailed_change_log(root_id, client, filtered=False)

    for operation_id, networkdelta in networkdeltas_by_operation.items():
        modified_nodes = pd.concat(
            (networkdelta.added_nodes, networkdelta.removed_nodes)
        )
        n_modified_nodes = len(modified_nodes)
        n_added_nodes = len(networkdelta.added_nodes)
        n_removed_nodes = len(networkdelta.removed_nodes)
        n_added_edges = len(networkdelta.added_edges)
        n_removed_edges = len(networkdelta.removed_edges)
        n_modified_edges = n_added_edges + n_removed_edges
        # modified_node_positions = get_positions(
        #     modified_nodes.index.tolist(), client=client, n_retries=0
        # )

        modified_nodes["root_id"] = root_id
        modified_nodes["operation_id"] = operation_id

        all_modified_nodes.append(modified_nodes)

        row = {
            **networkdelta.metadata,
            "user_name": change_log.loc[operation_id, "user_name"],
            "user_id": change_log.loc[operation_id, "user_id"],
            "root_id": root_id,
            "n_modified_nodes": n_modified_nodes,
            "n_modified_edges": n_modified_edges,
            "n_added_nodes": n_added_nodes,
            "n_removed_nodes": n_removed_nodes,
            "n_added_edges": n_added_edges,
            "n_removed_edges": n_removed_edges,
            "modified_nodes": modified_nodes.index.tolist(),
            "nuc_supervoxel": nuc_supervoxel,
            "current_nuc_level2": current_nuc_level2,
            "nuc_pt_nm": nuc_pt_nm,
            "nuc_x": nuc_pt_nm[0],
            "nuc_y": nuc_pt_nm[1],
            "nuc_z": nuc_pt_nm[2],
        }
        rows.append(row)

edit_stats = pd.DataFrame(rows)
all_modified_nodes = pd.concat(all_modified_nodes)

# %%

from pkg.utils import pt_to_xyz

raw_node_coords = client.l2cache.get_l2data(
    all_modified_nodes.index.to_list(), attributes=["rep_coord_nm"]
)

# %%
node_coords = pd.DataFrame(raw_node_coords).T
node_coords[["x", "y", "z"]] = pt_to_xyz(node_coords["rep_coord_nm"])
node_coords.index = node_coords.index.astype(int)


all_modified_nodes = all_modified_nodes.join(node_coords)

# %%
centroids = all_modified_nodes.groupby(["root_id", "operation_id"])[
    ["x", "y", "z"]
].mean()
centroids.columns = ["centroid_x", "centroid_y", "centroid_z"]
# %%
edit_stats = edit_stats.set_index(["root_id", "operation_id"]).join(centroids)


# %%
edit_stats["centroid_distance_to_nuc"] = (
    (edit_stats["centroid_x"] - edit_stats["nuc_x"]) ** 2
    + (edit_stats["centroid_y"] - edit_stats["nuc_y"]) ** 2
    + (edit_stats["centroid_z"] - edit_stats["nuc_z"]) ** 2
) ** 0.5

# %%

import seaborn.objects as so

# fig, ax = plt.subplots(figsize=(10, 10))

edit_stats["was_forrest"] = edit_stats["user_name"].str.contains("Forrest")

so.Plot(
    edit_stats.query("is_merge & (centroid_distance_to_nuc < 1e6)"),
    x="n_modified_nodes",
    y="centroid_distance_to_nuc",
    color="user_id",
).add(so.Dot(pointsize=3, alpha=0.5))

# %%

# edit_stats.query(
#     "is_merge & (centroid_distance_to_nuc < 3e6) & (n_modified_nodes > 50)"
# )

edit_stats.query(
    "(n_modified_nodes > 20) & is_merge & (centroid_distance_to_nuc < 3e6) & was_forrest"
).sort_values("centroid_distance_to_nuc")


# %%
edit_stats.query(
    "(n_modified_nodes > 20) & is_merge & (centroid_distance_to_nuc < 3e6) & was_forrest"
)

# %%

import seaborn.objects as so

fig, ax = plt.subplots(figsize=(10, 10))
so.Plot(edit_stats, x="n_modified_nodes", y="n_modified_edges", color="is_merge").add(
    so.Dot(pointsize=3, alpha=0.5)
).on(ax)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
so.Plot(
    edit_stats.query("is_merge"),
    x="n_modified_nodes",
    y="n_modified_edges",
    # color="is_merge",
).add(so.Dot(pointsize=3, alpha=0.5)).scale(x="log", y="log").on(ax)

# %%

networkdeltas_by_metaoperation[2]


networkdeltas_by_operation[74987]


# %%
import numpy as np

xmax = np.nanmax(rooted_partial_nf.nodes["x"])
ymax = np.nanmax(rooted_partial_nf.nodes["y"])
xmin = np.nanmin(rooted_partial_nf.nodes["x"])
ymin = np.nanmin(rooted_partial_nf.nodes["y"])


# %%
interesting_pool = pd.Series(interesting_pool)
interesting_pool.value_counts()


# %%


# %%

# 156730229585347516

query_id = 165806629286052511

print("Query in final node set:", query_id in final_nf.nodes.index)


for networkdelta in networkdeltas_by_operation.values():
    if query_id in networkdelta.removed_nodes.index:
        print("removed:", networkdelta.metadata)
    if query_id in networkdelta.added_nodes.index:
        print("added:", networkdelta.metadata)

# %%
nodes, edges = get_level2_nodes_edges(root_id, client=client)

# %%
query_id in nodes.index

# %%
nf = NetworkFrame(nodes, edges)

# %%

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

os.environ["SKEDITS_USE_CLOUD"] = "False"
os.environ["SKEDITS_RECOMPUTE"] = "True"

networkdeltas_by_operation, networkdeltas_by_metaoperation = lazy_load_network_edits(
    root_id, client=client
)

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

nf = lazy_load_initial_network(root_id, client=client)

for edit in tqdm(networkdeltas_by_operation.values()):
    apply_edit(nf, edit)

nodes, edges = get_level2_nodes_edges(root_id, client=client)

final_nf = NetworkFrame(nodes, edges)

assert nf == final_nf

# %%

# %%
