# %%

from datetime import datetime, timedelta

import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
import numpy as np
import pcg_skel

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph

cv = client.info.segmentation_cloudvolume()

# NOTE: the lowest level segmentation is done in 8x8x40,
# not 4x4x40 (client.info.viewer_resolution())
seg_res = client.chunkedgraph.segmentation_info["scales"][0]["resolution"]
res = client.info.viewer_resolution()

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
i = 1
target_id = meta.iloc[i]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
root_id = client.chunkedgraph.get_latest_roots(root_id)[0]

print("Root ID:", root_id)

# %%

change_log = cg.get_tabular_change_log(root_id)[root_id]
change_log.set_index("operation_id", inplace=True)
change_log.sort_values("timestamp", inplace=True)
change_log.drop(columns=["timestamp"], inplace=True)

merges = change_log.query("is_merge")
splits = change_log.query("~is_merge")

details = cg.get_operation_details(merges.index.to_list())
details = pd.DataFrame(details).T
details.index.name = "operation_id"
details.index = details.index.astype(int)
details = details.explode("roots")

merges = merges.join(details)
# %%
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

# %%


def get_pre_post_l2_ids(node_id, timestamp):
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)

    # 1 microsecond is the finest resolution allowed by timedelta
    delta = timedelta(microseconds=1)
    pre_operation_time = timestamp - delta
    post_operation_time = timestamp + delta

    pre_parent_id = cg.get_roots(node_id, timestamp=pre_operation_time, stop_layer=2)[0]
    post_parent_id = cg.get_roots(node_id, timestamp=post_operation_time, stop_layer=2)[
        0
    ]

    return pre_parent_id, post_parent_id


def get_changed_parent(node_id, timestamp):
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)

    # 1 microsecond is the finest resolution allowed by timedelta
    delta = timedelta(microseconds=1)
    pre_operation_time = timestamp - delta
    post_operation_time = timestamp + delta

    current_layer = cv.get_chunk_layer(node_id)
    parent_layer = current_layer + 1

    pre_parent_id = cg.get_roots(
        node_id, timestamp=pre_operation_time, stop_layer=parent_layer
    )[0]
    post_parent_id = cg.get_roots(
        node_id, timestamp=post_operation_time, stop_layer=parent_layer
    )[0]

    if pre_parent_id == post_parent_id:
        return get_changed_parent(pre_parent_id, timestamp)
    else:
        return pre_parent_id, post_parent_id, parent_layer


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


def format_edgelist(edgelist):
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


def networkplot(
    nodes,
    edges,
    x,
    y,
    node_palette=None,
    node_hue=None,
    node_color="grey",
    node_size=20,
    node_zorder=1,
    edge_palette=None,
    edge_hue=None,
    edge_color="grey",
    edge_linewidth=0.5,
    edge_alpha=1,
    edge_zorder=0,
    ax=None,
    figsize=(10, 10),
    scatterplot_kws={},
    linecollection_kws={},
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    nodes = nodes.copy()
    edges = edges.copy()

    # map the x and y coordinates to the edges
    edges["source_x"] = edges["source"].map(nodes[x])
    edges["source_y"] = edges["source"].map(nodes[y])
    edges["target_x"] = edges["target"].map(nodes[x])
    edges["target_y"] = edges["target"].map(nodes[y])

    if node_hue is None:
        scatterplot_kws["color"] = node_color

    sns.scatterplot(
        data=nodes,
        x=x,
        y=y,
        hue=node_hue,
        palette=node_palette,
        linewidth=0,
        s=node_size,
        ax=ax,
        zorder=node_zorder,
        **scatterplot_kws,
    )

    source_locs = list(zip(edges["source_x"], edges["source_y"]))
    target_locs = list(zip(edges["target_x"], edges["target_y"]))
    segments = list(zip(source_locs, target_locs))

    if edge_palette is not None:
        edge_colors = edges[edge_hue].map(edge_palette)
    else:
        edge_colors = edge_color

    lc = LineCollection(
        segments,
        linewidths=edge_linewidth,
        alpha=edge_alpha,
        color=edge_colors,
        zorder=edge_zorder,
        **linecollection_kws,
    )
    ax.add_collection(lc)


is_gap_merge = pd.Series(index=merges.index, dtype=bool)

sns.set_context("talk", font_scale=0.75)

for operation_id, row in list(merges.iterrows())[:3]:
    timestamp = row["timestamp"]

    # source_supervoxel_id = row["added_edges"][0][0]
    # target_supervoxel_id = row["added_edges"][0][1]

    # source_pre_l2_id, source_post_l2_id = get_pre_post_l2_ids(
    #     source_supervoxel_id, timestamp
    # )
    # target_pre_l2_id, target_post_l2_id = get_pre_post_l2_ids(
    #     target_supervoxel_id, timestamp
    # )

    before1_root_id, before2_root_id = row["before_root_ids"]
    after_root_id = row["after_root_ids"][0]

    after_l2_edgelist = cg.level2_chunk_graph(after_root_id)
    after_l2_nodes, after_l2_edges = format_edgelist(after_l2_edgelist)

    before1_nodes = cg.get_leaves(before1_root_id, stop_layer=2)
    before2_nodes = cg.get_leaves(before2_root_id, stop_layer=2)

    # maybe this should be a union, but if there are any nodes that were in both then
    # we're in trouble...
    before_nodes = np.concatenate([before1_nodes, before2_nodes])

    after_l2_edges["source_was_before1"] = after_l2_edges["source"].isin(before1_nodes)
    after_l2_edges["source_was_before2"] = after_l2_edges["source"].isin(before2_nodes)
    after_l2_edges["target_was_before1"] = after_l2_edges["target"].isin(before1_nodes)
    after_l2_edges["target_was_before2"] = after_l2_edges["target"].isin(before2_nodes)
    new_edges = after_l2_edges.query(
        "~((source_was_before1 & target_was_before1) | (source_was_before2 & target_was_before2))"
    )

    after_l2_nodes["provenance"] = "new"
    after_l2_nodes.loc[
        after_l2_nodes.index.intersection(before1_nodes), "provenance"
    ] = "before object 1"
    after_l2_nodes.loc[
        after_l2_nodes.index.intersection(before2_nodes), "provenance"
    ] = "before object 2"

    after_l2_edges["is_old"] = (
        after_l2_edges["source_was_before1"] & after_l2_edges["target_was_before1"]
    ) | (after_l2_edges["source_was_before2"] & after_l2_edges["target_was_before2"])

    new_edges = after_l2_edges.query("~is_old").copy()
    new_edges["source_x"] = new_edges["source"].map(after_l2_nodes["x"])
    new_edges["source_y"] = new_edges["source"].map(after_l2_nodes["y"])
    new_edges["source_z"] = new_edges["source"].map(after_l2_nodes["z"])
    new_edges["target_x"] = new_edges["target"].map(after_l2_nodes["x"])
    new_edges["target_y"] = new_edges["target"].map(after_l2_nodes["y"])
    new_edges["target_z"] = new_edges["target"].map(after_l2_nodes["z"])

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    node_palette = {
        "new": "firebrick",
        "before object 1": "dodgerblue",
        "before object 2": "mediumblue",
    }
    edge_palette = {True: "royalblue", False: "firebrick"}

    new_edge_mean = new_edges.mean(axis=0)

    x_center = (new_edge_mean["source_x"] + new_edge_mean["target_x"]) / 2
    y_center = (new_edge_mean["source_y"] + new_edge_mean["target_y"]) / 2
    z_center = (new_edge_mean["source_z"] + new_edge_mean["target_z"]) / 2
    gap = 20_000

    # plot in x-y
    ax = axs[0, 0]
    networkplot(
        after_l2_nodes,
        after_l2_edges,
        "x",
        "y",
        node_hue="provenance",
        node_palette=node_palette,
        edge_hue="is_old",
        edge_palette=edge_palette,
        ax=ax,
    )
    networkplot(
        skeleton_nodes,
        skeleton_edges,
        "x",
        "y",
        node_color="grey",
        edge_color="grey",
        ax=ax,
        node_size=1,
        edge_alpha=0.5,
        edge_linewidth=0.5,
        node_zorder=-1,
        edge_zorder=-2,
    )
    ax.set_xlim(x_center - gap, x_center + gap)
    ax.set_ylim(y_center - gap, y_center + gap)

    ax = axs[0, 1]
    ax.axis("off")

    # plot in x-z
    ax = axs[1, 0]
    networkplot(
        after_l2_nodes,
        after_l2_edges,
        "x",
        "z",
        node_hue="provenance",
        node_palette=node_palette,
        edge_hue="is_old",
        edge_palette=edge_palette,
        ax=ax,
    )
    networkplot(
        skeleton_nodes,
        skeleton_edges,
        "x",
        "z",
        node_color="grey",
        edge_color="grey",
        ax=ax,
        node_size=1,
        edge_alpha=0.5,
        edge_linewidth=0.5,
        node_zorder=-1,
        edge_zorder=-2,
    )
    ax.set_xlim(x_center - gap, x_center + gap)
    ax.set_ylim(z_center - gap, z_center + gap)

    # plot in y-z
    ax = axs[1, 1]
    networkplot(
        after_l2_nodes,
        after_l2_edges,
        "y",
        "z",
        node_hue="provenance",
        node_palette=node_palette,
        edge_hue="is_old",
        edge_palette=edge_palette,
        ax=ax,
    )
    networkplot(
        skeleton_nodes,
        skeleton_edges,
        "y",
        "z",
        node_color="grey",
        edge_color="grey",
        ax=ax,
        node_size=1,
        edge_alpha=0.5,
        edge_linewidth=0.5,
        node_zorder=-1,
        edge_zorder=-2,
    )
    ax.set_xlim(y_center - gap, y_center + gap)
    ax.set_ylim(z_center - gap, z_center + gap)


# %%

operation_id = 339158  # this is a L2-L2 merge
detail = cg.get_operation_details([operation_id])[str(operation_id)]
source_supervoxel_id = detail["added_edges"][0][0]
target_supervoxel_id = detail["added_edges"][0][1]

source_pre_l2_id, source_post_l2_id = get_pre_post_l2_ids(
    source_supervoxel_id, timestamp
)
target_pre_l2_id, target_post_l2_id = get_pre_post_l2_ids(
    target_supervoxel_id, timestamp
)

# %%


for operation_id, row in list(merges.iterrows())[:1]:
    detail = cg.get_operation_details([operation_id])[str(operation_id)]
    timestamp = detail["timestamp"]

    source_supervoxel_id = detail["added_edges"][0][0]
    target_supervoxel_id = detail["added_edges"][0][1]

    source_pre_l2_id, source_post_l2_id, source_layer = get_changed_parent(
        source_supervoxel_id, timestamp
    )
    target_pre_l2_id, target_post_l2_id, target_layer = get_changed_parent(
        target_supervoxel_id, timestamp
    )

    print(f"Operation ID: {operation_id}")
    print(f"Source: {source_pre_l2_id} -> {source_post_l2_id} (Level {source_layer})")
    print(f"Target: {target_pre_l2_id} -> {target_post_l2_id} (Level {target_layer})")

    # g.add_edge(
    #     source_pre_l2_id,
    #     source_post_l2_id,
    #     operation_id=operation_id,
    #     timestamp=timestamp,
    #     is_merge=True,
    # )
    # g.add_edge(
    #     target_pre_l2_id,
    #     target_post_l2_id,
    #     operation_id=operation_id,
    #     timestamp=timestamp,
    #     is_merge=True,
    # )

    print()

# # %%
# from anytree import Node


# def build_tree(root_id):
#     children = cg.get_children(root_id)
#     level = cv.get_chunk_layer(root_id)

#     if level == 2:
#         return Node(root_id, supervoxels=children)
#     else:
#         child_nodes = []
#         for child in children:
#             child_node = build_tree(child)
#             child_nodes.append(child_node)
#         root_node = Node(root_id, children=child_nodes)
#         print(level)
#         return root_node


# %%

# operation_id = 339363  # this is a L2-L2 merge


def get_pre_post_l2_ids(node_id, timestamp):
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)

    # 1 microsecond is the finest resolution allowed by timedelta
    delta = timedelta(microseconds=1)
    pre_operation_time = timestamp - delta
    post_operation_time = timestamp + delta

    pre_parent_id = cg.get_roots(node_id, timestamp=pre_operation_time, stop_layer=2)[0]
    post_parent_id = cg.get_roots(node_id, timestamp=post_operation_time, stop_layer=2)[
        0
    ]

    return pre_parent_id, post_parent_id


# %%

operation_id = 339158  # this is a L3-L3 merge

detail = cg.get_operation_details([operation_id])[str(operation_id)]

source_supervoxel_id = detail["added_edges"][0][0]
target_supervoxel_id = detail["added_edges"][0][1]

source_pre_l2_id, source_post_l2_id = get_pre_post_l2_ids(
    source_supervoxel_id, timestamp
)
target_pre_l2_id, target_post_l2_id = get_pre_post_l2_ids(
    target_supervoxel_id, timestamp
)

print(f"Operation ID: {operation_id}")
print(f"Source: {source_pre_l2_id} -> {source_post_l2_id} (Level 2)")
print(f"Target: {target_pre_l2_id} -> {target_post_l2_id} (Level 2)")

row = merges.loc[operation_id]

source1, source2 = row["before_root_ids"]
target = row["after_root_ids"][0]


# %%
row = merges.loc[operation_id]
source1, source2 = row["before_root_ids"]
target = row["after_root_ids"][0]

source1_edgelist = cg.level2_chunk_graph(source1)
source2_edgelist = cg.level2_chunk_graph(source2)
target_edgelist = cg.level2_chunk_graph(target)


source1_nodes, source1_edges = format_edgelist(source1_edgelist)
source2_nodes, source2_edges = format_edgelist(source2_edgelist)
target_nodes, target_edges = format_edgelist(target_edgelist)
# %%
source_index_union = source1_nodes.index.union(source2_nodes.index)

# NOTE: there are L2 nodes that were not in the source graphs, but these are exactly
# the nodes that were added in the merge operation
target_nodes.index.difference(source_index_union)

# %%
# NOTE: likewise, the nodes that were removed in the merge operation are no longer here
source_index_union.difference(target_nodes.index)

# %%
source_combined_edges = pd.concat([source1_edges, source2_edges])
source_combined_edges = source_combined_edges.drop_duplicates()
source_combined_edges["graph"] = "source_combined"

target_edges["graph"] = "target"

# %%

# NOTE: there are edges that changed from source to target
delta_edges = pd.concat([source_combined_edges, target_edges]).drop_duplicates(
    ["source", "target"], keep=False, ignore_index=True
)

id_map = {source_pre_l2_id: source_post_l2_id, target_pre_l2_id: target_post_l2_id}


def remap(x):
    if x in id_map:
        return id_map[x]
    else:
        return x


# but they all have to do with the nodes that were modified, so that's good
delta_edges["source"] = delta_edges["source"].map(remap)
delta_edges["target"] = delta_edges["target"].map(remap)
delta_edges.drop_duplicates(["source", "target"], keep=False, ignore_index=True)

# %%


def get_pre_post_l2_ids(node_id, timestamp):
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)

    # 1 microsecond is the finest resolution allowed by timedelta
    delta = timedelta(microseconds=1)
    pre_operation_time = timestamp - delta
    post_operation_time = timestamp + delta

    current_layer = cv.get_chunk_layer(node_id)
    parent_layer = current_layer + 1

    pre_parent_id = cg.get_roots(
        node_id, timestamp=pre_operation_time, stop_layer=parent_layer
    )[0]
    post_parent_id = cg.get_roots(
        node_id, timestamp=post_operation_time, stop_layer=parent_layer
    )[0]

    return pre_parent_id, post_parent_id


source_pre_l2_id, source_post_l2_id = get_pre_post_l2_ids(
    source_supervoxel_id, timestamp
)
target_pre_l2_id, target_post_l2_id = get_pre_post_l2_ids(
    target_supervoxel_id, timestamp
)

print(f"Operation ID: {operation_id}")
print(f"Source: {source_pre_l2_id} -> {source_post_l2_id} (Level 2)")
print(f"Target: {target_pre_l2_id} -> {target_post_l2_id} (Level 2)")
