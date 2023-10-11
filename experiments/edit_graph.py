# %%

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.paths import FIG_PATH
from pkg.plot import networkplot
from pkg.utils import get_level2_nodes_edges, get_skeleton_nodes_edges
from tqdm.autonotebook import tqdm

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph
cv = client.info.segmentation_cloudvolume()

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
i = 5
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
details = cg.get_operation_details(merges.index.to_list())
details = pd.DataFrame(details).T
details.index.name = "operation_id"
details.index = details.index.astype(int)
details = details.explode("roots")
merges = merges.join(details)

# %%
splits = change_log.query("~is_merge")

# %%
details = cg.get_operation_details(splits.index.to_list())
details = pd.DataFrame(details).T
details.index.name = "operation_id"
details.index = details.index.astype(int)
splits = splits.join(details)

# %%

# NOTE: this was helpful conceptually but don't think it's strictly necessary
# for the analysis going forward
if False:
    new_nodes_by_operation = {}
    for operation_id, row in tqdm(
        merges.iterrows(), total=len(merges), desc="Finding changed L2 nodes"
    ):
        before1_root_id, before2_root_id = row["before_root_ids"]
        after_root_id = row["after_root_ids"][0]

        before1_nodes = cg.get_leaves(before1_root_id, stop_layer=2)
        before2_nodes = cg.get_leaves(before2_root_id, stop_layer=2)
        after_nodes = cg.get_leaves(after_root_id, stop_layer=2)

        before_union = np.concatenate((before1_nodes, before2_nodes))
        new_nodes = np.setdiff1d(after_nodes, before_union)
        new_nodes_by_operation[operation_id] = list(new_nodes)

    new_nodes_by_operation = pd.Series(new_nodes_by_operation, name="new_l2_nodes")

    merges = merges.join(new_nodes_by_operation)
    merges

# %%


def get_edge_center(nodes, edges):
    edges = edges.copy()
    for col in ["source", "target"]:
        for dim in ["x", "y", "z"]:
            edges[f"{col}_{dim}"] = edges[col].map(nodes[dim])

    edge_mean = edges.mean(axis=0)

    x_center = (edge_mean["source_x"] + edge_mean["target_x"]) / 2
    y_center = (edge_mean["source_y"] + edge_mean["target_y"]) / 2
    z_center = (edge_mean["source_z"] + edge_mean["target_z"]) / 2
    return (x_center, y_center, z_center)


def editplot(after_nodes, after_edges, skeleton_nodes, skeleton_edges):
    gap = 20_000
    new_edges = after_edges.query("~is_old")
    x_center, y_center, z_center = get_edge_center(after_nodes, new_edges)

    fig, axs = plt.subplots(
        2, 2, figsize=(10, 10), constrained_layout=True, sharex="col", sharey="row"
    )
    node_palette = {
        "new": "firebrick",
        "before object 1": "dodgerblue",
        "before object 2": "mediumblue",
    }
    edge_palette = {True: "royalblue", False: "firebrick"}

    # plot in x-y
    ax = axs[0, 0]
    networkplot(
        after_nodes,
        after_edges,
        "x",
        "y",
        node_hue="provenance",
        node_palette=node_palette,
        edge_hue="is_old",
        edge_palette=edge_palette,
        edge_linewidth=1,
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
        edge_alpha=0.3,
        edge_linewidth=4,
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
        after_nodes,
        after_edges,
        "x",
        "z",
        node_hue="provenance",
        node_palette=node_palette,
        edge_hue="is_old",
        edge_palette=edge_palette,
        edge_linewidth=1,
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
        edge_alpha=0.3,
        edge_linewidth=4,
        node_zorder=-1,
        edge_zorder=-2,
    )
    ax.set_xlim(x_center - gap, x_center + gap)
    ax.set_ylim(z_center - gap, z_center + gap)
    ax.get_legend().remove()

    # plot in y-z
    ax = axs[1, 1]
    networkplot(
        after_nodes,
        after_edges,
        "y",
        "z",
        node_hue="provenance",
        node_palette=node_palette,
        edge_hue="is_old",
        edge_palette=edge_palette,
        edge_linewidth=1,
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
        edge_alpha=0.3,
        edge_linewidth=4,
        node_zorder=-1,
        edge_zorder=-2,
    )
    ax.set_xlim(y_center - gap, y_center + gap)
    ax.set_ylim(z_center - gap, z_center + gap)
    ax.get_legend().remove()

    for ax in axs.flat:
        ax.ticklabel_format(style="sci", scilimits=[-3, 3])

    return fig, axs


sns.set_context(
    "paper", font_scale=1.5, rc={"axes.spines.right": False, "axes.spines.top": False}
)

show_plots = False

import networkx as nx

edit_lineage_graph = nx.DiGraph()


for operation_id, row in tqdm(
    list(merges.iterrows())[:],
    desc="Finding (and maybe plotting) changes to spatial graph",
):
    after_root_id = row["after_root_ids"][0]
    before1_root_id, before2_root_id = row["before_root_ids"]

    after_nodes, after_edges = get_level2_nodes_edges(after_root_id, client)

    # find the nodes in the L2 graph that were added/removed
    before1_nodes = cg.get_leaves(before1_root_id, stop_layer=2)
    before2_nodes = cg.get_leaves(before2_root_id, stop_layer=2)
    before_nodes = np.concatenate((before1_nodes, before2_nodes))
    removed_nodes = np.setdiff1d(before_nodes, after_nodes.index)
    added_nodes = np.setdiff1d(after_nodes.index, before_nodes)
    for node1 in removed_nodes:
        for node2 in added_nodes:
            edit_lineage_graph.add_edge(
                node1, node2, operation_id=operation_id, operation_type="merge"
            )

    # apply similar logic to the node table
    after_nodes["provenance"] = "new"
    after_nodes.loc[
        after_nodes.index.intersection(before1_nodes), "provenance"
    ] = "before object 1"
    after_nodes.loc[
        after_nodes.index.intersection(before2_nodes), "provenance"
    ] = "before object 2"

    # use this information to label each edge
    after_edges["source_was_before1"] = after_edges["source"].isin(before1_nodes)
    after_edges["source_was_before2"] = after_edges["source"].isin(before2_nodes)
    after_edges["target_was_before1"] = after_edges["target"].isin(before1_nodes)
    after_edges["target_was_before2"] = after_edges["target"].isin(before2_nodes)

    # old edges are those from B1 to B1 or B2 to B2
    after_edges["is_old"] = (
        after_edges["source_was_before1"] & after_edges["target_was_before1"]
    ) | (after_edges["source_was_before2"] & after_edges["target_was_before2"])

    if show_plots:
        skeleton_nodes, skeleton_edges = get_skeleton_nodes_edges(root_id, client)

        fig, ax = editplot(after_nodes, after_edges, skeleton_nodes, skeleton_edges)
        fig.suptitle(f"Final root ID: {root_id}, Operation ID: {operation_id}")
        fig.savefig(
            FIG_PATH / f"edit_graph/root={root_id}-operation={operation_id}.png",
            dpi=300,
        )
        plt.close()

for operation_id, row in tqdm(
    list(splits.iterrows())[:], desc="Plotting changes to spatial graph"
):
    before_root_id = row["before_root_ids"][0]
    after1_root_id, after2_root_id = row["roots"]

    before_nodes, before_edges = get_level2_nodes_edges(before_root_id, client)

    after1_nodes = cg.get_leaves(after1_root_id, stop_layer=2)
    after2_nodes = cg.get_leaves(after2_root_id, stop_layer=2)

    # find the nodes in the L2 graph that were added/removed
    after_nodes = np.concatenate((after1_nodes, after2_nodes))
    removed_nodes = np.setdiff1d(before_nodes.index, after_nodes)
    added_nodes = np.setdiff1d(after_nodes, before_nodes.index)
    for node1 in removed_nodes:
        for node2 in added_nodes:
            edit_lineage_graph.add_edge(
                node1, node2, operation_id=operation_id, operation_type="split"
            )

    # apply similar logic to the node table
    before_nodes["fate"] = "removed"
    before_nodes.loc[
        before_nodes.index.intersection(after1_nodes), "fate"
    ] = "after object 1"
    before_nodes.loc[
        before_nodes.index.intersection(after2_nodes), "fate"
    ] = "after object 2"

    # use this information to label each edge
    before_edges["source_goes_after1"] = before_edges["source"].isin(after1_nodes)
    before_edges["source_goes_after2"] = before_edges["source"].isin(after2_nodes)
    before_edges["target_goes_after1"] = before_edges["target"].isin(after1_nodes)
    before_edges["target_goes_after2"] = before_edges["target"].isin(after2_nodes)

    # remaining edges are those from A1 to A1 or A2 to A2
    before_edges["is_remaining"] = (
        before_edges["source_goes_after1"] & before_edges["target_goes_after1"]
    ) | (before_edges["source_goes_after2"] & before_edges["target_goes_after2"])

    # after_nodes.loc[
    #     after_nodes.index.intersection(before1_nodes), "provenance"
    # ] = "before object 1"
    # after_nodes.loc[
    #     after_nodes.index.intersection(before2_nodes), "provenance"
    # ] = "before object 2"

    # after_edges["source_was_before1"] = after_edges["source"].isin(before1_nodes)
    # after_edges["source_was_before2"] = after_edges["source"].isin(before2_nodes)
    # after_edges["target_was_before1"] = after_edges["target"].isin(before1_nodes)
    # after_edges["target_was_before2"] = after_edges["target"].isin(before2_nodes)

    # after_edges["is_old"] = (
    #     after_edges["source_was_before1"] & after_edges["target_was_before1"]
    # ) | (after_edges["source_was_before2"] & after_edges["target_was_before2"])

    # skeleton_nodes, skeleton_edges = get_skeleton_nodes_edges(root_id, client)

    # fig, ax = editplot(after_nodes, after_edges, skeleton_nodes, skeleton_edges)
    # fig.suptitle(f"Final root ID: {root_id}, Operation ID: {operation_id}")
    # fig.savefig(
    #     FIG_PATH / f"edit_graph/root={root_id}-operation={operation_id}.png",
    #     dpi=300,
    # )
    # plt.close()

# %%

len(list(nx.weakly_connected_components(edit_lineage_graph)))

# %%

meta_operation_map = {}
for i, component in enumerate(nx.weakly_connected_components(edit_lineage_graph)):
    subgraph = edit_lineage_graph.subgraph(component)
    subgraph_operations = set()
    for source, target, data in subgraph.edges(data=True):
        subgraph_operations.add(data["operation_id"])
    meta_operation_map[i] = subgraph_operations

# %%
