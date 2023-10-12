# %%

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nglui.statebuilder import make_neuron_neuroglancer_link
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
i = 9
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
details = cg.get_operation_details(splits.index.to_list())
details = pd.DataFrame(details).T
details.index.name = "operation_id"
details.index = details.index.astype(int)
splits = splits.join(details)

# %%
print("Number of merges:", len(merges))
print("Number of splits:", len(splits))

# %%

if False:
    operation_id = 544353
    row = splits.loc[operation_id]
    details = cg.get_operation_details([operation_id])[str(operation_id)]

    root_ids = list(row["before_root_ids"]) + list(row["after_root_ids"])

    position = np.array(row["source_coords"][0]) * np.array([2, 2, 1])

    make_neuron_neuroglancer_link(
        client,
        root_ids,
        # view_kws={"position": position},
    )

# %%

import networkx as nx

edit_lineage_graph = nx.DiGraph()

for operation_id, row in tqdm(
    merges.iterrows(), total=len(merges), desc="Finding merge lineage relationships"
):
    before1_root_id, before2_root_id = row["before_root_ids"]
    after_root_id = row["after_root_ids"][0]

    before1_nodes = cg.get_leaves(before1_root_id, stop_layer=2)
    before2_nodes = cg.get_leaves(before2_root_id, stop_layer=2)
    after_nodes = cg.get_leaves(after_root_id, stop_layer=2)

    before_nodes = np.concatenate((before1_nodes, before2_nodes))
    removed_nodes = np.setdiff1d(before_nodes, after_nodes)
    added_nodes = np.setdiff1d(after_nodes, before_nodes)
    for node1 in removed_nodes:
        for node2 in added_nodes:
            edit_lineage_graph.add_edge(
                node1, node2, operation_id=operation_id, operation_type="merge"
            )

for operation_id, row in tqdm(
    splits.iterrows(), total=len(splits), desc="Finding split lineage relationships"
):
    before_root_id = row["before_root_ids"][0]

    # TODO: this is a hack to get around the fact that some splits have only one after
    # root ID. This is because sometimes a split is performed but the two objects are
    # still connected in another place, so they don't become two new roots.
    # Unsure how to handle this case in terms of tracking edits to replay laters

    # after1_root_id, after2_root_id = row["roots"]
    after_root_ids = row["roots"]

    before_nodes = cg.get_leaves(before_root_id, stop_layer=2)

    after_nodes = []
    for after_root_id in after_root_ids:
        after_nodes.append(cg.get_leaves(after_root_id, stop_layer=2))
    after_nodes = np.concatenate(after_nodes)

    removed_nodes = np.setdiff1d(before_nodes, after_nodes)
    added_nodes = np.setdiff1d(after_nodes, before_nodes)

    for node1 in removed_nodes:
        for node2 in added_nodes:
            edit_lineage_graph.add_edge(
                node1, node2, operation_id=operation_id, operation_type="split"
            )


# %%
meta_operation_map = {}
for i, component in enumerate(nx.weakly_connected_components(edit_lineage_graph)):
    subgraph = edit_lineage_graph.subgraph(component)
    subgraph_operations = set()
    for source, target, data in subgraph.edges(data=True):
        subgraph_operations.add(data["operation_id"])
    meta_operation_map[i] = subgraph_operations

print("Total operations: ", len(merges) + len(splits))
print("Number of meta-operations: ", len(meta_operation_map))

# %%


def get_removed_edges(before1_edges, before2_edges, after_edges, added_nodes):
    before1_edges = before1_edges[["source", "target"]]
    before2_edges = before2_edges[["source", "target"]]
    after_edges = after_edges[["source", "target"]]
    before_edges = pd.concat([before1_edges, before2_edges])
    before_edges.drop_duplicates()
    removed_edges = pd.concat([before_edges, after_edges]).drop_duplicates(
        ["source", "target"], keep=False
    )
    removed_edges = removed_edges.query(
        "(~source.isin(@added_nodes)) and (~target.isin(@added_nodes))"
    )
    return removed_edges


def get_changed_edges(befores, afters):
    before_edges = pd.concat(befores)
    before_edges.drop_duplicates()
    before_edges["is_before"] = True
    after_edges = pd.concat(afters)
    after_edges.drop_duplicates()
    after_edges["is_before"] = False
    delta_edges = pd.concat([before_edges, after_edges]).drop_duplicates(
        ["source", "target"], keep=False
    )
    removed_edges = delta_edges.query("is_before").drop(columns=["is_before"])
    added_edges = delta_edges.query("~is_before").drop(columns=["is_before"])
    return removed_edges, added_edges


# %%

import time

old_time = 0
new_time = 0

changes_by_operation = {}
for operation_id in tqdm(merges.index):
    row = merges.loc[operation_id]

    before1_root_id, before2_root_id = row["before_root_ids"]
    after_root_id = row["after_root_ids"][0]

    before1_nodes, before1_edges = get_level2_nodes_edges(
        before1_root_id, client, positions=False
    )
    before2_nodes, before2_edges = get_level2_nodes_edges(
        before2_root_id, client, positions=False
    )
    before_nodes = pd.concat((before1_nodes, before2_nodes), axis=0)
    before_edges = pd.concat((before1_edges, before2_edges), axis=0, ignore_index=True)
    after_nodes, after_edges = get_level2_nodes_edges(
        after_root_id, client, positions=False
    )

    added_nodes = after_nodes.index.difference(before_nodes.index)
    removed_nodes = before_nodes.index.difference(after_nodes.index)

    # methods using edge difference logic (feels more robust, but slower?)
    t0 = time.time()
    # removed_edges_old = get_removed_edges(
    #     before1_edges, before2_edges, after_edges, added_nodes
    # )
    removed_edges_old, added_edges_old = get_changed_edges(
        [before1_edges, before2_edges], [after_edges]
    )
    old_time += time.time() - t0

    # methods using node pre/post logic

    # old edges were B1 -> B1 or B2 -> B2
    # any edge B1 -> B2, B2 -> B1, or anything with a new node is new
    after_edges["is_old"] = (
        after_edges["source"].isin(before1_nodes.index)
        & after_edges["target"].isin(before1_nodes.index)
    ) | (
        after_edges["source"].isin(before2_nodes.index)
        & after_edges["target"].isin(before2_nodes.index)
    )
    added_edges = after_edges.query("~is_old").drop(columns=["is_old"])
    t0 = time.time()
    removed_edges = before_edges.query(
        "source.isin(@removed_nodes) or target.isin(@removed_nodes)"
    )
    new_time += time.time() - t0

    assert removed_edges.reset_index(drop=True).equals(
        removed_edges_old.reset_index(drop=True)
    )
    assert added_edges.reset_index(drop=True).equals(
        added_edges_old.reset_index(drop=True)
    )

    changes_by_operation[operation_id] = {
        "added_edges": added_edges[["source", "target"]].values.tolist(),
        "added_nodes": added_nodes.to_list(),
        "removed_nodes": removed_nodes.to_list(),
        "removed_edges": removed_edges[["source", "target"]].values.tolist(),
    }

changes_by_operation = pd.DataFrame(changes_by_operation).T

print("Old time:", old_time)
print("New time:", new_time)

# %%


def get_added_edges(before_edges, after_edges, added_nodes):
    before_edges = before_edges[["source", "target"]]
    after_edges = after_edges[["source", "target"]]
    before_edges.drop_duplicates()
    after_edges.drop_duplicates()
    added_edges = pd.concat([before_edges, after_edges]).drop_duplicates(
        ["source", "target"], keep=False
    )

    added_edges = added_edges.query()
    added_edges = added_edges.query(
        "(~source.isin(@added_nodes)) and (~target.isin(@added_nodes))"
    )

    # removed_edges = removed_edges.query(
    #     "(~source.isin(@added_nodes)) and (~target.isin(@added_nodes))"
    # )
    return added_edges


for operation_id in tqdm(splits.index[:1]):
    row = splits.loc[operation_id]

    before_root_id = row["before_root_ids"][0]
    after_root_ids = row["roots"]

    before_nodes, before_edges = get_level2_nodes_edges(
        before_root_id, client, positions=False
    )

    # Sometimes splits have only one child, so we need to handle that case
    after_nodes = []
    after_edges = []
    for after_root_id in after_root_ids:
        after_node, after_edge = get_level2_nodes_edges(
            after_root_id, client, positions=False
        )
        after_nodes.append(after_node)
        after_edges.append(after_edge)

    after_nodes = pd.concat(after_nodes, axis=0)
    after_edges = pd.concat(after_edges, axis=0, ignore_index=True)

    added_nodes = after_nodes.index.difference(before_nodes.index)
    removed_nodes = before_nodes.index.difference(after_nodes.index)

    added_edges = after_edges.query(
        "source.isin(@added_nodes) or target.isin(@added_nodes)"
    )

    removed_edges = before_edges.query(
        "source.isin(@removed_nodes) or target.isin(@removed_nodes)"
    )


# %%


def get_changed_edges(before_edges, after_edges):
    before_edges[["source", "target"]].drop_duplicates()
    before_edges["is_before"] = True
    after_edges[["source", "target"]].drop_duplicates()
    after_edges["is_before"] = False
    delta_edges = pd.concat([before_edges, after_edges]).drop_duplicates(
        ["source", "target"], keep=False
    )
    removed_edges = delta_edges.query("is_before").drop(columns=["is_before"])
    added_edges = delta_edges.query("~is_before").drop(columns=["is_before"])
    return removed_edges, added_edges


# TODO re-write using one for-loop over operations

changes_by_operation = {}

for operation_id in tqdm(change_log.index[:]):
    is_merge = change_log.loc[operation_id]["is_merge"]
    if is_merge:
        row = merges.loc[operation_id]
        before_root_ids = row["before_root_ids"]
        after_root_ids = row["after_root_ids"]
    else:
        row = splits.loc[operation_id]
        before_root_ids = row["before_root_ids"]
        # "after_root_ids" doesn't have both children
        after_root_ids = row["roots"]

    before_nodes = []
    before_edges = []
    for before_root_id in before_root_ids:
        before_node, before_edge = get_level2_nodes_edges(
            before_root_id, client, positions=False
        )
        before_nodes.append(before_node)
        before_edges.append(before_edge)
    before_nodes = pd.concat(before_nodes, axis=0)
    before_edges = pd.concat(before_edges, axis=0, ignore_index=True)

    after_nodes = []
    after_edges = []
    for after_root_id in after_root_ids:
        after_node, after_edge = get_level2_nodes_edges(
            after_root_id, client, positions=False
        )
        after_nodes.append(after_node)
        after_edges.append(after_edge)
    after_nodes = pd.concat(after_nodes, axis=0)
    after_edges = pd.concat(after_edges, axis=0, ignore_index=True)

    added_nodes = after_nodes.index.difference(before_nodes.index)
    removed_nodes = before_nodes.index.difference(after_nodes.index)

    removed_edges, added_edges = get_changed_edges(before_edges, after_edges)

    changes_by_operation[operation_id] = {
        "layer2_added_edges": added_edges[["source", "target"]].values.tolist(),
        "layer2_added_nodes": added_nodes.to_list(),
        "layer2_removed_nodes": removed_nodes.to_list(),
        "layer2_removed_edges": removed_edges[["source", "target"]].values.tolist(),
        **row.to_dict(),
    }

changes_by_operation = pd.DataFrame(changes_by_operation).T

# %%

changes_by_operation

# %%
# TODO start from the graph at t_0, then apply the changes in order
# make sure that the graph at the end is the same as the graph at t_end (the root_id
# we started with)

# %%
before_node, before_edge = get_level2_nodes_edges(
    before_root_id, client, positions=False
)

# %%
lineage_graph_dict = cg.get_lineage_graph(root_id)

# %%
len(lineage_graph_dict["links"])

# %%
pd.DataFrame(lineage_graph_dict["nodes"][1:]).set_index("operation_id")  # first is root
