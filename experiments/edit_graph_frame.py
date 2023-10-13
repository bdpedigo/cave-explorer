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
import networkx as nx
import time
from neuropull.graph import NetworkFrame
from neuropull import NetworkFrame
from datetime import datetime, timedelta


# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph
cv = client.info.segmentation_cloudvolume()


# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
i = 0
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


def get_changed_edges(before_edges, after_edges):
    before_edges.drop_duplicates()
    before_edges["is_before"] = True
    after_edges.drop_duplicates()
    after_edges["is_before"] = False
    delta_edges = pd.concat([before_edges, after_edges]).drop_duplicates(
        ["source", "target"], keep=False
    )
    removed_edges = delta_edges.query("is_before").drop(columns=["is_before"])
    added_edges = delta_edges.query("~is_before").drop(columns=["is_before"])
    return removed_edges, added_edges


def get_all_nodes_edges(root_ids, client):
    all_nodes = []
    all_edges = []
    for root_id in root_ids:
        nodes, edges = get_level2_nodes_edges(root_id, client, positions=False)
        nodes["root_id"] = root_id
        edges["root_id"] = root_id
        all_nodes.append(nodes)
        all_edges.append(edges)
    all_nodes = pd.concat(all_nodes, axis=0)
    all_edges = pd.concat(all_edges, axis=0, ignore_index=True)
    return all_nodes, all_edges


class NetworkDelta:
    def __init__(self, removed_nodes, added_nodes, removed_edges, added_edges):
        self.removed_nodes = removed_nodes
        self.added_nodes = added_nodes
        self.removed_edges = removed_edges
        self.added_edges = added_edges


changes_by_operation = {}
networkdeltas_by_operation = {}

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

    all_before_nodes, all_before_edges = get_all_nodes_edges(before_root_ids, client)
    all_after_nodes, all_after_edges = get_all_nodes_edges(after_root_ids, client)

    added_nodes_index = all_after_nodes.index.difference(all_before_nodes.index)
    added_nodes = all_after_nodes.loc[added_nodes_index]
    removed_nodes_index = all_before_nodes.index.difference(all_after_nodes.index)
    removed_nodes = all_before_nodes.loc[removed_nodes_index]

    removed_edges, added_edges = get_changed_edges(all_before_edges, all_after_edges)

    all_root_ids = np.concatenate((before_root_ids, after_root_ids))

    changes_by_root_id = {}
    for root_id in all_root_ids:
        these_removed_nodes = removed_nodes.query("root_id == @root_id").drop(
            columns=["root_id"]
        )
        these_removed_edges = removed_edges.query("root_id == @root_id").drop(
            columns=["root_id"]
        )
        these_added_edges = added_edges.query("root_id == @root_id").drop(
            columns=["root_id"]
        )
        these_added_nodes = added_nodes.query("root_id == @root_id").drop(
            columns=["root_id"]
        )
        changes = {
            "layer2_removed_nodes": these_removed_nodes.index,
            "layer2_removed_edges": these_removed_edges,
            "layer2_added_nodes": these_added_nodes,
            "layer2_added_edges": these_added_edges,
        }
        changes_by_root_id[root_id] = changes

    networkdeltas_by_operation[operation_id] = NetworkDelta(
        removed_nodes, added_nodes, removed_edges, added_edges
    )

    changes_by_operation[operation_id] = changes_by_root_id

# %%


pieces = {}
verbose = False
for operation_id in tqdm(change_log.index[:], disable=verbose):
    if verbose:
        print("Operation ID:", operation_id)
    is_merge = change_log.loc[operation_id, "is_merge"]
    if is_merge:
        if verbose:
            print("Merge")
        row = merges.loc[operation_id]
        before_root_ids = row["before_root_ids"]
        after_root_ids = row["after_root_ids"]
    else:
        if verbose:
            print("Split")
        row = splits.loc[operation_id]
        before_root_ids = row["before_root_ids"]
        # "after_root_ids" doesn't have both children
        after_root_ids = row["roots"]

    all_before_nodes = []
    all_before_edges = []
    for before_root_id in before_root_ids:
        # if we haven't seen this piece yet, pull it
        if before_root_id not in pieces:
            before_nodes, before_edges = get_level2_nodes_edges(
                before_root_id, client, positions=False
            )
            if verbose:
                print("Before network pulled from server")
        else:
            before_nodes = pieces[before_root_id].nodes
            before_edges = pieces[before_root_id].edges
            if verbose:
                print("Before network pulled from cache")
        all_before_nodes.append(before_nodes)
        all_before_edges.append(before_edges)
    all_before_nodes = pd.concat(all_before_nodes, axis=0)
    all_before_edges = pd.concat(all_before_edges, axis=0, ignore_index=True)

    nf = NetworkFrame(all_before_nodes, all_before_edges)
    if verbose:
        print(
            f"Network has {len(list(nf.connected_components()))} connected components pre-operation"
        )

    changes = changes_by_operation[operation_id][before_root_id]

    delta = networkdeltas_by_operation[operation_id]
    added_nodes = delta.added_nodes.drop(columns=["root_id"])
    added_edges = delta.added_edges
    removed_nodes = delta.removed_nodes
    removed_edges = delta.removed_edges

    nf.add_nodes(added_nodes, inplace=True)
    nf.add_edges(added_edges, inplace=True)
    if verbose:
        print(f"Network has {len(nf.nodes)} nodes post-add")

    nf.remove_nodes(removed_nodes.index, inplace=True)
    nf.remove_edges(removed_edges, inplace=True)
    if verbose:
        print(f"Network has {len(nf.nodes)} nodes post-remove")

    components = list(nf.connected_components())
    if verbose:
        print(f"Network has {len(components)} connected components post-operation")

    if is_merge:
        assert len(components) == 1
    else:
        assert (len(components) == 2) or (len(components) == 1)

    timestamp = datetime.fromisoformat(row["timestamp"]) + timedelta(microseconds=1)
    for component in components:
        new_root = cg.get_roots(component.nodes.index[0], timestamp=timestamp)[0]
        assert new_root in after_root_ids
        pieces[new_root] = component

    if verbose:
        print()

# %%

# %%

root_nodes, root_edges = get_level2_nodes_edges(root_id, client, positions=False)
root_nf = NetworkFrame(root_nodes, root_edges)
root_nf

# %%
cobbled_nf = pieces[root_id]
cobbled_nf

# %%
root_nf.nodes.index.isin(cobbled_nf.nodes.index).all()

# %%
edges1 = cobbled_nf.edges[["source", "target"]].copy()
edges2 = root_nf.edges[["source", "target"]].copy()
edges1.reset_index(inplace=True, drop=True)
edges2.reset_index(inplace=True, drop=True)
edges1.sort_values(["source", "target"], inplace=True)
edges2.sort_values(["source", "target"], inplace=True)
edges1.equals(edges2)

# %%
