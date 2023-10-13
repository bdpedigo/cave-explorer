# %%

from datetime import datetime, timedelta

import caveclient as cc
import pandas as pd
from pkg.utils import get_level2_nodes_edges
from tqdm.autonotebook import tqdm

from neuropull.graph import NetworkFrame

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph
cv = client.info.segmentation_cloudvolume()

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
        # nodes["root_id"] = root_id
        # edges["root_id"] = root_id
        all_nodes.append(nodes)
        all_edges.append(edges)
    all_nodes = pd.concat(all_nodes, axis=0)
    all_edges = pd.concat(all_edges, axis=0, ignore_index=True)
    return all_nodes, all_edges


class NetworkDelta:
    # TODO this is silly right now
    # but would like to add logic for the composition of multiple deltas
    def __init__(self, removed_nodes, added_nodes, removed_edges, added_edges):
        self.removed_nodes = removed_nodes
        self.added_nodes = added_nodes
        self.removed_edges = removed_edges
        self.added_edges = added_edges


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
        # NOTE: "after_root_ids" doesn't have both children
        after_root_ids = row["roots"]

    # grabbing the union of before/after nodes/edges
    all_before_nodes, all_before_edges = get_all_nodes_edges(before_root_ids, client)
    all_after_nodes, all_after_edges = get_all_nodes_edges(after_root_ids, client)

    # finding the nodes that were added or removed, simple set logic
    added_nodes_index = all_after_nodes.index.difference(all_before_nodes.index)
    added_nodes = all_after_nodes.loc[added_nodes_index]
    removed_nodes_index = all_before_nodes.index.difference(all_after_nodes.index)
    removed_nodes = all_before_nodes.loc[removed_nodes_index]

    # finding the edges that were added or removed, simple set logic again
    removed_edges, added_edges = get_changed_edges(all_before_edges, all_after_edges)

    networkdeltas_by_operation[operation_id] = NetworkDelta(
        removed_nodes, added_nodes, removed_edges, added_edges
    )

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

    delta = networkdeltas_by_operation[operation_id]
    added_nodes = delta.added_nodes
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


def are_frames_equal(frame1, frame2):
    frame1.nodes.sort_index(inplace=True)
    frame2.nodes.sort_index(inplace=True)
    frame1.edges.sort_values(["source", "target"], inplace=True)
    frame2.edges.sort_values(["source", "target"], inplace=True)
    if not frame1.nodes.equals(frame2.nodes):
        return False
    if not frame1.edges.equals(frame2.edges):
        return False
    return True


root_nodes, root_edges = get_level2_nodes_edges(root_id, client, positions=False)
root_nf = NetworkFrame(root_nodes, root_edges)

print("Frames match?", are_frames_equal(root_nf, pieces[root_id]))
