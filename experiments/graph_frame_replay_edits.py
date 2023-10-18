# %%

import caveclient as cc
import networkx as nx
import pandas as pd
from pkg.edits import get_changed_edges, get_detailed_change_log
from pkg.utils import get_all_nodes_edges, get_level2_nodes_edges
from tqdm.autonotebook import tqdm
from pkg.utils import get_lineage_tree
from neuropull.graph import NetworkFrame

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
# i = 2#
i = 6
target_id = meta.iloc[i]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
root_id = client.chunkedgraph.get_latest_roots(root_id)[0]

print("Root ID:", root_id)

# %%

change_log = get_detailed_change_log(root_id, client, filtered=False)
print("Number of merges:", change_log["is_merge"].sum())
print("Number of splits:", (~change_log["is_merge"]).sum())

# %%
lineage_root = get_lineage_tree(root_id, client, flip=True)

for leaf in lineage_root.leaves:
    new_root = get_lineage_tree(leaf.name, client)
    if not new_root.is_leaf:
        print(new_root.name)
        print(cg.get_root_timestamps(new_root.name))


# %%


class NetworkDelta:
    # TODO this is silly right now
    # but would like to add logic for the composition of multiple deltas
    def __init__(self, removed_nodes, added_nodes, removed_edges, added_edges):
        self.removed_nodes = removed_nodes
        self.added_nodes = added_nodes
        self.removed_edges = removed_edges
        self.added_edges = added_edges


def combine_deltas(deltas):
    total_added_nodes = pd.concat(
        [delta.added_nodes for delta in deltas], verify_integrity=True
    )
    total_removed_nodes = pd.concat(
        [delta.removed_nodes for delta in deltas], verify_integrity=True
    )

    total_added_edges = pd.concat(
        [
            delta.added_edges.set_index(["source", "target"], drop=True)
            for delta in deltas
        ],
        verify_integrity=True,
    ).reset_index(drop=False)
    total_removed_edges = pd.concat(
        [
            delta.removed_edges.set_index(["source", "target"], drop=True)
            for delta in deltas
        ],
        verify_integrity=True,
    ).reset_index(drop=False)

    return NetworkDelta(
        total_removed_nodes, total_added_nodes, total_removed_edges, total_added_edges
    )


# %%
edit_lineage_graph = nx.DiGraph()
networkdeltas_by_operation = {}
for operation_id in tqdm(change_log.index[:], desc="Finding deltas for each edit"):
    row = change_log.loc[operation_id]
    is_merge = row["is_merge"]
    before_root_ids = row["before_root_ids"]
    after_root_ids = row["roots"]

    # grabbing the union of before/after nodes/edges
    # NOTE: this is where all the compute time comes from
    all_before_nodes, all_before_edges = get_all_nodes_edges(
        before_root_ids, client, positions=False
    )
    all_after_nodes, all_after_edges = get_all_nodes_edges(
        after_root_ids, client, positions=False
    )

    # finding the nodes that were added or removed, simple set logic
    added_nodes_index = all_after_nodes.index.difference(all_before_nodes.index)
    added_nodes = all_after_nodes.loc[added_nodes_index]
    removed_nodes_index = all_before_nodes.index.difference(all_after_nodes.index)
    removed_nodes = all_before_nodes.loc[removed_nodes_index]

    # finding the edges that were added or removed, simple set logic again
    removed_edges, added_edges = get_changed_edges(all_before_edges, all_after_edges)

    # keep track of what changed
    networkdeltas_by_operation[operation_id] = NetworkDelta(
        removed_nodes, added_nodes, removed_edges, added_edges
    )

    # summarize in edit lineage for L2 level
    for node1 in removed_nodes.index:
        for node2 in added_nodes.index:
            edit_lineage_graph.add_edge(
                node1, node2, operation_id=operation_id, is_merge=is_merge
            )

# %%

meta_operation_map = {}
for i, component in enumerate(nx.weakly_connected_components(edit_lineage_graph)):
    subgraph = edit_lineage_graph.subgraph(component)
    subgraph_operations = set()
    for source, target, data in subgraph.edges(data=True):
        subgraph_operations.add(data["operation_id"])
    meta_operation_map[i] = subgraph_operations

print("Total operations: ", len(change_log))
print("Number of meta-operations: ", len(meta_operation_map))

# %%
networkdeltas_by_meta_operation = {}
for meta_operation_id, operation_ids in meta_operation_map.items():
    meta_networkdelta = combine_deltas(
        [networkdeltas_by_operation[operation_id] for operation_id in operation_ids]
    )
    networkdeltas_by_meta_operation[meta_operation_id] = meta_networkdelta

# %%
from requests import HTTPError

all_nodes = []
all_edges = []
for leaf in tqdm(
    lineage_root.leaves, desc="Finding all L2 components for lineage leaves"
):
    # new_root = get_lineage_tree(leaf.name, client)
    leaf_id = leaf.name
    try:
        nodes, edges = get_level2_nodes_edges(leaf_id, client, positions=False)
    except HTTPError:
        print("HTTPError on node", leaf_id)
        continue
    all_nodes.append(nodes)
    all_edges.append(edges)
all_nodes = pd.concat(all_nodes, axis=0)
all_edges = pd.concat(all_edges, axis=0, ignore_index=True)

nf = NetworkFrame(all_nodes, all_edges)

for operation_id in tqdm(change_log.index[:], desc="Replaying edits"):
    # get some info about the operation
    row = change_log.loc[operation_id]
    is_merge = change_log.loc[operation_id, "is_merge"]

    delta = networkdeltas_by_operation[operation_id]
    added_nodes = delta.added_nodes
    added_edges = delta.added_edges
    removed_nodes = delta.removed_nodes
    removed_edges = delta.removed_edges

    nf.add_nodes(added_nodes, inplace=True)
    nf.add_edges(added_edges, inplace=True)
    nf.remove_nodes(removed_nodes.index, inplace=True)
    nf.remove_edges(removed_edges, inplace=True)

# %%
nuc_supervoxel = nuc.loc[target_id, "pt_supervoxel_id"]

nuc_l2_id = cg.get_root_id(nuc_supervoxel, level2=True)

nuc_nf = None
for component in nf.connected_components():
    if nuc_l2_id in component.nodes.index:
        nuc_nf = component
        break

nuc_nf = nuc_nf.copy()

# %%


root_nodes, root_edges = get_level2_nodes_edges(root_id, client, positions=False)
root_nf = NetworkFrame(root_nodes, root_edges)

print("Frames match?", root_nf == nuc_nf)
