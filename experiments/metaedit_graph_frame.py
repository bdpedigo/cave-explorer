# %%

from datetime import datetime, timedelta

import caveclient as cc
import networkx as nx
import pandas as pd
from pkg.edits import get_changed_edges, get_detailed_change_log
from pkg.utils import get_all_nodes_edges, get_level2_nodes_edges
from tqdm.autonotebook import tqdm
from anytree import Node

from neuropull.graph import NetworkFrame

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
i = 2
target_id = meta.iloc[i]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
root_id = client.chunkedgraph.get_latest_roots(root_id)[0]

print("Root ID:", root_id)

# %%

change_log = get_detailed_change_log(root_id, client, filtered=False)

# %%
print("Number of merges:", change_log["is_merge"].sum())
print("Number of splits:", (~change_log["is_merge"]).sum())


# %%


class NetworkDelta:
    # TODO this is silly right now
    # but would like to add logic for the composition of multiple deltas
    def __init__(self, removed_nodes, added_nodes, removed_edges, added_edges):
        self.removed_nodes = removed_nodes
        self.added_nodes = added_nodes
        self.removed_edges = removed_edges
        self.added_edges = added_edges

    # def __add__(self, other):
    #     # define a new NetworkDelta that is the composition of two deltas
    #     total_added_nodes = pd.concat(
    #         [self.added_nodes, other.added_nodes], verify_integrity=True
    #     )
    #     total_removed_nodes = pd.concat(
    #         [self.removed_nodes, other.removed_nodes], verify_integrity=True
    #     )
    #     later_removed = total_removed_nodes.index.intersection(total_added_nodes.index)
    #     total_added_nodes = total_added_nodes.drop(index=later_removed)


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


#     later_removed = total_removed_edges.index.intersection(total_added_edges.index)
#     total_added_edges = total_added_edges.drop(index=later_removed)

#     return NetworkDelta(
#         total_removed_nodes, total_added_nodes, total_removed_edges, total_added_edges
#     )


# %%
edit_lineage_graph = nx.DiGraph()
networkdeltas_by_operation = {}
for operation_id in tqdm(change_log.index[:]):
    row = change_log.loc[operation_id]
    is_merge = row["is_merge"]
    before_root_ids = row["before_root_ids"]
    after_root_ids = row["roots"]

    # grabbing the union of before/after nodes/edges
    # NOTE: this is where all the compute time comes from
    all_before_nodes, all_before_edges = get_all_nodes_edges(before_root_ids, client)
    all_after_nodes, all_after_edges = get_all_nodes_edges(after_root_ids, client)

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

from pkg.utils import get_lineage_tree
import numpy as np
from datetime import datetime

root = get_lineage_tree(root_id, client, flip=True)

leaves = root.leaves
leaf_ids = np.array([leaf.name for leaf in leaves])

times = pd.Series(pd.to_datetime(cg.get_root_timestamps(leaf_ids)), index=leaf_ids)

date = "2020-07-01"
times[times < date]

good_leaves = times[times < date].index


component_counter = 0

all_nodes = []
pieces = {}
for leaf_id in tqdm(good_leaves):
    # ts = cg.get_root_timestamps(leaf_id)
    # print(ts)
    nodes, edges = get_level2_nodes_edges(leaf_id, client, positions=False)
    # pieces[leaf_id]
    pieces[component_counter] = NetworkFrame(nodes.copy(), edges.copy())

    nodes["component"] = component_counter
    component_counter += 1
    all_nodes.append(nodes)

all_nodes = pd.concat(all_nodes, axis=0)

assert all_nodes.index.value_counts().max() == 1


for operation_id in tqdm(change_log.index[:]):
    # get some info about the operation
    row = change_log.loc[operation_id]
    is_merge = change_log.loc[operation_id, "is_merge"]

    delta = networkdeltas_by_operation[operation_id]
    added_nodes = delta.added_nodes
    added_edges = delta.added_edges
    removed_nodes = delta.removed_nodes
    removed_edges = delta.removed_edges

    references = np.concatenate(
        (
            added_edges["source"],
            added_edges["target"],
            removed_edges["source"],
            removed_edges["target"],
        )
    )
    references = np.unique(references)
    references = references[~np.isin(references, added_nodes.index)]
    known_references = references[np.isin(references, all_nodes.index)]
    reference_component_counts = all_nodes.loc[
        known_references, "component"
    ].value_counts()
    if len(reference_component_counts) > 2:
        raise ValueError(
            f"More than two objects referenced by operation {operation_id}"
        )
    else:
        before_root_ids = reference_component_counts.index

    # before_root_ids = row["before_root_ids"]
    # after_root_ids = row["roots"]

    # collate all the nodes and edges from the before pieces
    all_before_nodes = []
    all_before_edges = []
    for before_root_id in before_root_ids:
        all_before_nodes.append(pieces[before_root_id].nodes)
        all_before_edges.append(pieces[before_root_id].edges)
    all_before_nodes = pd.concat(all_before_nodes, axis=0)
    all_before_edges = pd.concat(all_before_edges, axis=0, ignore_index=True)

    # do the operation
    nf = NetworkFrame(all_before_nodes, all_before_edges)
    nf.add_nodes(added_nodes, inplace=True)
    nf.add_edges(added_edges, inplace=True)
    nf.remove_nodes(removed_nodes.index, inplace=True)
    nf.remove_edges(removed_edges, inplace=True)

    # check that the operation was a valid one
    components = list(nf.connected_components())
    if is_merge:
        assert len(components) == 1
    else:
        assert (len(components) == 2) or (len(components) == 1)

    # this is just necessary for naming the new pieces of neuron in the same way that
    # pychunkedgraph did in reality
    # timestamp = datetime.fromisoformat(row["timestamp"]) + timedelta(microseconds=1)
    # for component in components:
    #     new_root = cg.get_roots(component.nodes.index[0], timestamp=timestamp)[0]
    #     assert new_root in after_root_ids
    #     pieces[new_root] = component

    # TODO don't rely on original names
    for component in components:
        pieces[component_counter] = component

        new_node_ids = component.nodes.index.difference(all_nodes.index)
        new_nodes = pd.DataFrame(index=new_node_ids, columns=all_nodes.columns)
        all_nodes = pd.concat([all_nodes, new_nodes], axis=0)
        all_nodes.loc[component.nodes.index, "component"] = component_counter

        component_counter += 1

# %%


# %%

root_nodes, root_edges = get_level2_nodes_edges(root_id, client, positions=False)
root_nf = NetworkFrame(root_nodes, root_edges)

print("Frames match?", root_nf == pieces[root_id])

print("Different frames don't match?", root_nf != pieces[leaf_ids[0]])

# %%

# TODO Make an animation of the edits happening over time on the neuronal skeleton


# %%
node = Node("a")
node.x = "y"

# %%
hue = "x"
node.__getattribute__(hue)

# %%
from pkg.plot import treeplot
from anytree import PreOrderIter
from anytree.search import find_by_attr

root = get_lineage_tree(root_id, client, flip=True, order="edits")
node_ids = [node.name for node in PreOrderIter(root)]
timestamps = cg.get_root_timestamps(node_ids)

first_month = min(timestamps)
last_month = max(timestamps)

# get the month for each timestamp numbered from first to last
# (so that the hue is a continuous variable)
for node_id, timestamp in zip(node_ids, timestamps):
    node = find_by_attr(root, node_id, name="name")
    node.timescale = (timestamp - first_month) / (last_month - first_month)
    if node.is_leaf and node.timescale > 0.1:
        print(node.timescale)
        print(node.name)
        print(timestamp)


# %%

ax = treeplot(
    root, hue="timescale", palette="RdBu", scatterplot_kws=dict(hue_norm=(0, 1))
)
query_node = 864691135132881568
timestamp = cg.get_root_timestamps(query_node)[0]
node = find_by_attr(root, query_node)
x, y = node._span_position, node.depth

ax.annotate(
    f"{query_node}\nCreated on {timestamp.strftime('%Y-%m-%d')}",
    (x, y),
    xytext=(x + 0.5, y + 6),
    arrowprops=dict(facecolor="black", shrink=0.1, width=1, headwidth=5, headlength=5),
)
ax.set_title(f"Edit lineage for root_id={root_id}")

# %%
get_lineage_tree(query_node, client, flip=True, order="edits")


# %%
for node in root.leaves:
    new_root = get_lineage_tree(node.name, client)
    if not new_root.is_leaf:
        print(new_root.name)

# %%

root = get_lineage_tree(root_id, client, flip=True, order="edits")


# def check_leaf(node_id):
#     out = cg.get_lineage_graph(node_id)


all_nodes = []
all_edges = []
for leaf in tqdm(root.leaves):
    new_root = get_lineage_tree(leaf.name, client)
    if not new_root.is_leaf:
        print(new_root.name)
    if new_root.is_leaf:
        nodes, edges = get_level2_nodes_edges(leaf.name, client, positions=True)
        nodes["leaf"] = leaf.name
        all_nodes.append(nodes)
        all_edges.append(edges)


all_nodes = pd.concat(all_nodes, axis=0)
all_edges = pd.concat(all_edges, axis=0, ignore_index=True)

# %%

from pkg.plot import networkplot

networkplot(all_nodes, all_edges, x="x", y="y", node_hue="leaf")


# %%
cg.get_lineage_graph(864691132360346984)

# %%
leaf = root.leaves[0]
leaf_root = get_lineage_tree(leaf.name, client)
leaf_root.is_leaf

# %%

treeplot(leaf_root)

# %%

leaf_id = 864691136112914108

edges = client.chunkedgraph.level2_chunk_graph(leaf_id)
nodelist = np.unique(edges)

l2data = client.l2cache.get_l2data(nodelist, attributes=["rep_coord_nm"])
l2data = pd.DataFrame(l2data).T
l2data.loc[l2data.isna().any(axis=1)]
