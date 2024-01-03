# %%

import time

t0 = time.time()

from datetime import timedelta

import caveclient as cc
import numpy as np
from networkframe import NetworkFrame
from tqdm.auto import tqdm

from pkg.edits import (
    find_supervoxel_component,
    get_initial_network,
    get_network_edits,
    get_network_metaedits,
)
from pkg.utils import get_level2_nodes_edges

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
# i = 2#
# i = 14

# i = 23
# i = 6  # this one works
# i = 4
i = 6

target_id = meta.iloc[i]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
root_id = client.chunkedgraph.get_latest_roots(root_id)[0]


# %%

networkdeltas_by_operation, edit_lineage_graph = get_network_edits(
    root_id, client, filtered=False
)
print()

# %%

print("Pulling initial state of the network")
nf = get_initial_network(root_id, client, positions=False)
print()
for edit_id, delta in networkdeltas_by_operation.items():
    # delta = networkdeltas_by_meta_operation[metaedit_id]
    nf.add_nodes(delta.added_nodes, inplace=True)
    nf.add_edges(delta.added_edges, inplace=True)
    nf.remove_nodes(delta.removed_nodes, inplace=True)
    nf.remove_edges(delta.removed_edges, inplace=True)

print("Finding final fragment with nucleus attached")
nuc_supervoxel = nuc.loc[target_id, "pt_supervoxel_id"]

nuc_nf = find_supervoxel_component(nuc_supervoxel, nf, client)
print()

print("Checking for correspondence of final edited neuron and original root neuron")
root_nodes, root_edges = get_level2_nodes_edges(root_id, client, positions=False)
root_nf = NetworkFrame(root_nodes, root_edges)

print("L2 graphs match?", root_nf == nuc_nf)
print()

# %%

import pandas as pd

mod_sets = {}
for edit_id, delta in networkdeltas_by_operation.items():
    mod_set = []
    mod_set += list(delta.added_nodes.index)
    mod_set += list(delta.removed_nodes.index)
    mod_set += delta.added_edges["source"].tolist()
    mod_set += delta.added_edges["target"].tolist()
    mod_set += delta.removed_edges["source"].tolist()
    mod_set += delta.removed_edges["target"].tolist()
    mod_set = np.unique(mod_set)
    mod_sets[edit_id] = mod_set
index = np.unique(np.concatenate(list(mod_sets.values())))
node_edit_indicators = pd.DataFrame(
    index=index, columns=networkdeltas_by_operation.keys(), data=False
)

for edit_id, mod_set in mod_sets.items():
    node_edit_indicators.loc[mod_set, edit_id] = True

X = node_edit_indicators.values.astype(int)
product = X.T @ X
product = pd.DataFrame(
    index=node_edit_indicators.columns,
    columns=node_edit_indicators.columns,
    data=product,
)

from scipy.sparse.csgraph import connected_components

n_components, labels = connected_components(product.values, directed=False)

meta_operation_map = {}
for label in np.unique(labels):
    meta_operation_map[label] = node_edit_indicators.columns[labels == label].tolist()


# %%

print("Finding meta-operations")
networkdeltas_by_meta_operation, meta_operation_map = get_network_metaedits(
    networkdeltas_by_operation
)
print()

# %%
print("Pulling initial state of the network")
nf = get_initial_network(root_id, client, positions=False)
print()

print()

np.random.seed(7)  # 1 works


metaedit_ids = np.array(list(networkdeltas_by_meta_operation.keys()))
random_metaedit_ids = np.random.permutation(metaedit_ids)

for metaedit_id in tqdm(
    random_metaedit_ids, desc="Playing meta-edits in random order", disable=True
):
    delta = networkdeltas_by_meta_operation[metaedit_id]
    nf.add_nodes(delta.added_nodes, inplace=True)
    nf.add_edges(delta.added_edges, inplace=True)
    nf.remove_nodes(delta.removed_nodes, inplace=True)
    nf.remove_edges(delta.removed_edges, inplace=True)

print()


print("Finding final fragment with nucleus attached")
nuc_supervoxel = nuc.loc[target_id, "pt_supervoxel_id"]

nuc_nf = find_supervoxel_component(nuc_supervoxel, nf, client)
print()

print("Checking for correspondence of final edited neuron and original root neuron")
root_nodes, root_edges = get_level2_nodes_edges(root_id, client, positions=False)
root_nf = NetworkFrame(root_nodes, root_edges)

print("L2 graphs match?", root_nf == nuc_nf)
print()
# %%

diffs = root_nf.nodes.index.difference(nuc_nf.nodes.index)
print(diffs)

edge_diffs = root_nf.edges.set_index(["source", "target"]).index.difference(
    nf.edges.set_index(["source", "target"]).index
)
print(edge_diffs)

# nuc_nf.edges.set_index(["source", "target"]).index.difference(
#     root_nf.edges.set_index(["source", "target"]).index
# )

# %%
query = edge_diffs[0]

# %%
edges_copy = nf.edges.copy().set_index(["source", "target"])

# %%

print(query in root_nf.edges.set_index(["source", "target"]).index)

print(query in nf.edges.set_index(["source", "target"]).index)

# %%

print("Pulling initial state of the network")
nf = get_initial_network(root_id, client, positions=False)
print()

query = 161570210917124738

for edit_id, delta in networkdeltas_by_operation.items():
    nf.add_nodes(delta.added_nodes, inplace=True)
    nf.add_edges(delta.added_edges, inplace=True)
    nf.remove_nodes(delta.removed_nodes, inplace=True)
    nf.remove_edges(delta.removed_edges, inplace=True)
    if query in nf.nodes.index:
        print("Node present at edit: ", edit_id)

    # added = delta.added_edges.set_index(["source", "target"]).index
    # removed = delta.removed_edges.set_index(["source", "target"]).index
    # if query in added:
    #     print("added")
    #     print(edit_id)
    # if query in removed:
    #     print("removed")
    #     print(edit_id)

# %%
print("Pulling initial state of the network")
nf = get_initial_network(root_id, client, positions=False)
print()

edges = nf.edges.set_index(["source", "target"]).index
if query in edges:
    print("Edge present at start")

np.random.seed(7)  # 1 works

metaedit_ids = np.array(list(networkdeltas_by_meta_operation.keys()))
random_metaedit_ids = np.random.permutation(metaedit_ids)

for metaedit_id in tqdm(
    random_metaedit_ids, desc="Playing meta-edits in random order", disable=True
):
    delta = networkdeltas_by_meta_operation[metaedit_id]

    # if metaedit_id == 32:
    # print("here")
    # print(query in delta.added_edges.set_index(["source", "target"]).index)
    # print(query in delta.removed_edges.set_index(["source", "target"]).index)

    nf.add_nodes(delta.added_nodes, inplace=True)
    nf.add_edges(delta.added_edges, inplace=True)
    edges = nf.edges.set_index(["source", "target"]).index
    if metaedit_id == 32:
        if query in edges:
            print("Edge present at metaedit after additions: ", metaedit_id)
            print(delta.added_edges)
            print(nf.edges.query(f"source == {query[0]} and target == {query[1]}"))

    if metaedit_id == 32:
        print(query[0] in nf.nodes.index)
        print(query[1] in nf.nodes.index)

    # SOMEHOW THE EDGE DISAPPEARS AFTER THIS LINE
    nf.remove_nodes(delta.removed_nodes, inplace=True)
    edges = nf.edges.set_index(["source", "target"]).index

    if metaedit_id == 32:
        print(nf.edges.query(f"source == {query[0]} and target == {query[1]}"))
        if query in edges:
            print("Edge present at metaedit after remove nodes: ", metaedit_id)

    nf.remove_edges(delta.removed_edges, inplace=True)
    edges = nf.edges.set_index(["source", "target"]).index
    if metaedit_id == 32:
        if query in edges:
            print("Edge present at metaedit after remove edges: ", metaedit_id)

# %%
networkdeltas_by_meta_operation[32].added_nodes

# %%

for metaedit_id in tqdm(
    random_metaedit_ids, desc="Playing meta-edits in random order", disable=True
):
    delta = networkdeltas_by_meta_operation[metaedit_id]
    if query[0] in delta.removed_edges["source"].values:
        print(metaedit_id)
    if query in delta.added_edges.set_index(["source", "target"]):
        print(metaedit_id)

    if metaedit_id == 31:
        print(delta.added_edges)
    # if query in delta.removed_edges.set_index(["source", "target"]).index:
    #     print("here")
# %%
for metaedit_id in tqdm(
    random_metaedit_ids, desc="Playing meta-edits in random order", disable=True
):
    delta = networkdeltas_by_meta_operation[metaedit_id]
    for diff in diffs:
        if diff in delta.added_nodes.index:
            print("Added node: ", diff)
            print(metaedit_id)
            print(delta.added_edges)
        elif diff in delta.removed_nodes.index:
            print("Removed node: ", diff)
            print(metaedit_id)

# %%
networkdeltas_by_operation[meta_operation_map[32][0]].added_edges


# %%
for metaedit_id in tqdm(
    random_metaedit_ids, desc="Playing meta-edits in random order", disable=True
):
    delta = networkdeltas_by_meta_operation[metaedit_id]

    # nf.add_nodes(delta.added_nodes, inplace=True)
    # nf.add_edges(delta.added_edges, inplace=True)
    # nf.remove_nodes(delta.removed_nodes, inplace=True)
    # nf.remove_edges(delta.removed_edges, inplace=True)


# %%
nuc_edge_index = nuc_nf.edges.set_index(["source", "target"]).index
root_edge_index = root_nf.edges.set_index(["source", "target"]).index

root_edge_index.difference(nuc_edge_index)

# %%


# %%
delta = timedelta(seconds=time.time() - t0)
print("Time elapsed: ", delta)
print()
