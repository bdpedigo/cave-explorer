# %%

import time

t0 = time.time()

from datetime import timedelta

import caveclient as cc
import numpy as np
from pkg.edits import (
    find_supervoxel_component,
    get_initial_network,
    get_network_edits,
    get_network_metaedits,
)
from pkg.utils import get_level2_nodes_edges
from tqdm.autonotebook import tqdm
from pkg.edits import get_lineage_tree
from anytree import PreOrderIter

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

print("Finding meta-operations")
networkdeltas_by_meta_operation, meta_operation_map = get_network_metaedits(
    networkdeltas_by_operation, edit_lineage_graph
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


diffs = root_nf.nodes.index.difference(nuc_nf.nodes.index)
print(diffs)

#%%



# %%
for metaedit_id in tqdm(
    random_metaedit_ids, desc="Playing meta-edits in random order", disable=True
):
    delta = networkdeltas_by_meta_operation[metaedit_id]
    for diff in diffs:
        if diff in delta.added_nodes.index:
            print("Added node: ", diff)
            print(metaedit_id)
        elif diff in delta.removed_nodes.index:
            print("Removed node: ", diff)
            print(metaedit_id)

#%%
networkdeltas_by_operation[meta_operation_map[32][0]]

    
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
