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
from pkg.edits import NetworkDelta

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

from pkg.paths import OUT_PATH
from cloudfiles import CloudFiles

save = True
reload = True

out_path = OUT_PATH / "store_edit_info"
cf = CloudFiles("file://" + str(out_path))

if save:
    print("Stashing network edits and meta-edits")
    if not out_path.exists():
        out_path.mkdir()

    networkdelta_dicts = {}
    for operation_id, delta in networkdeltas_by_operation.items():
        networkdelta_dicts[operation_id] = delta.to_dict()

    _ = cf.put_json(f"{root_id}_operations.json", networkdelta_dicts)

    networkdelta_dicts = {}
    for meta_operation_id, delta in networkdeltas_by_meta_operation.items():
        networkdelta_dicts[meta_operation_id] = delta.to_dict()

    _ = cf.put_json(f"{root_id}_meta_operations.json", networkdelta_dicts)
    _ = cf.put_json(f"{root_id}_meta_operation_map.json", meta_operation_map)

if reload:
    print("Reloading network edits and meta-edits")
    networkdelta_dicts = cf.get_json(f"{root_id}_operations.json")
    networkdeltas_by_operation = {}
    for operation_id, delta in networkdelta_dicts.items():
        networkdeltas_by_operation[int(operation_id)] = NetworkDelta.from_dict(delta)

    networkdelta_dicts = cf.get_json(f"{root_id}_meta_operations.json")
    networkdeltas_by_meta_operation = {}
    for meta_operation_id, delta in networkdelta_dicts.items():
        networkdeltas_by_meta_operation[
            int(meta_operation_id)
        ] = NetworkDelta.from_dict(delta)


# %%


def apply_edit(network_frame, network_delta):
    network_frame.add_nodes(network_delta.added_nodes, inplace=True)
    network_frame.add_edges(network_delta.added_edges, inplace=True)
    network_frame.remove_nodes(network_delta.removed_nodes, inplace=True)
    network_frame.remove_edges(network_delta.removed_edges, inplace=True)


print("Pulling initial state of the network")
nf = get_initial_network(root_id, client, positions=False)
print()
print()


metaedit_ids = np.array(list(networkdeltas_by_meta_operation.keys()))
np.random.seed(1)
random_metaedit_ids = np.random.permutation(metaedit_ids)
for metaedit_id in tqdm(random_metaedit_ids, desc="Playing meta-edits in random order"):
    metaedit = networkdeltas_by_meta_operation[metaedit_id]
    apply_edit(nf, metaedit)
print()

# %%

print("Finding final fragment with nucleus attached")
nuc_supervoxel = nuc.loc[target_id, "pt_supervoxel_id"]

nuc_nf = find_supervoxel_component(nuc_supervoxel, nf, client)
print()

# %%

print("Checking for correspondence of final edited neuron and original root neuron")
root_nodes, root_edges = get_level2_nodes_edges(root_id, client, positions=False)
root_nf = NetworkFrame(root_nodes, root_edges)

print("L2 graphs match?", root_nf == nuc_nf)
print()


# %%

# there are 2 nodes in "root_nf" that are missing in "nuc_nf"

# %%
delta = timedelta(seconds=time.time() - t0)
print("Time elapsed: ", delta)
print()
