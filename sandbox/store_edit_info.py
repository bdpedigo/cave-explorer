# %%

import time

t0 = time.time()

from datetime import timedelta

import caveclient as cc
import numpy as np
from cloudfiles import CloudFiles
from networkframe import NetworkFrame
from tqdm.auto import tqdm

from pkg.edits import (
    NetworkDelta,
    find_supervoxel_component,
    get_initial_network,
    get_network_edits,
    get_network_metaedits,
)
from pkg.paths import OUT_PATH
from pkg.utils import get_level2_nodes_edges

# %%
recompute = False
cloud = False

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
for i in range(0, 10):
    target_id = meta.iloc[i]["target_id"]
    root_id = nuc.loc[target_id]["pt_root_id"]
    root_id = client.chunkedgraph.get_latest_roots(root_id)[0]

    if cloud:
        out_path = "allen-minnie-phase3/edit_info"
        cf = CloudFiles("gs://" + out_path)
    else:
        out_path = OUT_PATH / "store_edit_info"
        cf = CloudFiles("file://" + str(out_path))

    out_file = f"{root_id}_operations.json"

    if not cf.exists(out_file) or recompute:
        print("Pulling network edits")
        networkdeltas_by_operation = get_network_edits(root_id, client, filtered=False)

        networkdelta_dicts = {}
        for operation_id, delta in networkdeltas_by_operation.items():
            networkdelta_dicts[operation_id] = delta.to_dict()

        _ = cf.put_json(out_file, networkdelta_dicts)
        print()

    print("Reloading network edits")
    networkdelta_dicts = cf.get_json(out_file)
    networkdeltas_by_operation = {}
    for operation_id, delta in networkdelta_dicts.items():
        networkdeltas_by_operation[int(operation_id)] = NetworkDelta.from_dict(delta)

    print()

    out_file = f"{root_id}_meta_operations.json"

    if not cf.exists(out_file) or recompute:
        print("Compiling meta-edits")
        networkdeltas_by_meta_operation, meta_operation_map = get_network_metaedits(
            networkdeltas_by_operation
        )

        # remap all of the keys to strings
        networkdelta_dicts = {}
        for meta_operation_id, delta in networkdeltas_by_meta_operation.items():
            networkdelta_dicts[str(meta_operation_id)] = delta.to_dict()
        out_meta_operation_map = {}
        for meta_operation_id, operation_ids in meta_operation_map.items():
            out_meta_operation_map[str(meta_operation_id)] = operation_ids

        _ = cf.put_json(out_file, networkdelta_dicts)
        _ = cf.put_json(f"{root_id}_meta_operation_map.json", out_meta_operation_map)

        print()

    print("Reloading meta-edits")
    networkdelta_dicts = cf.get_json(out_file)
    networkdeltas_by_meta_operation = {}
    for meta_operation_id, delta in networkdelta_dicts.items():
        networkdeltas_by_meta_operation[
            int(meta_operation_id)
        ] = NetworkDelta.from_dict(delta)
    in_meta_operation_map = cf.get_json(f"{root_id}_meta_operation_map.json")
    meta_operation_map = {}
    for meta_operation_id, operation_ids in in_meta_operation_map.items():
        meta_operation_map[int(meta_operation_id)] = operation_ids

    print()

    def apply_edit(network_frame, network_delta):
        network_frame.add_nodes(network_delta.added_nodes, inplace=True)
        network_frame.add_edges(network_delta.added_edges, inplace=True)
        network_frame.remove_nodes(network_delta.removed_nodes, inplace=True)
        network_frame.remove_edges(network_delta.removed_edges, inplace=True)

    # TODO could make this JSON serializable
    nf = get_initial_network(root_id, client, positions=False)
    print()

    metaedit_ids = np.array(list(networkdeltas_by_meta_operation.keys()))
    random_metaedit_ids = np.random.permutation(metaedit_ids)
    for metaedit_id in tqdm(
        random_metaedit_ids, desc="Playing meta-edits in random order"
    ):
        metaedit = networkdeltas_by_meta_operation[metaedit_id]
        apply_edit(nf, metaedit)
    print()

    print("Finding final fragment with nucleus attached")
    nuc_supervoxel = nuc.loc[target_id, "pt_supervoxel_id"]

    nuc_nf = find_supervoxel_component(nuc_supervoxel, nf, client)
    print()

    root_nodes, root_edges = get_level2_nodes_edges(root_id, client, positions=False)
    root_nf = NetworkFrame(root_nodes, root_edges)

    assert root_nf == nuc_nf
    print()

# %%
delta = timedelta(seconds=time.time() - t0)
print("Time elapsed: ", delta)
print()
