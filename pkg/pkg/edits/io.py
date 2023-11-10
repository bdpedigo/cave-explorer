import os

import pandas as pd
from cloudfiles import CloudFiles
from neuropull.graph import NetworkFrame

from ..edits import (
    NetworkDelta,
    get_initial_network,
    get_network_edits,
    get_network_metaedits,
)
from ..paths import OUT_PATH
from .changes import get_supervoxel_level2_map


def get_environment_variables():
    cloud = os.environ.get("SKEDITS_USE_CLOUD") == "True"
    recompute = os.environ.get("SKEDITS_RECOMPUTE") == "True"
    return cloud, recompute


def get_cloud_paths(cloud):
    if cloud:
        out_path = "allen-minnie-phase3/edit_info"
        cf = CloudFiles("gs://" + out_path)
    else:
        out_path = OUT_PATH / "store_edit_info"
        cf = CloudFiles("file://" + str(out_path))
    return cf


def lazy_load_network_edits(root_id, client):
    cloud, recompute = get_environment_variables()
    cf = get_cloud_paths(cloud)

    out_file = f"{root_id}_operations.json"

    if not cf.exists(out_file) or recompute:
        networkdeltas_by_operation = get_network_edits(root_id, client, filtered=False)

        networkdelta_dicts = {}
        for operation_id, delta in networkdeltas_by_operation.items():
            networkdelta_dicts[operation_id] = delta.to_dict()

        _ = cf.put_json(out_file, networkdelta_dicts)
        assert cf.exists(out_file)

    networkdelta_dicts = cf.get_json(out_file)
    networkdeltas_by_operation = {}
    for operation_id, delta in networkdelta_dicts.items():
        networkdeltas_by_operation[int(operation_id)] = NetworkDelta.from_dict(delta)

    out_file = f"{root_id}_meta_operations.json"

    if not cf.exists(out_file) or recompute:
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
        assert cf.exists(out_file)
        _ = cf.put_json(f"{root_id}_meta_operation_map.json", out_meta_operation_map)
        assert cf.exists(f"{root_id}_meta_operation_map.json")

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

    return networkdeltas_by_operation, networkdeltas_by_meta_operation


def lazy_load_initial_network(root_id, client, positions=True):
    cloud, recompute = get_environment_variables()
    cf = get_cloud_paths(cloud)

    out_file = f"{root_id}_initial_network.json"
    if not cf.exists(out_file) or recompute:
        initial_network = get_initial_network(root_id, client, positions=positions)

        cf.put_json(out_file, initial_network.to_dict())

    initial_network_dict = cf.get_json(out_file)
    initial_network = NetworkFrame.from_dict(initial_network_dict)

    return initial_network


def lazy_load_supervoxel_level2_map(root_id, networkdeltas_by_operation, client):
    cloud, recompute = get_environment_variables()
    cf = get_cloud_paths(cloud)

    out_file = f"{root_id}_supervoxel_level2_map.json"
    if not cf.exists(out_file) or recompute:
        supervoxel_map = get_supervoxel_level2_map(
            root_id, networkdeltas_by_operation, client
        )
        cf.put_json(out_file, supervoxel_map.to_dict())

    supervoxel_map = cf.get_json(out_file)
    supervoxel_map = pd.Series(supervoxel_map)
    supervoxel_map.index = supervoxel_map.index.astype(int)
    return supervoxel_map
