# %%
import time
from datetime import datetime, timedelta

import caveclient as cc
import numpy as np
import pandas as pd

from pkg.edits import NetworkDelta, get_changed_edges, get_detailed_change_log
from pkg.utils import get_all_nodes_edges

# %%


def get_network_delta_for_operation(operation_details):
    before_root_ids = operation_details["before_root_ids"]
    after_root_ids = operation_details["roots"]

    # grabbing the union of before/after nodes/edges
    # this works for merge or split
    # NOTE: this is where all the time comes from
    currtime = time.time()
    all_before_nodes, all_before_edges = get_all_nodes_edges(
        before_root_ids, client, positions=False
    )
    all_after_nodes, all_after_edges = get_all_nodes_edges(
        after_root_ids, client, positions=False
    )
    print(f"{time.time() - currtime:.3f} seconds elapsed to hit level2_chunk_graph().")

    # finding the nodes that were added or removed, simple set logic
    added_nodes_index = all_after_nodes.index.difference(all_before_nodes.index)
    added_nodes = all_after_nodes.loc[added_nodes_index]
    removed_nodes_index = all_before_nodes.index.difference(all_after_nodes.index)
    removed_nodes = all_before_nodes.loc[removed_nodes_index]

    # finding the edges that were added or removed, simple set logic again
    removed_edges, added_edges = get_changed_edges(all_before_edges, all_after_edges)

    # keep track of what changed
    metadata = {
        **operation_details.to_dict(),
        "operation_id": operation_id,
        "root_id": root_id,
        "n_added_nodes": len(added_nodes),
        "n_removed_nodes": len(removed_nodes),
        "n_modified_nodes": len(added_nodes) + len(removed_nodes),
        "n_added_edges": len(added_edges),
        "n_removed_edges": len(removed_edges),
        "n_modified_edges": len(added_edges) + len(removed_edges),
    }

    return NetworkDelta(
        removed_nodes, added_nodes, removed_edges, added_edges, metadata=metadata
    )


client = cc.CAVEclient("minnie65_phase3_v1")

root_id = 864691135697251738

change_log = get_detailed_change_log(root_id, client, filtered=False)

# %%
# running list of example cases:
# split operation where added edges and removed edges are both empty: 100606
# split operation where removed edges is not empty (i think this gets "overwritten" later): 157790

# %%
change_log[~change_log["is_merge"]]

# %%%
operation_id = 390649
details = change_log.loc[operation_id]

# %%
delta = get_network_delta_for_operation(details)

delta

currtime = time.time()
before_root_ids = details["before_root_ids"]
after_root_ids = details["roots"]
pre_l2_nodes = []
for root in before_root_ids:
    pre_l2_nodes += list(client.chunkedgraph.get_leaves(root, stop_layer=2))
pre_l2_nodes = pd.Index(pre_l2_nodes)

post_l2_nodes = []
for root in after_root_ids:
    post_l2_nodes += list(client.chunkedgraph.get_leaves(root, stop_layer=2))
post_l2_nodes = pd.Index(post_l2_nodes)
print(f"{time.time() - currtime:.3f} seconds elapsed to hit get_leaves().")
# %%

removed_nodes = pre_l2_nodes.difference(post_l2_nodes)
added_nodes = post_l2_nodes.difference(pre_l2_nodes)

print(removed_nodes)
print(added_nodes)

# %%
currtime = time.time()
client.chunkedgraph.get_leaves(root_id, stop_layer=2)

print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%


seg_res = np.array(client.chunkedgraph.segmentation_info["scales"][0]["resolution"])


def get_bbox_cg(point_in_cg, bbox_halfwidth=10_000):
    point_in_nm = point_in_cg * seg_res

    x_center, y_center, z_center = point_in_nm

    x_start = x_center - bbox_halfwidth
    x_stop = x_center + bbox_halfwidth
    y_start = y_center - bbox_halfwidth
    y_stop = y_center + bbox_halfwidth
    z_start = z_center - bbox_halfwidth
    z_stop = z_center + bbox_halfwidth

    start_point_cg = np.round(np.array([x_start, y_start, z_start]) / seg_res)
    stop_point_cg = np.round(np.array([x_stop, y_stop, z_stop]) / seg_res)

    bbox_cg = np.array([start_point_cg, stop_point_cg], dtype=int).T

    return bbox_cg


point_in_cg = 0.5 * np.mean(details["sink_coords"], axis=0) + 0.5 * np.mean(
    details["source_coords"], axis=0
)

bbox = get_bbox_cg(point_in_cg, bbox_halfwidth=2_500)

currtime = time.time()
before_root_ids = details["before_root_ids"]
after_root_ids = details["roots"]
pre_l2_nodes = []
for root in before_root_ids:
    pre_l2_nodes += list(
        client.chunkedgraph.get_leaves(root, stop_layer=2, bounds=bbox)
    )
pre_l2_nodes = pd.Index(pre_l2_nodes)

post_l2_nodes = []
for root in after_root_ids:
    post_l2_nodes += list(
        client.chunkedgraph.get_leaves(root, stop_layer=2, bounds=bbox)
    )
post_l2_nodes = pd.Index(post_l2_nodes)
print(f"{time.time() - currtime:.3f} seconds elapsed to hit get_leaves() with bbox.")


# %%

removed_nodes = pre_l2_nodes.difference(post_l2_nodes)
added_nodes = post_l2_nodes.difference(pre_l2_nodes)

print(removed_nodes)
print(added_nodes)

#####################

#%%
delta.removed_edges

#%%
delta.added_edges

# %%

final_edges = client.chunkedgraph.level2_chunk_graph(root_id)
final_edges = pd.DataFrame(final_edges, columns=["source", "target"])
final_edges = final_edges.set_index(["source", "target"], drop=False)
final_node_index = np.unique(final_edges[["source", "target"]].values)
final_nodes = pd.DataFrame(index=final_node_index)

# %%


def map_l1_to_l2_edges(l1_edges, timestamp, client):
    uni_supervoxel_nodes = np.unique(l1_edges.values)

    l2_nodes = client.chunkedgraph.get_roots(
        uni_supervoxel_nodes, stop_layer=2, timestamp=timestamp
    )

    supervoxel_to_l2_map = dict(zip(uni_supervoxel_nodes, l2_nodes))

    l2_edges = l1_edges.replace(supervoxel_to_l2_map)

    l2_edges.drop_duplicates(keep="first", inplace=True)

    l2_edges = l2_edges[(l2_edges["source"] != l2_edges["target"])]

    return l2_edges


operation_timestamp = details["timestamp"]
post_timestamp = datetime.fromisoformat(operation_timestamp) + timedelta(microseconds=1)
pre_timestamp = datetime.fromisoformat(operation_timestamp) - timedelta(microseconds=1)
removed_edges = details["removed_edges"]
removed_edges = pd.DataFrame(removed_edges, columns=["source", "target"])
l2_edges = map_l1_to_l2_edges(removed_edges, pre_timestamp, client)


# %%
l1_edges = removed_edges
timestamp = pre_timestamp
uni_supervoxel_nodes = np.unique(l1_edges.values)

l2_nodes = client.chunkedgraph.get_roots(
    uni_supervoxel_nodes, stop_layer=2, timestamp=timestamp
)

supervoxel_to_l2_map = dict(zip(uni_supervoxel_nodes, l2_nodes))

l2_edges = l1_edges.replace(supervoxel_to_l2_map)
l2_edges
# %%
l2_edges.drop_duplicates(keep="first", inplace=True)

l2_edges
# %%
