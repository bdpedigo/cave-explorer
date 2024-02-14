# %%
from datetime import datetime, timedelta

import caveclient as cc
import numpy as np
import pandas as pd

from pkg.edits import get_detailed_change_log

client = cc.CAVEclient("minnie65_phase3_v1")

seg_res = np.array(client.chunkedgraph.segmentation_info["scales"][0]["resolution"])

root_id = 864691135737446276

operation_id = 202841

# change_log = client.chunkedgraph.get_tabular_change_log(root_id)[root_id].set_index(
#     "operation_id"
# )
# details = client.chunkedgraph.get_operation_details([operation_id])[str(operation_id)]


change_log = get_detailed_change_log(root_id, client)
details = change_log.loc[operation_id]


point_in_cg = 0.5 * np.mean(details["sink_coords"], axis=0) + 0.5 * np.mean(
    details["source_coords"], axis=0
)

# %%


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


bbox_cg = get_bbox_cg(point_in_cg)

# %%

operation_timestamp = details["timestamp"]
post_timestamp = datetime.fromisoformat(operation_timestamp) + timedelta(microseconds=1)
pre_timestamp = datetime.fromisoformat(operation_timestamp) - timedelta(microseconds=1)

pre_l2_edges = []
for before_id in details["before_root_ids"]:
    l1_edges, affinities, areas = client.chunkedgraph.get_subgraph(before_id, bbox_cg)

    l1_edges = pd.DataFrame(l1_edges, columns=["source", "target"])

    l2_edges = map_l1_to_l2_edges(l1_edges, pre_timestamp, client)
    pre_l2_edges.append(l2_edges)
pre_l2_edges = pd.concat(pre_l2_edges, axis=0)

post_l2_edges = []
for after_id in details["roots"]:
    l1_edges, affinities, areas = client.chunkedgraph.get_subgraph(after_id, bbox_cg)

    l1_edges = pd.DataFrame(l1_edges, columns=["source", "target"])

    l2_edges = map_l1_to_l2_edges(l1_edges, post_timestamp, client)
    post_l2_edges.append(l2_edges)
post_l2_edges = pd.concat(post_l2_edges, axis=0)


pre_l2_edges = pre_l2_edges.set_index(["source", "target"]).index
post_l2_edges = post_l2_edges.set_index(["source", "target"]).index

removed_edges_v1 = pre_l2_edges.difference(post_l2_edges).to_frame()
added_edges_v1 = post_l2_edges.difference(pre_l2_edges).to_frame()
print(len(removed_edges_v1), len(added_edges_v1))
# %%
removed_edges_v1
# %%
added_edges_v1
# %%
before_root_ids = details["before_root_ids"]
after_root_ids = details["after_root_ids"]

from pkg.edits import get_changed_edges
from pkg.utils import get_all_nodes_edges

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
removed_edges_v2, added_edges_v2 = get_changed_edges(all_before_edges, all_after_edges)
print(len(removed_edges_v2), len(added_edges_v2))
# %%
removed_edges_v2
# %%
added_edges_v2
