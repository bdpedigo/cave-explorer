# %%

from datetime import datetime, timedelta

import caveclient as cc
import numpy as np
import pandas as pd

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
splits = change_log.query("~is_merge")

details = cg.get_operation_details(merges.index.to_list())
details = pd.DataFrame(details).T
details.index.name = "operation_id"
details.index = details.index.astype(int)
details = details.explode("roots")

merges = merges.join(details)


# %%
def get_pre_post_l2_ids(node_id, timestamp):
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)

    # 1 microsecond is the finest resolution allowed by timedelta
    delta = timedelta(microseconds=1)
    pre_operation_time = timestamp - delta
    post_operation_time = timestamp + delta

    pre_parent_id = cg.get_roots(node_id, timestamp=pre_operation_time, stop_layer=2)[0]
    post_parent_id = cg.get_roots(node_id, timestamp=post_operation_time, stop_layer=2)[
        0
    ]

    return pre_parent_id, post_parent_id


def get_changed_ancestor(node_id, timestamp):
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)

    # 1 microsecond is the finest resolution allowed by timedelta
    delta = timedelta(microseconds=1)
    pre_operation_time = timestamp - delta
    post_operation_time = timestamp + delta

    current_layer = cv.get_chunk_layer(node_id)
    parent_layer = current_layer + 1

    pre_parent_id = cg.get_roots(
        node_id, timestamp=pre_operation_time, stop_layer=parent_layer
    )[0]
    post_parent_id = cg.get_roots(
        node_id, timestamp=post_operation_time, stop_layer=parent_layer
    )[0]

    if pre_parent_id == post_parent_id:
        return get_changed_ancestor(pre_parent_id, timestamp)
    else:
        return pre_parent_id, post_parent_id, parent_layer


# %%
operation_id = 339158
row = merges.loc[operation_id]

source_supervoxel_id = row["added_edges"][0][0]
target_supervoxel_id = row["added_edges"][0][1]
timestamp = row["timestamp"]

source_pre_l2_id, source_post_l2_id = get_pre_post_l2_ids(
    source_supervoxel_id, timestamp
)
target_pre_l2_id, target_post_l2_id = get_pre_post_l2_ids(
    target_supervoxel_id, timestamp
)

print(f"Operation ID: {operation_id}")
print(f"Source: {source_pre_l2_id} -> {source_post_l2_id} (Level 2)")
print(f"Target: {target_pre_l2_id} -> {target_post_l2_id} (Level 2)")

# %%
source_pre_l2_id, source_post_l2_id, source_layer = get_changed_ancestor(
    source_supervoxel_id, timestamp
)
target_pre_l2_id, target_post_l2_id, target_layer = get_changed_ancestor(
    target_supervoxel_id, timestamp
)

print(f"Operation ID: {operation_id}")
print(f"Source: {source_pre_l2_id} -> {source_post_l2_id} (Level {source_layer})")
print(f"Target: {target_pre_l2_id} -> {target_post_l2_id} (Level {target_layer})")

# %%

before1_root_id, before2_root_id = row["before_root_ids"]
after_root_id = row["after_root_ids"][0]

before1_nodes = cg.get_leaves(before1_root_id, stop_layer=2)
before2_nodes = cg.get_leaves(before2_root_id, stop_layer=2)
after_nodes = cg.get_leaves(after_root_id, stop_layer=2)

before_union = np.union1d(before1_nodes, before2_nodes)

np.setdiff1d(after_nodes, before_union)
