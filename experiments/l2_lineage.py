# %%

from datetime import datetime, timedelta

import caveclient as cc
import networkx as nx


# %%

client = cc.CAVEclient("minnie65_phase3_v1")
cg = client.chunkedgraph

# %%

# NOTE: the lowest level segmentation is done in 8x8x40,
# not 4x4x40 (client.info.viewer_resolution())
seg_res = client.chunkedgraph.segmentation_info["scales"][0]["resolution"]
res = client.info.viewer_resolution()

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
i = 1
target_id = meta.iloc[i]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
latest = client.chunkedgraph.get_latest_roots(root_id)
assert len(latest) == 1
root_id = latest[0]

print("Root ID:", root_id)

# %%

change_log = cg.get_tabular_change_log(root_id)[root_id]
change_log.set_index("operation_id", inplace=True)
change_log.sort_values("timestamp", inplace=True)
merges = change_log.query("is_merge")

# %%

cv = client.info.segmentation_cloudvolume()


def get_changed_parent(node_id, timestamp):
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
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
        return get_changed_parent(pre_parent_id, timestamp)
    else:
        return pre_parent_id, post_parent_id, parent_layer


for operation_id, row in list(merges.iterrows())[:30]:
    detail = cg.get_operation_details([operation_id])[str(operation_id)]
    timestamp = detail["timestamp"]

    # TODO make sure that 1 microsecond is a good resolution
    
    source_supervoxel_id = detail["added_edges"][0][0]
    target_supervoxel_id = detail["added_edges"][0][1]

    source_pre_l2_id = cg.get_roots(
        source_supervoxel_id, timestamp=pre_operation_time, stop_layer=2
    )[0]

    target_pre_l2_id = cg.get_roots(
        target_supervoxel_id, timestamp=pre_operation_time, stop_layer=2
    )[0]

    source_post_l2_id = cg.get_roots(
        source_supervoxel_id, timestamp=post_operation_time, stop_layer=2
    )[0]

    target_post_l2_id = cg.get_roots(
        target_supervoxel_id, timestamp=post_operation_time, stop_layer=2
    )[0]

    print(f"Operation ID: {operation_id}")
    print(f"Source: {source_pre_l2_id} -> {source_post_l2_id}")
    print(f"Target: {target_pre_l2_id} -> {target_post_l2_id}")

    # g.add_edge(
    #     source_pre_l2_id,
    #     source_post_l2_id,
    #     operation_id=operation_id,
    #     timestamp=timestamp,
    #     is_merge=True,
    # )
    # g.add_edge(
    #     target_pre_l2_id,
    #     target_post_l2_id,
    #     operation_id=operation_id,
    #     timestamp=timestamp,
    #     is_merge=True,
    # )

    print()

# %%
operation_id = 339158
detail = cg.get_operation_details([operation_id])[str(operation_id)]
timestamp = detail["timestamp"]

# TODO make sure that 1 microsecond is a good resolution
delta = timedelta(microseconds=1)
operation_time = datetime.fromisoformat(timestamp)
pre_operation_time = operation_time - delta
post_operation_time = operation_time + delta

source_supervoxel_id = detail["added_edges"][0][0]
target_supervoxel_id = detail["added_edges"][0][1]

source_pre_l2_id = cg.get_roots(
    source_supervoxel_id, timestamp=pre_operation_time, stop_layer=2
)[0]

target_pre_l2_id = cg.get_roots(
    target_supervoxel_id, timestamp=pre_operation_time, stop_layer=2
)[0]

source_post_l2_id = cg.get_roots(
    source_supervoxel_id, timestamp=post_operation_time, stop_layer=2
)[0]

target_post_l2_id = cg.get_roots(
    target_supervoxel_id, timestamp=post_operation_time, stop_layer=2
)[0]

print(f"Operation ID: {operation_id}")
print("Layer 2:")
print(f"Source: {source_pre_l2_id} -> {source_post_l2_id}")
print(f"Target: {target_pre_l2_id} -> {target_post_l2_id}")

# %%
if False:
    source_pre_l2_id = cg.get_roots(
        source_supervoxel_id, timestamp=pre_operation_time, stop_layer=3
    )[0]

    target_pre_l2_id = cg.get_roots(
        target_supervoxel_id, timestamp=pre_operation_time, stop_layer=3
    )[0]

    source_post_l2_id = cg.get_roots(
        source_supervoxel_id, timestamp=post_operation_time, stop_layer=3
    )[0]

    target_post_l2_id = cg.get_roots(
        target_supervoxel_id, timestamp=post_operation_time, stop_layer=3
    )[0]

    print("Layer 3:")
    print(f"Source: {source_pre_l2_id} -> {source_post_l2_id}")
    print(f"Target: {target_pre_l2_id} -> {target_post_l2_id}")

# %%
cg.get_children(source_pre_l2_id)

# %%

row = merges.loc[operation_id]
source1, source2 = row["before_root_ids"]
target = row["after_root_ids"][0]

source1_edgelist = cg.level2_chunk_graph(source1)
source2_edgelist = cg.level2_chunk_graph(source2)
target_edgelist = cg.level2_chunk_graph(target)


import pandas as pd


def pt_to_xyz(pts):
    name = pts.name
    idx_name = pts.index.name
    if idx_name is None:
        idx_name = "index"
    positions = pts.explode().reset_index()

    def to_xyz(order):
        if order % 3 == 0:
            return "x"
        elif order % 3 == 1:
            return "y"
        else:
            return "z"

    positions["axis"] = positions.index.map(to_xyz)
    positions = positions.pivot(index=idx_name, columns="axis", values=name)

    return positions


def edgelist_to_graph(edgelist):
    nodelist = set()
    for edge in edgelist:
        for node in edge:
            nodelist.add(node)
    nodelist = list(nodelist)

    l2stats = client.l2cache.get_l2data(nodelist, attributes=["rep_coord_nm"])
    nodes = pd.DataFrame(l2stats).T
    positions = pt_to_xyz(nodes["rep_coord_nm"])
    nodes = pd.concat([nodes, positions], axis=1)
    nodes.index = nodes.index.astype(int)
    nodes.index.name = "l2_id"

    edges = pd.DataFrame(edgelist)
    edges.columns = ["source", "target"]

    return nodes, edges


source1_nodes, source1_edges = edgelist_to_graph(source1_edgelist)
source2_nodes, source2_edges = edgelist_to_graph(source2_edgelist)
target_nodes, target_edges = edgelist_to_graph(target_edgelist)
# %%
source_index_union = source1_nodes.index.union(source2_nodes.index)
target_nodes.index.difference(source_index_union)

# %%
target_edges.query("target == 160807562097198220")


# %%
cv = client.info.segmentation_cloudvolume()
cv.get_chunk_layer(160807562097197861)

# %%


dir(cv)

# %%
chunk_mappings = cv.get_chunk_mappings(160807562097197861)
# %%
mesh = cv.mesh.get(160807562097197861)[160807562097197861]

# %%
l2_id = 160807562097197861
supervoxels = cg.get_children(l2_id)
cv.mesh.get(supervoxels)

# %%
dir(cv.image.download())
# %%
mesh = cv.mesh.get(l2_id)[l2_id]
verts = mesh.vertices
bounds = verts.min(axis=0), verts.max(axis=0)
# %%
cv.image.download()
