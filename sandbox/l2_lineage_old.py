# %%

from datetime import datetime, timedelta

import caveclient as cc


# %%

client = cc.CAVEclient("minnie65_phase3_v1")
cg = client.chunkedgraph
cv = client.info.segmentation_cloudvolume()

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


def get_changed_parent(node_id, timestamp):
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
        return get_changed_parent(pre_parent_id, timestamp)
    else:
        return pre_parent_id, post_parent_id, parent_layer


for operation_id, row in list(merges.iterrows())[:1]:
    detail = cg.get_operation_details([operation_id])[str(operation_id)]
    timestamp = detail["timestamp"]

    source_supervoxel_id = detail["added_edges"][0][0]
    target_supervoxel_id = detail["added_edges"][0][1]

    source_pre_l2_id, source_post_l2_id, source_layer = get_changed_parent(
        source_supervoxel_id, timestamp
    )
    target_pre_l2_id, target_post_l2_id, target_layer = get_changed_parent(
        target_supervoxel_id, timestamp
    )

    print(f"Operation ID: {operation_id}")
    print(f"Source: {source_pre_l2_id} -> {source_post_l2_id} (Level {source_layer})")
    print(f"Target: {target_pre_l2_id} -> {target_post_l2_id} (Level {target_layer})")

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

# # %%
# from anytree import Node


# def build_tree(root_id):
#     children = cg.get_children(root_id)
#     level = cv.get_chunk_layer(root_id)

#     if level == 2:
#         return Node(root_id, supervoxels=children)
#     else:
#         child_nodes = []
#         for child in children:
#             child_node = build_tree(child)
#             child_nodes.append(child_node)
#         root_node = Node(root_id, children=child_nodes)
#         print(level)
#         return root_node


# %%

# operation_id = 339363  # this is a L2-L2 merge

operation_id = 339158  # this is a L3-L3 merge

detail = cg.get_operation_details([operation_id])[str(operation_id)]

source_supervoxel_id = detail["added_edges"][0][0]
target_supervoxel_id = detail["added_edges"][0][1]

source_pre_l2_id, source_post_l2_id, source_layer = get_changed_parent(
    source_supervoxel_id, timestamp
)
target_pre_l2_id, target_post_l2_id, target_layer = get_changed_parent(
    target_supervoxel_id, timestamp
)

print(f"Operation ID: {operation_id}")
print(f"Source: {source_pre_l2_id} -> {source_post_l2_id} (Level {source_layer})")
print(f"Target: {target_pre_l2_id} -> {target_post_l2_id} (Level {target_layer})")


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

    edges = edges.drop_duplicates(keep="first")

    return nodes, edges


source1_nodes, source1_edges = edgelist_to_graph(source1_edgelist)
source2_nodes, source2_edges = edgelist_to_graph(source2_edgelist)
target_nodes, target_edges = edgelist_to_graph(target_edgelist)
# %%
source_index_union = source1_nodes.index.union(source2_nodes.index)

# NOTE: there are L2 nodes that were not in the source graphs, but these are exactly
# the nodes that were added in the merge operation
target_nodes.index.difference(source_index_union)

# %%
# NOTE: likewise, the nodes that were removed in the merge operation are no longer here
source_index_union.difference(target_nodes.index)

# %%
source_combined_edges = pd.concat([source1_edges, source2_edges])
source_combined_edges = source_combined_edges.drop_duplicates()
source_combined_edges["graph"] = "source_combined"

target_edges["graph"] = "target"

# %%

# NOTE: there are edges that changed from source to target
delta_edges = pd.concat([source_combined_edges, target_edges]).drop_duplicates(
    ["source", "target"], keep=False, ignore_index=True
)

id_map = {source_pre_l2_id: source_post_l2_id, target_pre_l2_id: target_post_l2_id}


def remap(x):
    if x in id_map:
        return id_map[x]
    else:
        return x


# but they all have to do with the nodes that were modified, so that's good
delta_edges["source"] = delta_edges["source"].map(remap)
delta_edges["target"] = delta_edges["target"].map(remap)
delta_edges.drop_duplicates(["source", "target"], keep=False, ignore_index=True)

# %%


def get_pre_post_l2_ids(node_id, timestamp):
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

    return pre_parent_id, post_parent_id


source_pre_l2_id, source_post_l2_id = get_pre_post_l2_ids(
    source_supervoxel_id, timestamp
)
target_pre_l2_id, target_post_l2_id = get_pre_post_l2_ids(
    target_supervoxel_id, timestamp
)

print(f"Operation ID: {operation_id}")
print(f"Source: {source_pre_l2_id} -> {source_post_l2_id} (Level 2)")
print(f"Target: {target_pre_l2_id} -> {target_post_l2_id} (Level 2)")
