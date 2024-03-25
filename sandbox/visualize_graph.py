# %%

from pathlib import Path

import caveclient as cc

from graspologic.plot import networkplot

import matplotlib.pyplot as plt
import seaborn as sns


FIG_PATH = Path("cave-explorer/results/figs/visualize_edits")

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
n_show = 20
sub_meta = meta.sample(n_show, random_state=0)

# %%
i = 1
target_id = sub_meta.iloc[i]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
latest = client.chunkedgraph.get_latest_roots(root_id)
assert len(latest) == 1
root_id = latest[0]

change_log = client.chunkedgraph.get_change_log(root_id)
operation_ids = change_log["operations_ids"]
details = client.chunkedgraph.get_operation_details(change_log["operations_ids"])

# %%
change_log = cg.get_tabular_change_log(root_id)[root_id]

# %%
# roots: the new root node that was created from this operation (?)
# added_edges: the edges that were added to the L2 (?) graph
# source/sink coords: the coordinates of the edit...

detail = details[str(operation_ids[0])]

pre_supervoxel_id = detail["added_edges"][0][0]
post_supervoxel_id = detail["added_edges"][0][1]

# %%
# is a node...
client.chunkedgraph.is_valid_nodes([pre_supervoxel_id])[0]

# %%

# is a level 1 node (supervoxel)
cv = client.info.segmentation_cloudvolume()
cv.get_chunk_layer(pre_supervoxel_id)

# %%

# go up one level in the chunkedgraph
pre_l2_root = client.chunkedgraph.get_root_id(pre_supervoxel_id, level2=True)
post_l2_root = client.chunkedgraph.get_root_id(post_supervoxel_id, level2=True)

pre_l2_root == post_l2_root

# this was a merge, so this makes sense

# %%
import datetime

# get lineage graph is only for roots
client.chunkedgraph.get_lineage_graph(
    pre_l2_root, timestamp_past=datetime.datetime(2000, 12, 1)
)

# %%

detail = details[str(operation_ids[1])]

# this happens to be a split
pre_supervoxel_id = detail["removed_edges"][0][0]
post_supervoxel_id = detail["removed_edges"][0][1]

# go up one level in the chunkedgraph
pre_l2_root = client.chunkedgraph.get_root_id(pre_supervoxel_id, level2=True)
post_l2_root = client.chunkedgraph.get_root_id(post_supervoxel_id, level2=True)

# now these are not the same
pre_l2_root == post_l2_root

# %%
leaves = client.chunkedgraph.get_leaves(pre_l2_root)
pre_supervoxel_id in leaves

# %%

merge_log_out = cg.get_merge_log(root_id)

# %%

change_log = cg.get_tabular_change_log(root_id)[root_id]
change_log.set_index("operation_id", inplace=True)

# %%
merges = change_log.query("is_merge")

for operation_id, row in merges.iterrows():
    detail = cg.get_operation_details([operation_id])[str(operation_id)]

# %%
source1, source2 = row["before_root_ids"]
target = row["after_root_ids"][0]

source1_edgelist = cg.level2_chunk_graph(source1)
source2_edgelist = cg.level2_chunk_graph(source2)
target_edgelist = cg.level2_chunk_graph(target)


# %%

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
# no overlap in node sets (expected, these were distinct before the merge)
source1_nodes.index.isin(source2_nodes.index).any()

# %%
source_node_ids_union = source1_nodes.index.union(source2_nodes.index)

# %%
target_nodes[~target_nodes.index.isin(source_node_ids_union)]

# %%
new_l2_id = target_nodes[~target_nodes.index.isin(source_node_ids_union)].index[0]

# %%
timestamp = detail["timestamp"]

from datetime import datetime, timedelta

operation_time = datetime.fromisoformat(timestamp)

delta = timedelta(microseconds=1)

pre_operation_time = operation_time - delta
post_operation_time = operation_time + delta

source_supervoxel_id = detail["added_edges"][0][0]
target_supervoxel_id = detail["added_edges"][0][1]

# %%
source_pre_l2_id = cg.get_roots(
    source_supervoxel_id, timestamp=pre_operation_time, stop_layer=2
)[0]

# %%
target_pre_l2_id = cg.get_roots(
    target_supervoxel_id, timestamp=pre_operation_time, stop_layer=2
)[0]

# %%
source_post_l2_id = cg.get_roots(
    source_supervoxel_id, timestamp=post_operation_time, stop_layer=2
)[0]

# %%
target_post_l2_id = cg.get_roots(
    target_supervoxel_id, timestamp=post_operation_time, stop_layer=2
)[0]

# %%
from datetime import datetime, timedelta

import networkx as nx

g = nx.MultiDiGraph()

for operation_id, row in list(merges.iterrows())[:20]:
    detail = cg.get_operation_details([operation_id])[str(operation_id)]
    timestamp = detail["timestamp"]

    # TODO make sure that 1 microsecond is a good resolution
    # it seemed that smaller than this wasn't showing the deltas
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
    print(f"Source: {source_pre_l2_id} -> {source_post_l2_id}")
    print(f"Target: {target_pre_l2_id} -> {target_post_l2_id}")
    print()

    g.add_edge(
        source_pre_l2_id,
        source_post_l2_id,
        operation_id=operation_id,
        timestamp=timestamp,
        is_merge=True,
    )
    g.add_edge(
        target_pre_l2_id,
        target_post_l2_id,
        operation_id=operation_id,
        timestamp=timestamp,
        is_merge=True,
    )

# %%
