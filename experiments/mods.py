# %%
from itertools import pairwise

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anytree import Node, Walker
from anytree.iterators import PreOrderIter
from requests import HTTPError
from tqdm.autonotebook import tqdm
import pcg_skel
import skeleton_plot

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")
target_id = meta.iloc[26]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
latest = client.chunkedgraph.get_latest_roots(root_id)
assert len(latest) == 1
root_id = latest[0]

# %%
# root_id = 864691135510015497
# root_id = client.chunkedgraph.get_latest_roots(root_id)[0]

# %%

lineage_graph = client.chunkedgraph.get_lineage_graph(root_id)
links = lineage_graph["links"]

len(lineage_graph["nodes"])

# %%
change_log = client.chunkedgraph.get_change_log(root_id)

# %%

details = client.chunkedgraph.get_operation_details(change_log["operations_ids"])

merges = {}
splits = {}
for operation, detail in details.items():
    operation = int(operation)
    source_coords = detail["source_coords"][0]
    sink_coords = detail["sink_coords"][0]
    # x = (source_coords[0] + sink_coords[0]) / 2
    # y = (source_coords[1] + sink_coords[1]) / 2
    # z = (source_coords[2] + sink_coords[2]) / 2
    x = source_coords[0]
    y = source_coords[1]
    z = source_coords[2]
    pt = [x, y, z]
    row = {"x": x, "y": y, "z": z, "pt": pt}
    if "added_edges" in detail:
        merges[operation] = row
    elif "removed_edges" in detail:
        splits[operation] = row

import pandas as pd

merges = pd.DataFrame.from_dict(merges, orient="index")
merges.index.name = "operation"
splits = pd.DataFrame.from_dict(splits, orient="index")
splits.index.name = "operation"
splits

# %%
detail
edges = detail["added_edges"]
for edge in edges:
    pre_id, post_id = edge

# %%
[print(detail["roots"]) for detail in list(details.values())[:5]]

for detail in list(details.values())[:20]:
    if "added_edges" in detail:
        edges = detail["added_edges"]
    else: 
        edges = detail["removed_edges"]
    for edge in edges:

        # print(client.l2cache.get_l2data(edge, attributes=["rep_coord_nm"]))

#%%
supervoxel_ids = client.chunkedgraph.get_leaves(root_id)
#%%
client.l2cache.get_l2data(supervoxel_ids[:100])

# %%
client.chunkedgraph.get_roots([pre_id])

# %%
from time import sleep

for i in range(10):
    print(client.l2cache.get_l2data([pre_id, post_id], attributes=["rep_coord_nm"]))
    # sleep(10)

# %%


# %%

res = client.info.viewer_resolution()
splits[["x", "y", "z"]] = splits[["x", "y", "z"]] * res
merges[["x", "y", "z"]] = merges[["x", "y", "z"]] * res

# %%

import pcg_skel

meshwork = pcg_skel.coord_space_meshwork(
    root_id,
    client=client,
    synapses="all",
    synapse_table=client.materialize.synapse_table,
)

# %%


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


pt_to_xyz(meshwork.anno["pre_syn"]["ctr_pt_position"])


# %%
def summarize(arr):
    xmin = arr[:, 0].min()
    xmax = arr[:, 0].max()
    ymin = arr[:, 1].min()
    ymax = arr[:, 1].max()
    zmin = arr[:, 2].min()
    zmax = arr[:, 2].max()

    print("x: ", xmin, "-", xmax)
    print("y: ", ymin, "-", ymax)
    print("z: ", zmin, "-", zmax)


print("skeleton")
summarize(meshwork.skeleton.vertices)
print()
print("merges")
summarize(merges[["x", "y", "z"]].values)
print()
print("synapses")
synapses = meshwork.anno["pre_syn"]
synapses_xyz = pt_to_xyz(synapses["ctr_pt_position"])
summarize(synapses_xyz[["x", "y", "z"]].values * res)

# %%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
skeleton_plot.plot_tools.plot_mw_skel(
    meshwork,
    ax=ax,
    # pull_radius=True,
    invert_y=True,
    # plot_soma=True,
    plot_presyn=True,
    plot_postsyn=True,
    # pre_anno={"splits": "pt"},
    # line_width=0.05,
    # color="black",
    x="x",
    y="y",
    presyn_size=20,
    postsyn_size=20,
)

# skeleton_plot.plot_tools.plot_verts(merges[["x", "y", "z"]].values, ax=ax, color="red")
skeleton_plot.plot_tools.plot_synapses(
    presyn_verts=merges[["x", "y", "z"]].values,
    # postsyn_verts=postsyn_verts,
    ax=ax,
    invert_y=True,
    presyn_color="red",
    presyn_size=100,
    x="x",
    y="y",
)

# %%

meshwork.add_annotations("splits_numpy", splits[["x", "y", "z"]].values)
# meshwork.add_annotations("merges", merges)
#
# %%
splits["mesh_index_filt"] = meshwork.anno["splits_numpy"]["mesh_index_filt"]

meshwork.add_annotations("splits", splits)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
skeleton_plot.plot_tools.plot_mw_skel(
    meshwork,
    ax=ax,
    # pull_radius=True,
    invert_y=True,
    # plot_soma=True,
    # pull_compartment_colors=True,
    plot_presyn=True,
    plot_postsyn=False,
    pre_anno={"splits": "pt"},
    line_width=0.05,
    color="black",
    presyn_size=100,
)
