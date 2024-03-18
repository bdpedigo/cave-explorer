# %%
from random import shuffle

import caveclient as cc
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from cloudfiles import CloudFiles
from networkframe import NetworkFrame
from tqdm.auto import tqdm

from pkg.neuronframe import load_neuronframe
from pkg.plot import set_up_camera

# %%
cloud_bucket = "allen-minnie-phase3"
folder = "edit_sequences"

cf = CloudFiles(f"gs://{cloud_bucket}/{folder}")

files = list(cf.list())
files = pd.DataFrame(files, columns=["file"])

# pattern is root_id=number as the beginning of the file name
# extract the number from the file name and store it in a new column
files["root_id"] = files["file"].str.split("=").str[1].str.split("-").str[0].astype(int)
files["order_by"] = files["file"].str.split("=").str[2].str.split("-").str[0]
files["random_seed"] = files["file"].str.split("=").str[3].str.split("-").str[0]


file_counts = files.groupby("root_id").size()
has_all = file_counts[file_counts == 12].index

files_finished = files.query("root_id in @has_all")

root_options = files_finished["root_id"].unique()

# %%

client = cc.CAVEclient("minnie65_phase3_v1")
root_id = root_options[0]
neuron = load_neuronframe(root_id, client)

# %%
neuron.n_connected_components()

# %%

pv.set_jupyter_backend("client")

plotter = pv.Plotter()
set_up_camera(plotter, neuron, -2_000_000, 25, "-y")

neuron.label_nodes_by_component(inplace=True)

colors = sns.color_palette("husl", neuron.n_connected_components()).as_hex()

components = neuron.groupby_nodes("component", induced=True)

for i, (label, component) in enumerate(components):
    plotter.add_mesh(component.to_skeleton_polydata(), color=colors[i], line_width=0.1)
plotter.show()

# %%
prefix = "meta"
# label edges which cross operations/metaoperations
neuron.edges["cross_operation"] = (
    neuron.edges["source_operation_added"] != neuron.edges["target_operation_added"]
)
neuron.edges["cross_metaoperation"] = (
    neuron.edges["source_metaoperation_added"]
    != neuron.edges["target_metaoperation_added"]
)
# %%

meta = True
if meta:
    prefix = "meta"
else:
    prefix = ""

# now, create a view of the graph such that we are only looking at edges which go
# between nodes that were added at the same time AND which were never removed later.
# this should give us a set of connected components which are meaningful "chunks" of
# neuron that share the same edit history/relationship to the nucleus in terms of
# operations.
no_cross_neuron = neuron.query_edges(f"(~cross_{prefix}operation) & (~was_removed)")

# %%

n_connected_components = no_cross_neuron.n_connected_components()
print(n_connected_components)
# %%

# create labels for these different connected component pieces

neuron.nodes["segment"] = -1

segment_nodes = []

for i, component in tqdm(
    enumerate(no_cross_neuron.connected_components(directed=False)),
    total=n_connected_components,
):
    data = {
        "segment": i,
        "n_nodes": len(component.nodes),
        "n_edges": len(component.edges),
    }
    added = component.nodes[f"{prefix}operation_added"].iloc[0]
    removed = component.nodes[f"{prefix}operation_removed"].iloc[0]
    assert (component.nodes[f"{prefix}operation_added"] == added).all()
    assert (component.nodes[f"{prefix}operation_removed"] == removed).all()
    x, y, z = component.nodes[["x", "y", "z"]].mean()
    data[f"{prefix}operation_added"] = added
    data[f"{prefix}operation_removed"] = removed
    if added == -1 and removed == -1:
        data["segment_type"] = "unmodified"
    elif added == -1 and removed != -1:
        data["segment_type"] = "removed"
    elif added != -1 and removed == -1:
        data["segment_type"] = "added"
    else:
        data["segment_type"] = "modified"
    data["component_nodes"] = component.nodes.index.to_list()
    data["x"] = x
    data["y"] = y
    data["z"] = z

    neuron.nodes.loc[component.nodes.index, "segment"] = i
    segment_nodes.append(data)

segment_nodes = pd.DataFrame(segment_nodes)
segment_nodes[f"{prefix}operation_added"] = segment_nodes[
    f"{prefix}operation_added"
].astype(int)
segment_nodes[f"{prefix}operation_removed"] = segment_nodes[
    f"{prefix}operation_removed"
].astype(int)
segment_nodes.set_index("segment", inplace=True)
segment_nodes

# %%

colors = list(sns.color_palette("tab20", len(segment_nodes)).as_hex())

neuron.nodes["segment_color"] = neuron.nodes["segment"].astype(float)

shuffle(colors)

colors = [color.upper() for color in colors]


# %%
plotter = pv.Plotter()

i = 5
row = neuron.metaedits.query("~has_merge").iloc[i]

set_up_camera(
    plotter,
    row[["centroid_x", "centroid_y", "centroid_z"]],
    -2_000_000,
    25,
    "-y",
)

for segment_id in tqdm(segment_nodes.index):
    segment = neuron.query_nodes(f"segment == {segment_id}")
    assert segment.n_connected_components() == 1
    poly = segment.to_skeleton_polydata()
    plotter.add_mesh(poly, line_width=0.5, color=colors[segment_id])

plotter.show()


# %%
neuron.apply_node_features("segment", inplace=True)

# %%
neuron.nodes["segment"].value_counts()
# %%
len(neuron.groupby_nodes("segment"))

# %%
neuron.groupby_nodes("segment").size_edges()

# %%
segment_nf = neuron.condense(by='segment', func='size')


# # %%

# segment_edges = neuron.edges[["source_segment", "target_segment"]]

# edges_array = np.unique(np.sort(segment_edges.values, axis=1), axis=0)

# segment_edges = pd.DataFrame(edges_array, columns=["source", "target"]).set_index(
#     ["source", "target"], drop=False
# )
# segment_edges = segment_edges.query("source != target")
# segment_edges

# # %%
# segment_nf = NetworkFrame(segment_nodes, segment_edges)

# segment_nf

# %%
plotter = pv.Plotter()

row = neuron.nodes.loc[neuron.nucleus_id]

set_up_camera(
    plotter,
    row[["x", "y", "z"]],
    -2_000_000,
    25,
    "-y",
)

nuc_segment = neuron.nodes.loc[neuron.nucleus_id, "segment"]


# %%
sub_seg_nf = segment_nf.k_hop_neighborhood(nuc_segment, 3)


plotter = pv.Plotter()

row = neuron.nodes.loc[neuron.nucleus_id]

set_up_camera(
    plotter,
    row[["x", "y", "z"]],
    -2_000_000,
    25,
    "-y",
)

# for node_id in sub_seg_nf.nodes.index:
#     segment = neuron.query_nodes(f"segment == {node_id}")
#     poly = segment.to_skeleton_polydata()
#     plotter.add_mesh(poly, line_width=3, color=colors[node_id])

sub_neuron = neuron.query_nodes(
    "segment.isin(@sub_seg_nf.nodes.index)", local_dict=locals()
)
poly = sub_neuron.to_skeleton_polydata(label="segment_color")
plotter.add_mesh(poly, line_width=3)

# sub_neuron = segment.query_edges(
#     "source_segment.isin(@sub_seg_nf.nodes.index) | target_segment.isin(@sub_seg_nf.nodes.index) & (source_segment != target_segment)",
#     local_dict=locals(),
# )

# merges = sub_neuron.to_merge_polydata(draw_edges=True, prefix=prefix)
# splits = sub_neuron.to_split_polydata(draw_edges=True, prefix=prefix)
# merges_points = sub_neuron.to_merge_polydata(draw_edges=False, prefix=prefix)
# splits_points = sub_neuron.to_split_polydata(draw_edges=False, prefix=prefix)

# plotter.add_mesh(merges, color="blue", point_size=10, line_width=2)
# plotter.add_mesh(splits, color="red", point_size=10, line_width=2)
# plotter.add_mesh(merges_points, color="blue", point_size=10)
# plotter.add_mesh(splits_points, color="red", point_size=10)

plotter.show()

# %%
# this used to have (~was_removed)
cross_neuron = neuron.query_edges(f"cross_{prefix}operation").remove_unused_nodes()


# %%


component_edges = cross_neuron.edges[
    ["source_component_label", "target_component_label"]
].copy()

edges_array = np.unique(np.sort(component_edges.values, axis=1), axis=0)

component_edges = pd.DataFrame(edges_array, columns=["source", "target"]).set_index(
    ["source", "target"], drop=False
)
component_edges


component_nodelist = neuron.nodes["segment"].unique()
component_nodes = pd.DataFrame(index=component_nodelist)
component_nf = NetworkFrame(component_nodes, component_edges)

# %%
component_nf.edges.head(30)

# %%
plotter = pv.Plotter()
zeros = neuron.query_nodes("component_label == -1")
plotter.add_mesh(zeros.to_skeleton_polydata(), color="black", line_width=0.5)
plotter.show()

# %%
neuron.apply_node_features("segment", inplace=True)

# %%
