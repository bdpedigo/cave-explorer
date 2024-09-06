# %%
from random import shuffle

import pandas as pd
import pyvista as pv
import seaborn as sns
from tqdm.auto import tqdm

from pkg.neuronframe import load_neuronframe
from pkg.plot import set_up_camera
from pkg.utils import start_client

root_id = 864691135213953920

client = start_client()

neuron = load_neuronframe(root_id, client)

# %%
# give the edges info about when those nodes were added
neuron.apply_node_features("operation_added", inplace=True)
neuron.apply_node_features("metaoperation_added", inplace=True)

# label edges which cross operations/metaoperations
neuron.edges["cross_operation"] = (
    neuron.edges["source_operation_added"] != neuron.edges["target_operation_added"]
)
neuron.edges["cross_metaoperation"] = (
    neuron.edges["source_metaoperation_added"]
    != neuron.edges["target_metaoperation_added"]
)
neuron.edges["was_removed"] = neuron.edges["operation_removed"] != -1
neuron.nodes["was_removed"] = neuron.nodes["operation_removed"] != -1

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
    enumerate(no_cross_neuron.connected_components()), total=n_connected_components
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
    # if added == removed: 
    #     continue
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

from pkg.constants import MERGE_COLOR, SPLIT_COLOR

colors = list(sns.color_palette("tab20", len(segment_nodes)).as_hex())

neuron.nodes["segment_color"] = neuron.nodes["segment"].astype(float)

shuffle(colors)

colors = [color.upper() for color in colors]

plotter = pv.Plotter()

i = 10
row = neuron.metaedits.query("~has_merge").iloc[i]

set_up_camera(
    plotter,
    location=row[["centroid_x", "centroid_y", "centroid_z"]],
    setback=-2_000_000,
    elevation=25,
)

# merges = neuron.to_merge_polydata(draw_edges=True, prefix=prefix)
# splits = neuron.to_split_polydata(draw_edges=True, prefix=prefix)
merges_points = neuron.to_merge_polydata(draw_edges=False, prefix=prefix)
splits_points = neuron.to_split_polydata(draw_edges=False, prefix=prefix)

# plotter.add_mesh(merges, color="blue", point_size=20, line_width=10)
# plotter.add_mesh(splits, color="red", point_size=20, line_width=10)
plotter.add_mesh(
    merges_points, color=MERGE_COLOR, point_size=10, render_points_as_spheres=True
)
plotter.add_mesh(
    splits_points, color=SPLIT_COLOR, point_size=10, render_points_as_spheres=True
)

# poly = neuron.query_edges("~was_removed").to_skeleton_polydata()
# plotter.add_mesh(poly, line_width=3)
show_neuron = neuron.query_nodes("~was_removed")
poly = show_neuron.to_skeleton_polydata(label="segment_color")
plotter.add_mesh(poly, line_width=0.5, scalars="segment_color", cmap=colors)

# point_poly = show_neuron.to_skeleton_polydata(label="segment_color", draw_lines=False)
# plotter.add_mesh(poly, line_width=3, scalars="segment_color", cmap=colors)
# plotter.add_mesh(point_poly, scalars="segment_color", cmap=colors, point_size=3)

# poly = neuron.to_skeleton_polydata()
# plotter.add_mesh(poly, line_width=0.5, color="black")

plotter.show()

# %%

import numpy as np
from networkframe import NetworkFrame

neuron.apply_node_features("segment", inplace=True)

segment_edges = neuron.edges[["source_segment", "target_segment"]]


edges_array = np.unique(np.sort(segment_edges.values, axis=1), axis=0)

segment_edges = pd.DataFrame(edges_array, columns=["source", "target"]).set_index(
    ["source", "target"], drop=False
)
segment_edges = segment_edges.query("source != target")
segment_edges

# %%
segment_nf = NetworkFrame(segment_nodes, segment_edges)

segment_nf

# %%
select_nodes = segment_nodes.loc[408]['component_nodes']

sub_neuron = neuron.query_nodes(f"index in {select_nodes}")

sub_neuron.to_skeleton_polydata().plot()

#%%
possible_targets = segment_nf.edges.query('source==408')['target'].values
segment_nf.nodes.loc[possible_targets]
# %%
