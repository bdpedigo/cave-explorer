# %%

import time

t0 = time.time()

from datetime import timedelta

import caveclient as cc
import networkx as nx
import numpy as np
import pandas as pd
import pcg_skel.skel_utils as sk_utils
from graspologic.layouts.colors import _get_colors
from meshparty import skeletonize, trimesh_io, trimesh_vtk
from networkframe import NetworkFrame
from pcg_skel.chunk_tools import build_spatial_graph
from tqdm.auto import tqdm

from pkg.edits import (
    get_initial_network,
    get_network_edits,
    get_network_metaedits,
)
from pkg.utils import get_level2_nodes_edges

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
# i = 2#
# i = 14

i = 6  # this one works

target_id = meta.iloc[i]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
root_id = client.chunkedgraph.get_latest_roots(root_id)[0]


# %%

networkdeltas_by_operation, edit_lineage_graph = get_network_edits(
    root_id, client, filtered=False
)
print()

# %%

print("Finding meta-operations")
networkdeltas_by_meta_operation, meta_operation_map = get_network_metaedits(
    networkdeltas_by_operation, edit_lineage_graph
)
print()

# %%

print("Pulling initial state of the network")
nf = get_initial_network(root_id, client, positions=False)
nf.label_nodes_by_component(inplace=True, name="component")
print()
print()

metaedit_ids = np.array(list(networkdeltas_by_meta_operation.keys()))
random_metaedit_ids = np.random.permutation(metaedit_ids)
nf_timeseries = [nf.copy()]
for metaedit_id in tqdm(random_metaedit_ids, desc="Playing meta-edits in random order"):
    delta = networkdeltas_by_meta_operation[metaedit_id]
    nf = nf.copy()
    nf.add_nodes(delta.added_nodes, inplace=True)
    nf.add_edges(delta.added_edges, inplace=True)
    nf.remove_nodes(delta.removed_nodes, inplace=True)
    nf.remove_edges(delta.removed_edges, inplace=True)
    nf.label_nodes_by_component(inplace=True, name="component")
    nf_timeseries.append(nf.copy())
print()

# %%

components_over_time = []
for i, nf in enumerate(nf_timeseries):
    components = nf.nodes["component"]
    components.name = f"t{i}"
    components_over_time.append(components)

components_over_time = pd.concat(components_over_time, axis=1)
components_over_time

# %%

component_graph = nx.DiGraph()

for t in range(len(components_over_time.columns)):
    for component_id in components_over_time[f"t{t}"].unique():
        if not pd.isna(component_id):
            component_graph.add_node((t, component_id))

for t in tqdm(range(len(components_over_time.columns) - 1)):
    labels_i = components_over_time[f"t{t}"]
    labels_j = components_over_time[f"t{t+1}"]
    for label_k in labels_i.unique():
        if not pd.isna(label_k):
            this_component = labels_i[labels_i == label_k].index.sort_values()
            for label_l in labels_j.unique():
                if not pd.isna(label_l):
                    that_component = labels_j[labels_j == label_l].index.sort_values()

                    if this_component.equals(that_component):
                        component_graph.add_edge((t, label_k), (t + 1, label_l))

# %%

# for each connected component in the component graph, find the set of L2 nodes that it
# corresponds to
object_to_l2_map = {}
timecomponent_to_object_map = {}
for object_id, cc_time_nodes in enumerate(
    nx.weakly_connected_components(component_graph)
):
    exemplar = next(iter(cc_time_nodes))
    t = exemplar[0]
    component_id = exemplar[1]
    labels = components_over_time[f"t{t}"]
    l2_nodes = labels[labels == component_id].index
    object_to_l2_map[object_id] = l2_nodes

    for node in cc_time_nodes:
        timecomponent_to_object_map[node] = object_id

timecomponent_to_object_map = pd.Series(timecomponent_to_object_map)
timecomponent_to_object_map.sort_index()

# %%
node_objects_over_time = []
for t in range(len(components_over_time.columns)):
    column = components_over_time[f"t{t}"]
    new_column = column.map(timecomponent_to_object_map.loc[t]).astype("Int64")
    new_column.name = f"t{t}"
    node_objects_over_time.append(new_column)
node_objects_over_time = pd.concat(node_objects_over_time, axis=1)
node_objects_over_time


# %%


# %%


def skeletonize_component(networkframe, nan_rounds=10, require_complete=False):
    cv = client.info.segmentation_cloudvolume()

    lvl2_eg = component.edges.values.tolist()
    eg, l2dict_mesh, l2dict_r_mesh, x_ch = build_spatial_graph(
        lvl2_eg,
        cv,
        client=client,
        method="service",
        require_complete=require_complete,
    )
    mesh = trimesh_io.Mesh(
        vertices=x_ch,
        faces=[[0, 0, 0]],  # Some functions fail if no faces are set.
        link_edges=eg,
    )

    sk_utils.fix_nan_verts_mesh(mesh, nan_rounds)

    sk = skeletonize.skeletonize_mesh(
        mesh,
        invalidation_d=10_000,
        compute_radius=False,
        cc_vertex_thresh=0,
        remove_zero_length_edges=True,
    )
    return sk


# %%
skeletons_per_object = {}

for object_id, l2_nodes in tqdm(object_to_l2_map.items()):
    first_occurance = timecomponent_to_object_map[
        timecomponent_to_object_map == object_id
    ].index[0][0]
    current_nf = nf_timeseries[first_occurance]
    component_nodes = object_to_l2_map[object_id]
    component = current_nf.loc[component_nodes, component_nodes]
    if len(component.edges) > 0:
        sk = skeletonize_component(component)
        skeletons_per_object[object_id] = sk

# %%


def hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) / 255 for i in range(0, lv, lv // 3))


color_dict = _get_colors(True, None)
colors = color_dict["nominal"]
colors = [hex_to_rgb(c) for c in colors]
colors = colors + colors

sk_actors = []

for i, sk in tqdm(skeletons_per_object.items()):
    sk_actor = trimesh_vtk.skeleton_actor(sk, color=colors[i])
    sk_actors.append(sk_actor)

# %%

import vtk
from meshparty.trimesh_vtk import _setup_renderer


class vtkTimerCallback:
    def __init__(self, steps, actor, iren):
        self.timer_count = 0
        self.steps = steps
        self.actor = actor
        self.iren = iren
        self.timerId = None

    def execute(self, obj, event):
        step = 0
        while step < self.steps:
            self.actor.SetColor(
                (
                    self.timer_count / 100.0,
                    self.timer_count / 100.0,
                    self.timer_count / 100.0,
                )
            )
            iren.GetRenderWindow().Render()
            self.timer_count += 1
            step += 1
        if self.timerId:
            iren.DestroyTimer(self.timerId)


ren, renWin, iren = _setup_renderer(
    video_width=1080, video_height=720, back_color=(1, 1, 1), camera=None
)
# for sk_actor in sk_actors:
ren.AddActor(sk_actor)


ren.ResetCamera()
renWin.Render()

trackCamera = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(trackCamera)
# enable user interface interactor
iren.Initialize()

cb = vtkTimerCallback(100, sk_actor, iren)
iren.AddObserver("TimerEvent", cb.execute)
cb.timerId = iren.CreateRepeatingTimer(100)

iren.Render()
iren.Start()

# %%

color_dict = _get_colors(True, None)


nuc_supervoxel = nuc.loc[target_id, "pt_supervoxel_id"]

supervoxel_l2_id = client.chunkedgraph.get_root_id(nuc_supervoxel, level2=True)


colors = color_dict["nominal"]
colors = [hex_to_rgb(c) for c in colors]

sk_actors = []
for i, component in enumerate(tqdm(list(nf.connected_components())[:])):
    if len(component.edges) > 0:
        if supervoxel_l2_id in component.nodes.index:
            color = (0.05, 0.05, 0.05)
        else:
            color = colors[i]
        sk = skeletonize_component(component)
        sk_actor = trimesh_vtk.skeleton_actor(sk, color=color)
        sk_actors.append(sk_actor)

trimesh_vtk.render_actors(sk_actors)


# %%


# %%

# %%

# %%


# %%

print("Checking for correspondence of final edited neuron and original root neuron")
root_nodes, root_edges = get_level2_nodes_edges(root_id, client, positions=False)
root_nf = NetworkFrame(root_nodes, root_edges)

print("L2 graphs match?", root_nf == nuc_nf)
print()

# %%
delta = timedelta(seconds=time.time() - t0)
print("Time elapsed: ", delta)
print()
