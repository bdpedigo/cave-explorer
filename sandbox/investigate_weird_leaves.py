# %%

import time

t0 = time.time()

from datetime import timedelta

import caveclient as cc
import numpy as np
from pkg.edits import (
    find_supervoxel_component,
    get_initial_network,
    get_network_edits,
    get_network_metaedits,
)
from pkg.utils import get_level2_nodes_edges
from tqdm.autonotebook import tqdm
from pcg_skel.chunk_tools import build_spatial_graph
import pcg_skel.skel_utils as sk_utils
from meshparty import trimesh_io
from graspologic.layouts.colors import _get_colors
from meshparty import trimesh_vtk
from meshparty import skeletonize
import networkx as nx

from neuropull.graph import NetworkFrame
import pandas as pd

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

i = 5  # this one works

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

# print("Pulling initial state of the network")
# nf = get_initial_network(root_id, client, positions=False)
# nf.label_nodes_by_component(inplace=True, name="component")
# print()
# print()

# %%
from pkg.edits import get_lineage_tree
from requests.exceptions import HTTPError

positions = False
lineage_root = get_lineage_tree(root_id, client, flip=True, order="edits")
all_nodes = []
all_edges = []
for leaf in tqdm(
    lineage_root.leaves, desc="Finding all L2 components for lineage leaves"
):
    leaf_id = leaf.name
    nodes, edges = get_level2_nodes_edges(leaf_id, client, positions=positions)
    nodes["leaf_id"] = leaf_id
    all_nodes.append(nodes)
    all_edges.append(edges)
all_nodes = pd.concat(all_nodes, axis=0)
all_edges = pd.concat(all_edges, axis=0, ignore_index=True)

# nf = NetworkFrame(all_nodes, all_edges)


# %%
from pkg.plot import treeplot

treeplot(
    lineage_root,
    node_hue="status",
)

# %%
from anytree import find_by_attr

weird_node = find_by_attr(lineage_root, "weird leaf", name="status")
weird_node.name

weird_l2_nodes, _ = get_level2_nodes_edges(weird_node.name, client, positions=False)

# %%


def get_level2_nodes(node_id, client):
    level2_nodes = client.chunkedgraph.get_leaves(node_id, stop_layer=2)
    level2_nodes = pd.DataFrame(index=level2_nodes)
    return level2_nodes


weird_l2_nodes = get_level2_nodes(weird_node.name, client)

n_total = len(lineage_root.descendants) + 1

overlappers = []
# for other_node in tqdm(PreOrderIter(lineage_root), total=n_total):
for other_node in tqdm(lineage_root.leaves):
    nodes = get_level2_nodes(other_node.name, client)

    union = np.union1d(weird_l2_nodes.index, nodes.index)
    n_union = len(union)

    intersection = np.intersect1d(weird_l2_nodes.index, nodes.index)
    n_intersection = len(intersection)

    if n_intersection > 0:
        print(
            f"{other_node.name}: { n_intersection / n_union:.2g}",
        )
        overlappers.append(other_node.name)

# %%

import pcg_skel
import navis


def get_tree_neuron(root_id, client):
    skeleton = pcg_skel.coord_space_skeleton(root_id, client)
    f = "temp.swc"
    skeleton.export_to_swc(f)
    swc = pd.read_csv(f, sep=" ", header=None)
    swc.columns = ["node_id", "structure", "x", "y", "z", "radius", "parent_id"]
    tn = navis.TreeNeuron(swc)
    return tn


tree_neurons = []

for overlapper in overlappers:
    tn = get_tree_neuron(overlapper, client)
    tree_neurons.append(tn)


# %%
fig = navis.plot3d(tree_neurons, soma=False, inline=False)
fig.update_layout(
    template="plotly_white",
    plot_bgcolor="rgba(1,1,0.5,0)",
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
    ),
)
fig.show()

# %%
weird_tree = get_lineage_tree(weird_node.name, client)

# %%
treeplot(
    weird_tree,
    # node_hue="status",
)

# %%
rec_lineage_tree = get_lineage_tree(
    root_id, client, recurse=True, order=None, labels=True
)

# %%
treeplot(
    rec_lineage_tree,
    node_hue="status",
)
#%%

for leaf in np.unique(rec_lineage_tree.leaves): 
    print(leaf.name)

#%%


# %%
treeplot(
    lineage_root,
    node_hue="status",
)
# %
