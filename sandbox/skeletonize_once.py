# %%

import time

t0 = time.time()

import caveclient as cc
import matplotlib.pyplot as plt
import navis
import numpy as np
import pandas as pd
import pcg_skel.skel_utils as sk_utils
from cloudfiles import CloudFiles
from meshparty import skeletonize, trimesh_io
from networkframe import NetworkFrame
from pcg_skel.chunk_tools import build_spatial_graph
from sklearn.neighbors import NearestNeighbors

from pkg.constants import OUT_PATH
from pkg.edits import (
    NetworkDelta,
    get_initial_network,
    get_network_edits,
    get_network_metaedits,
)
from pkg.plot import networkplot
from pkg.utils import get_positions

# %%
recompute = False
cloud = False

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
i = 6
target_id = meta.iloc[i]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
root_id = client.chunkedgraph.get_latest_roots(root_id)[0]

if cloud:
    out_path = "allen-minnie-phase3/edit_info"
    cf = CloudFiles("gs://" + out_path)
else:
    out_path = OUT_PATH / "store_edit_info"
    cf = CloudFiles("file://" + str(out_path))

out_file = f"{root_id}_operations.json"

if not cf.exists(out_file) or recompute:
    print("Pulling network edits")
    networkdeltas_by_operation = get_network_edits(root_id, client, filtered=False)

    networkdelta_dicts = {}
    for operation_id, delta in networkdeltas_by_operation.items():
        networkdelta_dicts[operation_id] = delta.to_dict()

    _ = cf.put_json(out_file, networkdelta_dicts)
    print()

print("Reloading network edits")
networkdelta_dicts = cf.get_json(out_file)
networkdeltas_by_operation = {}
for operation_id, delta in networkdelta_dicts.items():
    networkdeltas_by_operation[int(operation_id)] = NetworkDelta.from_dict(delta)

print()

out_file = f"{root_id}_meta_operations.json"

if not cf.exists(out_file) or recompute:
    print("Compiling meta-edits")
    networkdeltas_by_meta_operation, meta_operation_map = get_network_metaedits(
        networkdeltas_by_operation
    )

    # remap all of the keys to strings
    networkdelta_dicts = {}
    for meta_operation_id, delta in networkdeltas_by_meta_operation.items():
        networkdelta_dicts[str(meta_operation_id)] = delta.to_dict()
    out_meta_operation_map = {}
    for meta_operation_id, operation_ids in meta_operation_map.items():
        out_meta_operation_map[str(meta_operation_id)] = operation_ids

    _ = cf.put_json(out_file, networkdelta_dicts)
    _ = cf.put_json(f"{root_id}_meta_operation_map.json", out_meta_operation_map)

    print()

print("Reloading meta-edits")
networkdelta_dicts = cf.get_json(out_file)
networkdeltas_by_meta_operation = {}
for meta_operation_id, delta in networkdelta_dicts.items():
    networkdeltas_by_meta_operation[int(meta_operation_id)] = NetworkDelta.from_dict(
        delta
    )
in_meta_operation_map = cf.get_json(f"{root_id}_meta_operation_map.json")
meta_operation_map = {}
for meta_operation_id, operation_ids in in_meta_operation_map.items():
    meta_operation_map[int(meta_operation_id)] = operation_ids

print()


def apply_edit(network_frame, network_delta):
    network_frame.add_nodes(network_delta.added_nodes, inplace=True)
    network_frame.add_edges(network_delta.added_edges, inplace=True)
    network_frame.remove_nodes(network_delta.removed_nodes, inplace=True)
    network_frame.remove_edges(network_delta.removed_edges, inplace=True)


def apply_additions(network_frame, network_delta):
    network_frame.add_nodes(network_delta.added_nodes, inplace=True)
    network_frame.add_edges(network_delta.added_edges, inplace=True)


# TODO could make this JSON serializable
nf = get_initial_network(root_id, client, positions=False)
print()

# %%

metaedit_ids = pd.Series(list(networkdeltas_by_meta_operation.keys()))
nuc_supervoxel = nuc.loc[target_id, "pt_supervoxel_id"]


def skeletonize_networkframe(networkframe, nan_rounds=10, require_complete=False):
    cv = client.info.segmentation_cloudvolume()

    lvl2_eg = networkframe.edges.values.tolist()
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


def skeleton_to_treeneuron(skeleton):
    f = "temp.swc"
    skeleton.export_to_swc(f)
    swc = pd.read_csv(f, sep=" ", header=None)
    swc.columns = ["node_id", "structure", "x", "y", "z", "radius", "parent_id"]
    tn = navis.TreeNeuron(swc)
    return tn


# %%


nf = get_initial_network(root_id, client, positions=True)

# make the biggest neuron we can make, union of all of the L2 graphs
for metaedit_id, metaedit in networkdeltas_by_meta_operation.items():
    operation_ids = meta_operation_map[metaedit_id]
    is_merges = []
    for operation_id in operation_ids:
        edit = networkdeltas_by_operation[operation_id]
        is_merges.append(edit.metadata["is_merge"])
        apply_additions(nf, edit)

# get the connected component for this union neuron
# TODO not sure how anybody could be disconnected here...
nodes = pd.DataFrame(index=np.unique(nf.nodes.index))
nodes[["x", "y", "z"]] = nf.nodes.loc[nodes.index, ["x", "y", "z"]].drop_duplicates()
edges = nf.edges.set_index(["source", "target"]).index.unique().to_frame(index=False)
union_nf = NetworkFrame(nodes, edges)
union_nf = union_nf.largest_connected_component(verbose=True).copy()

# %%

# skeletonize the union neuron
# this will be the skeleton that we map all of the nodes to
union_skeleton = skeletonize_networkframe(union_nf)

union_skeleton_nf = NetworkFrame(
    nodes=pd.DataFrame(data=np.array(union_skeleton.vertices), columns=["x", "y", "z"]),
    edges=pd.DataFrame(union_skeleton.edges, columns=["source", "target"]),
)

# %%

# map each node in the union networkframe to the nearest node in the skeleton
nner = NearestNeighbors(n_neighbors=1).fit(union_skeleton_nf.nodes[["x", "y", "z"]])

positions = get_positions(list(union_nf.nodes.index), client)
nn_in_skeleton = nner.kneighbors(positions[["x", "y", "z"]], return_distance=False)

node_to_skeleton_map = pd.Series(index=union_nf.nodes.index, data=nn_in_skeleton[:, 0])

# %%


def skeleton_and_level2_plot(skeleton_nf, level2_nf):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    networkplot(
        nodes=skeleton_nf.nodes,
        edges=skeleton_nf.edges,
        x="x",
        y="y",
        node_size=1,
        edge_linewidth=4,
        ax=ax,
        edge_alpha=0.3,
    )

    if "x" not in level2_nf.nodes.columns:
        nodes = get_positions(level2_nf.nodes.index, client)
    else:
        nodes = level2_nf.nodes

    networkplot(
        nodes=nodes,
        edges=level2_nf.edges,
        x="x",
        y="y",
        node_size=1,
        ax=ax,
        node_color="darkred",
        node_zorder=2,
        edge_zorder=2,
        edge_color="darkred",
        edge_linewidth=0.75,
    )
    return fig, ax


skeleton_and_level2_plot(union_skeleton_nf, union_nf)


# %%
og_nf = get_initial_network(root_id, client, positions=False)


# %%
def nf_to_skeleton_nf(nf, node_to_skeleton_map):
    index = nf.nodes.index.map(node_to_skeleton_map).unique().dropna().astype(int)
    nodes = union_skeleton_nf.nodes.loc[index].copy()

    sources = nf.edges["source"].map(node_to_skeleton_map)
    targets = nf.edges["target"].map(node_to_skeleton_map)
    edges = pd.DataFrame({"source": sources, "target": targets})
    edges = (
        edges.set_index(["source", "target"])
        .index.unique()
        .dropna(how="any")
        .to_frame(index=False)
        .astype(int)
    )
    return NetworkFrame(nodes, edges)


# %%

og_nf.nodes.index.map(node_to_skeleton_map).isna().mean()
og_skeleton_nf = nf_to_skeleton_nf(og_nf, node_to_skeleton_map)

skeleton_and_level2_plot(og_skeleton_nf, og_nf)

# %%
nf = og_nf.copy()
for metaedit_id, metaedit in networkdeltas_by_meta_operation.items():
    # apply_edit(nf, metaedit)

    skeleton_added_nodes = metaedit.added_nodes.index.map(node_to_skeleton_map)
