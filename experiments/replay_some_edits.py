# %%

import time

t0 = time.time()

from datetime import timedelta

import caveclient as cc
import numpy as np
from cloudfiles import CloudFiles
from pkg.edits import (
    NetworkDelta,
    find_supervoxel_component,
    get_initial_network,
    get_network_edits,
    get_network_metaedits,
)
from pkg.paths import OUT_PATH
from pkg.utils import get_level2_nodes_edges
from tqdm.autonotebook import tqdm

from neuropull.graph import NetworkFrame

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


# TODO could make this JSON serializable
nf = get_initial_network(root_id, client, positions=False)
print()

# %%
import pandas as pd
from pcg_skel.chunk_tools import build_spatial_graph
import pcg_skel.skel_utils as sk_utils
from meshparty import trimesh_io
from meshparty import skeletonize
import navis
from tqdm import tqdm

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


skeleton_samples = {}
treeneuron_samples = {}
n_samples = 3
n_steps = 4
edit_proportions = np.linspace(0, 1, n_steps + 2)[1:-1]
n_total = n_samples * n_steps
with tqdm(total=n_total, desc="Sampling skeletonized fragments") as pbar:
    for edit_proportion in edit_proportions:
        for i in range(n_samples):
            selected_edits = metaedit_ids.sample(frac=edit_proportion, replace=False)

            # do the editing
            this_nf = nf.copy()
            for metaedit_id in selected_edits:
                metaedit = networkdeltas_by_meta_operation[metaedit_id]
                apply_edit(this_nf, metaedit)

            # deal with the result
            nuc_nf = find_supervoxel_component(nuc_supervoxel, this_nf, client)
            skeleton = skeletonize_networkframe(nuc_nf)
            skeleton_samples[(edit_proportion, i)] = skeleton
            treeneuron = skeleton_to_treeneuron(skeleton)
            treeneuron_samples[(edit_proportion, i)] = treeneuron
            print(treeneuron.n_nodes)
            pbar.update(1)

# %%

import plotly.graph_objs as go
from plotly.subplots import make_subplots


n_rows = n_samples
n_cols = len(edit_proportions)
specs = n_rows * [n_cols * [{"type": "scene"}]]
fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    specs=specs,
    shared_xaxes=True,
    shared_yaxes=True,
    column_titles=[f"{p:.2g}" for p in edit_proportions],
    row_titles=[j + 1 for j in range(n_samples)],
    horizontal_spacing=0,
    vertical_spacing=0,
    x_title="Proportion of edits",
    y_title="Sample",
)

for i, edit_proportion in enumerate(edit_proportions):
    for j in range(n_samples):
        treeneuron = treeneuron_samples[(edit_proportion, j)]
        fake_fig = go.Figure()
        navis.plot3d(treeneuron, fig=fake_fig, soma=False, inline=False)
        trace = fake_fig.data[0]
        fig.add_trace(trace, row=j + 1, col=i + 1)

fig.update_layout(
    showlegend=False,
    template="plotly_white",
    plot_bgcolor="rgba(1,1,1,0)",

)

fig.update_scenes(
    dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
    ),
    row=None,
    col=None,
)

fig.show()

# %%


# %%
random_metaedit_ids = np.random.permutation(metaedit_ids)
for metaedit_id in tqdm(random_metaedit_ids, desc="Playing meta-edits in random order"):
    metaedit = networkdeltas_by_meta_operation[metaedit_id]
    apply_edit(nf, metaedit)
print()

print("Finding final fragment with nucleus attached")
nuc_supervoxel = nuc.loc[target_id, "pt_supervoxel_id"]

nuc_nf = find_supervoxel_component(nuc_supervoxel, nf, client)
print()

root_nodes, root_edges = get_level2_nodes_edges(root_id, client, positions=False)
root_nf = NetworkFrame(root_nodes, root_edges)

assert root_nf == nuc_nf
print()
