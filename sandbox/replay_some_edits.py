# %%

import time

t0 = time.time()

import caveclient as cc
import navis
import numpy as np
import pandas as pd
import pcg_skel.skel_utils as sk_utils
import plotly.graph_objs as go
from cloudfiles import CloudFiles
from meshparty import skeletonize, trimesh_io
from pcg_skel.chunk_tools import build_spatial_graph
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from pkg.constants import FIG_PATH, OUT_PATH
from pkg.edits import (
    NetworkDelta,
    find_supervoxel_component,
    get_initial_network,
    get_network_edits,
    get_network_metaedits,
)

# %%
recompute = False
cloud = True

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
i = 8
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
    networkdeltas_by_operation = get_network_edits(root_id, client)

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
n_samples = 15
n_steps = 4
edit_proportions = np.round(np.linspace(0, 1, n_steps + 2)[1:-1], 2)
edit_proportions = list(edit_proportions) + [1]

use_cc = True
n_total = n_samples * n_steps + 1
with tqdm(total=n_total, desc="Sampling skeletonized fragments") as pbar:
    for edit_proportion in edit_proportions:
        if edit_proportion == 1:
            n_samples_ = 1
        else:
            n_samples_ = n_samples
        for i in range(n_samples_):
            selected_edits = metaedit_ids.sample(frac=edit_proportion, replace=False)

            # do the editing
            this_nf = nf.copy()
            for metaedit_id in selected_edits:
                metaedit = networkdeltas_by_meta_operation[metaedit_id]
                apply_edit(this_nf, metaedit)

            # deal with the result
            if use_cc:
                largest_cc = None
                largest_cc_size = 0
                for component in this_nf.connected_components():
                    if len(component.nodes) > largest_cc_size:
                        largest_cc = component
                        largest_cc_size = len(component.nodes)
                nuc_nf = largest_cc
            else:
                nuc_nf = find_supervoxel_component(nuc_supervoxel, this_nf, client)
            skeleton = skeletonize_networkframe(nuc_nf)
            skeleton_samples[(edit_proportion, i)] = skeleton
            treeneuron = skeleton_to_treeneuron(skeleton)
            treeneuron_samples[(edit_proportion, i)] = treeneuron
            pbar.update(1)


# %%


n_rows = 3
n_cols = len(edit_proportions) - 1
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

for i, edit_proportion in enumerate(edit_proportions[:-1]):
    for j in range(n_rows):
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
fig.write_html(
    FIG_PATH
    / "replay_some_edits"
    / f"skeleton_samples_root_id={root_id}_use_cc={use_cc}.html"
)

# %%

# TODO add soma to the computations in order to easily get some of the other features
# described in the Gouwens et al 2019 paper (supplemental table 2).

# TODO will also somehow need to get axon dendrite stuff in there

# TODO also think about the logic for only "clean" neurons


def compute_height(treeneuron):
    return treeneuron.extents[1]


def compute_width(treeneuron):
    return treeneuron.extents[0]


def compute_n_branches(treeneuron):
    return len(treeneuron.branch_points)


def compute_total_length(treeneuron):
    return treeneuron.cable_length


def compute_features(treeneuron):
    out = {
        "height": compute_height(treeneuron),
        "width": compute_width(treeneuron),
        "n_branches": compute_n_branches(treeneuron),
        "total_length": compute_total_length(treeneuron),
    }
    return pd.Series(out)


feature_df = []
for edit_proportion in edit_proportions:
    if edit_proportion == 1:
        n_samples_ = 1
    else:
        n_samples_ = n_samples
    for i in range(n_samples_):
        treeneuron = treeneuron_samples[(edit_proportion, i)]
        features = compute_features(treeneuron)
        features["edit_proportion"] = edit_proportion
        features["sample"] = i
        feature_df.append(features)
feature_df = pd.DataFrame(feature_df)
feature_df

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so


def nice_labeler(x):
    x = str.capitalize(x)
    x = x.replace("_", " ")
    x = x.replace("N ", "# ")
    return x


sns.set_context("talk", font_scale=1)
fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
sns.despine(fig, offset=25)

vars = ["height", "width", "n_branches", "total_length"]

for i, var in enumerate(vars):
    plot = (
        so.Plot(
            data=feature_df,
            x="edit_proportion",
            y=var,
            color="edit_proportion",
        )
        .on(axs.flat[i])
        .add(so.Dots(fill=True, fillalpha=1), so.Jitter(0.3))
        .add(so.Dash(width=0.5, linewidth=3), so.Agg(np.mean))
        .scale(color="flare", x=so.Nominal())
        .label(x=nice_labeler, y=nice_labeler)
        .share(x=True)
    )
    plot.plot()


[fig.legends[i].set_visible(False) for i in range(4)]


plt.savefig(
    FIG_PATH
    / "replay_some_edits"
    / f"feature_samples_root_id={root_id}_use_cc={use_cc}.png",
    dpi=300,
)
