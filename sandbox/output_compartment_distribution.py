# %%
import io
import json

import caveclient as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import Node, RenderTree
from anytree.iterators import LevelOrderIter
from cloudfiles import CloudFiles
from graspologic.plot import networkplot
from matplotlib.colors import Normalize, to_hex
from matplotlib.patches import ConnectionPatch
from meshparty import meshwork
from networkframe import NetworkFrame
from nglui import statebuilder as sb
from scipy.stats import chi2_contingency

from pkg.constants import MTYPES_TABLE
from pkg.plot import radial_hierarchy_pos

# %%
client = cc.CAVEclient("minnie65_public_v661")


proofreading_table = client.materialize.query_table(
    "proofreading_status_public_release"
)
proofreading_table = proofreading_table.query(
    "status_axon == 'extended' & status_dendrite == 'extended'"
)

nucs = client.materialize.query_table("nucleus_detection_v0")


# %%

# mtypes_661 = client.materialize.query_table('column_atype')

# %%
# ptc_roots = mtypes_661.query("cell_type == 'PTC'")["pt_root_id"]
#
# %%
nucs.drop_duplicates(subset="pt_root_id", keep=False, inplace=True)
nucs.set_index("pt_root_id", inplace=True)

# %%
# 864691136100014453 has an interesting long range projection that gets picked up
# 864691135759685966 has a region of tangles, kinda
# 864691135234167385 another interesting long range projection
# 864691135082074359 not much going on here besides the soma
# 864691135509890057 many sigs
# 864691135082074359 not much going on here besides the soma
# 864691136195284556 crazy axon, projection to L2 gets picked up

# root_id = ptc_roots.sample(1).values[0]
# root_id = 864691135234167385
# root_id = 864691136195284556
# root_id = 864691135082074359
# root_id = 864691135082074359
root_id = 864691135396580129
nuc_id = nucs.loc[root_id, "id"]

filename = f"{root_id}_{nuc_id}.h5"
skel_path = "https://storage.googleapis.com/allen-minnie-phase3/minniephase3-emily-pcg-skeletons/minnie_all/v661/meshworks"


def load_mw(directory, filename):
    # REF: stolen from https://github.com/AllenInstitute/skeleton_plot/blob/main/skeleton_plot/skel_io.py
    # filename = f"{root_id}_{nuc_id}/{root_id}_{nuc_id}.h5"
    '''
    """loads a meshwork file from .h5 into meshparty.meshwork object

    Args:
        directory (str): directory location of meshwork .h5 file. in cloudpath format as seen in https://github.com/seung-lab/cloud-files
        filename (str): full .h5 filename

    Returns:
        meshwork (meshparty.meshwork): meshwork object containing .h5 data
    """'''

    if "://" not in directory:
        directory = "file://" + directory

    cf = CloudFiles(directory)
    binary = cf.get([filename])

    with io.BytesIO(cf.get(binary[0]["path"])) as f:
        f.seek(0)
        mw = meshwork.load_meshwork(f)

    mw.reset_mask()
    return mw


neuron = load_mw(skel_path, filename)
neuron.reset_mask()

# %%

client = cc.CAVEclient("minnie65_phase3_v1")
mtypes = client.materialize.query_table(MTYPES_TABLE, desired_resolution=[1, 1, 1])
mtypes = mtypes.drop_duplicates(subset="pt_root_id", keep=False)
mtypes.set_index("pt_root_id", inplace=True)

# %%

# updating and annotating synapse tables

pre_syns = neuron.anno["pre_syn"].df.set_index("id")
modern_pre_syns = (
    client.materialize.query_table(
        "synapses_pni_2", filter_in_dict={"id": pre_syns.index}
    )
    .set_index("id")
    .loc[pre_syns.index]
)
pre_syns = modern_pre_syns
pre_syns["skel_index"] = neuron.anno["pre_syn"].mesh_index.to_skel_index_padded
pre_syns["segment"] = neuron.skeleton.segment_map[pre_syns["skel_index"]]
pre_syns["post_mtype"] = pre_syns["post_pt_root_id"].map(mtypes["cell_type"])
print(
    "Proportion of pre-synapses mapped to cell type:",
    pre_syns["post_mtype"].notna().mean(),
)

post_syns = neuron.anno["post_syn"].df.set_index("id")
modern_post_syns = (
    client.materialize.query_table(
        "synapses_pni_2", filter_in_dict={"id": post_syns.index}
    )
    .set_index("id")
    .loc[post_syns.index]
)
post_syns = modern_post_syns
post_syns["skel_index"] = neuron.anno["post_syn"].mesh_index.to_skel_index_padded
post_syns["segment"] = neuron.skeleton.segment_map[post_syns["skel_index"]]
post_syns["pre_mtype"] = post_syns["pre_pt_root_id"].map(mtypes["cell_type"])
print(
    "Proportion of post-synapses mapped to cell type:",
    post_syns["pre_mtype"].notna().mean(),
)

# %%
roots = pre_syns["post_pt_root_id"]
level2_ids = client.chunkedgraph.get_roots(
    pre_syns["post_pt_supervoxel_id"], stop_layer=2
)
pre_syns["post_pt_level2_id"] = level2_ids
points = pre_syns["post_pt_position"]
points = points.apply(lambda x: x * [4, 4, 40])

points.index = roots

import time

from troglobyte.features import CAVEWrangler

currtime = time.time()

wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=True)
wrangler.set_objects(roots.values)
wrangler.set_query_boxes_from_points(points, box_width=10_000)
wrangler.query_level2_ids()
wrangler.query_level2_edges(warn_on_missing=False)
wrangler.query_level2_shape_features()

# %%
wrangler.query_level2_synapse_features(method="existing")
wrangler.aggregate_features_by_neighborhood(aggregations=["mean", "std"])

print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%

from pathlib import Path

from skops.io import load

model_path = Path("data/models/local_compartment_classifier_ej_skeletons.skops")

model = load(model_path)

wrangler.register_model(model, "local_compartment_classifier")
wrangler.stack_model_predict("local_compartment_classifier")
wrangler.stack_model_predict_proba("local_compartment_classifier")

# %%

X = wrangler.features_.dropna()
y_pred = model.predict(X[model.feature_names_in_])
posterior = model.predict_proba(X[model.feature_names_in_])
y_pred = pd.Series(y_pred, index=X.index, name="compartment")
posterior = pd.DataFrame(posterior, index=X.index, columns=model.classes_)
predictions = pd.concat([y_pred, posterior], axis=1)
predictions

# %%
wrangler.query_level2_networks()


# %%

from typing import Optional, Union

import cloudvolume


def write_networkframes_to_skeletons(
    networkframes: Union[NetworkFrame, dict[NetworkFrame]],
    client: cc.CAVEclient,
    attribute: Optional[str] = None,
    directory: str = "gs://allen-minnie-phase3/tempskel",
    spatial_columns: Optional[list[str]] = None,
):
    # register an info file and set up CloudVolume
    base_info = client.chunkedgraph.segmentation_info
    base_info["skeletons"] = "skeleton"
    info = base_info.copy()

    cv = cloudvolume.CloudVolume(
        f"precomputed://{directory}",
        mip=0,
        info=info,
        compress=False,
    )
    cv.commit_info()

    sk_info = cv.skeleton.meta.default_info()
    sk_info["vertex_attributes"] = [
        {"id": "radius", "data_type": "float32", "num_components": 1},
        {"id": "vertex_types", "data_type": "float32", "num_components": 1},
    ]
    cv.skeleton.meta.info = sk_info
    cv.skeleton.meta.commit_info()

    sks = []
    if isinstance(networkframes, NetworkFrame):
        networkframes = {0: networkframes}

    for name, networkframe in networkframes.items():
        # extract vertex information
        vertices = networkframe.nodes[spatial_columns].values
        edges_unmapped = networkframe.edges[["source", "target"]].values
        edges = networkframe.nodes.index.get_indexer_for(
            edges_unmapped.flatten()
        ).reshape(edges_unmapped.shape)

        vertex_types = networkframe.nodes[attribute].values.astype(np.float32)

        radius = np.ones(len(vertices), dtype=np.float32)

        sk_cv = cloudvolume.Skeleton(
            vertices,
            edges,
            radius,
            None,
            segid=name,
            extra_attributes=sk_info["vertex_attributes"],
            space="physical",
        )
        sk_cv.vertex_types = vertex_types

        sks.append(sk_cv)

    cv.skeleton.upload(sks)


# %%

networkframes = wrangler.object_level2_networks_

for frame in networkframes:
    frame.nodes["local_compartment_classifier_predict_float"] = (
        frame.nodes["local_compartment_classifier_predict"]
        .map({"soma": 0, "axon": 1, "dendrite": 2})
        .fillna(-1)
        .astype(float)
    )

# %%

write_networkframes_to_skeletons(
    networkframes,
    client=client,
    attribute="local_compartment_classifier_predict_float",
    spatial_columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"],
)

# %%


import json
from pathlib import Path
from typing import Optional, Union

import caveclient as cc
import cloudvolume
import numpy as np
import pandas as pd
import seaborn as sns
from networkframe import NetworkFrame
from nglui import statebuilder
from skops.io import load

sbs = []
dfs = []
layers = []
viewer_resolution = client.info.viewer_resolution()
img_layer = statebuilder.ImageLayerConfig(
    client.info.image_source(),
)
seg_layer = statebuilder.SegmentationLayerConfig(
    client.info.segmentation_source(),
    alpha_3d=0.6,
    name="seg",
)
seg_layer.add_selection_map(selected_ids_column="object_id")

base_sb = statebuilder.StateBuilder(
    [img_layer, seg_layer],
    client=client,
    resolution=viewer_resolution,
)

sbs.append(base_sb)
dfs.append(pd.DataFrame({"object_id": [root_id]}))

skel_layer = statebuilder.SegmentationLayerConfig(
    "precomputed://gs://allen-minnie-phase3/tempskel",
    name="skeleton",
)
skel_layer.add_selection_map(selected_ids_column="object_id")

sb = statebuilder.StateBuilder(
    [skel_layer],
    client=client,
    resolution=viewer_resolution,
)

sbs.append(sb)
dfs.append(pd.DataFrame({"object_id": networkframes.index}))


sb = statebuilder.ChainedStateBuilder(sbs)
json_out = statebuilder.helpers.package_state(dfs, sb, client=client, return_as="json")
state_dict = json.loads(json_out)

shader = """
void main() {
    float compartment = vCustom2;
    vec4 uColor = segmentColor();
    if (compartment == 0.0) {
        emitRGB(vec3(0.5, 0.5, 0.5));
    } else if (compartment == 1.0) {
        emitRGB(vec3(0.5, 0.5, 0.5));
    } else if (compartment == 2.0) {
        emitRGB(vec3(0.5, 0.5, 0.5));
    } else {
        emitRGB(vec3(0.5, 0.5, 0.5));
    }
}
"""
skel_rendering_kws = {
    "shader": shader,
    "mode2d": "lines_and_points",
    "mode3d": "lines",
    "lineWidth3d": 2.5,
}

state_dict["layers"][2]["skeletonRendering"] = skel_rendering_kws


statebuilder.StateBuilder(base_state=state_dict, client=client).render_state(
    return_as="html"
)

# %%

level2_class_predictions = {}

for object_id, networkframe in networkframes.items():
    relevant_level2_ids = networkframe.nodes.index.intersection(level2_ids)
    for level2_id in relevant_level2_ids:
        neighborhood = networkframe.k_hop_neighborhood(level2_id, k=5)
        mean_posteriors = neighborhood.nodes[
            [
                "local_compartment_classifier_predict_proba_axon",
                "local_compartment_classifier_predict_proba_dendrite",
                "local_compartment_classifier_predict_proba_soma",
            ]
        ].mean()
        if mean_posteriors.isna().any():
            class_prediction = "unknown"
        else:
            max_col = mean_posteriors.idxmax()
            class_prediction = max_col.split("_")[-1]
        level2_class_predictions[level2_id] = class_prediction

level2_class_predictions = pd.Series(level2_class_predictions, name="compartment")

pre_syns["post_pt_compartment"] = pre_syns["post_pt_level2_id"].map(
    level2_class_predictions
)

# %%

skeleton_nodes = pd.DataFrame(neuron.skeleton.vertices, columns=["x", "y", "z"])
skeleton_edges = pd.DataFrame(
    neuron.skeleton.edges, columns=["target", "source"]
)  # note the swap here - edges going away from root now
skeleton_nodes["segment"] = neuron.skeleton.segment_map
skeleton_nodes["is_branch_point"] = False
skeleton_nodes.loc[neuron.skeleton.branch_points, "is_branch_point"] = True

nf = NetworkFrame(nodes=skeleton_nodes, edges=skeleton_edges)
adj = nf.to_sparse_adjacency()
out_degrees = adj.sum(axis=1)
nf.nodes["out_degree"] = out_degrees


nf.apply_node_features("segment", inplace=True)
nf.edges
condensed_edges = (
    nf.edges.groupby(["source_segment", "target_segment"])
    .any()
    .query("source_segment != target_segment")
    .reset_index()
    .drop(columns=["source", "target"])
    .rename(columns={"source_segment": "source", "target_segment": "target"})
)

condensed_nodes = nf.nodes.groupby("segment").mean()
condensed_nodes["skel_node_count"] = nf.nodes.groupby("segment").size()
condensed_nodes["pre_syn_count"] = pre_syns.groupby("segment").size()
condensed_nodes["post_syn_count"] = post_syns.groupby("segment").size()
condensed_nodes["pre_syn_count"].fillna(0, inplace=True)
condensed_nodes["post_syn_count"].fillna(0, inplace=True)

segment_branch_points = (
    skeleton_nodes.query("is_branch_point").groupby("segment").first()
)
condensed_nodes["branch_x"] = segment_branch_points["x"] / 4
condensed_nodes["branch_y"] = segment_branch_points["y"] / 4
condensed_nodes["branch_z"] = segment_branch_points["z"] / 40

condensed_nf = NetworkFrame(nodes=condensed_nodes, edges=condensed_edges)

# %%
condensed_g = condensed_nf.to_networkx(create_using=nx.DiGraph)

nx.is_tree(condensed_g)

# %%

root_seg = int(skeleton_nodes.loc[neuron.skeleton.root, "segment"])

# convert to an anytree object
paths = nx.shortest_path(condensed_g, source=root_seg)

root = Node(root_seg)
nodes = {root_seg: root}
for source, path in paths.items():
    for node_i, node_j in nx.utils.pairwise(path):
        if node_j not in nodes:
            nodes[node_j] = Node(node_j)
        nodes[node_j].parent = nodes[node_i]

for pre, _, node in RenderTree(root):
    print("%s%s" % (pre, node.name))

# %%

condensed_nf.nodes["descendant_pre_syn_count"] = 0
condensed_nf.nodes["descendant_post_syn_count"] = 0

for node in LevelOrderIter(root):
    descendants = [n.name for n in node.descendants] + [node.name]
    sub_nodes = condensed_nodes.query("segment.isin(@descendants)")
    counts = sub_nodes[["pre_syn_count", "post_syn_count"]].sum(axis=0)
    condensed_nf.nodes.loc[node.name, "descendant_pre_syn_count"] = counts[
        "pre_syn_count"
    ]
    condensed_nf.nodes.loc[node.name, "descendant_post_syn_count"] = counts[
        "post_syn_count"
    ]

# %%
pos = radial_hierarchy_pos(condensed_g, root=root_seg)

# %%

condensed_nf.nodes["pos_x"] = condensed_nf.nodes.index.map(lambda x: pos[x][0])
condensed_nf.nodes["pos_y"] = condensed_nf.nodes.index.map(lambda x: pos[x][1])

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

networkplot(
    adjacency=condensed_nf.to_sparse_adjacency(),
    node_data=condensed_nf.nodes,
    node_size="descendant_pre_syn_count",
    node_sizes=(10, 100),
    edge_linewidth=3.0,
    x="pos_x",
    y="pos_y",
    ax=ax,
)


# %%

synapse_label = "post_pt_compartment"
pre_syn_tables_by_node = {}
post_syn_tables_by_node = {}
rows = []
for node in LevelOrderIter(root):
    lineage_pre_syns = []
    lineage_post_syns = []
    for child in node.children:
        descendants = [n.name for n in child.descendants] + [child.name]
        descendant_pre_syns = pre_syns[pre_syns["segment"].isin(descendants)].copy()
        descendant_post_syns = post_syns[post_syns["segment"].isin(descendants)].copy()
        descendant_pre_syns["lineage"] = child.name
        descendant_post_syns["lineage"] = child.name
        lineage_pre_syns.append(descendant_pre_syns)
        lineage_post_syns.append(descendant_post_syns)

    if len(lineage_pre_syns) > 0:
        lineage_pre_syns = pd.concat(lineage_pre_syns)
        pre_syn_lineage_counts = (
            lineage_pre_syns.groupby(["lineage", synapse_label])
            .size()
            .unstack()
            .fillna(0)
        )
        pre_syn_tables_by_node[node.name] = pre_syn_lineage_counts
        pre_syn_total_counts = pre_syn_lineage_counts.sum(axis=1).to_list()
        if not pre_syn_lineage_counts.empty:
            pre_syn_stat, pre_syn_pvalue, _, _ = chi2_contingency(
                pre_syn_lineage_counts
            )
        else:
            pre_syn_stat = None
            pre_syn_pvalue = 1
    else:
        pre_syn_stat = None
        pre_syn_pvalue = 1.0
        pre_syn_total_counts = []

    # if len(lineage_post_syns) > 0:
    #     lineage_post_syns = pd.concat(lineage_post_syns)
    #     post_syn_lineage_counts = (
    #         lineage_post_syns.groupby(["lineage", synapse_label])
    #         .size()
    #         .unstack()
    #         .fillna(0)
    #     )
    #     post_syn_tables_by_node[node.name] = post_syn_lineage_counts
    #     post_syn_total_counts = post_syn_lineage_counts.sum(axis=1).to_list()
    #     if not post_syn_lineage_counts.empty:
    #         post_syn_stat, post_syn_pvalue, _, _ = chi2_contingency(
    #             post_syn_lineage_counts
    #         )
    #     else:
    #         post_syn_stat = None
    #         post_syn_pvalue = 1.0
    # else:
    #     post_syn_stat = None
    #     post_syn_pvalue = 1.0
    #     post_syn_total_counts = []

    rows.append(
        {
            "segment": node.name,
            "pre_syn_stat": pre_syn_stat,
            "pre_syn_pvalue": pre_syn_pvalue,
            "pre_syn_counts": pre_syn_total_counts,
            # "post_syn_stat": post_syn_stat,
            # "post_syn_pvalue": post_syn_pvalue,
            # "post_syn_counts": post_syn_lineage_counts.sum(axis=1).to_list(),
        }
    )


results = pd.DataFrame(rows)
results.set_index("segment", inplace=True)


# %%

condensed_nf.nodes["pre_syn_pvalue"] = results["pre_syn_pvalue"]
# condensed_nf.nodes["post_syn_pvalue"] = results["post_syn_pvalue"]

condensed_nf.nodes["pre_syn_log_pvalue"] = np.log10(
    condensed_nf.nodes["pre_syn_pvalue"]
)
# condensed_nf.nodes["post_syn_log_pvalue"] = np.log10(
#     condensed_nf.nodes["post_syn_pvalue"]
# )
condensed_nf.nodes["pre_syn_log_pvalue_bin"] = np.ceil(
    condensed_nf.nodes["pre_syn_log_pvalue"]
)
# condensed_nf.nodes["post_syn_log_pvalue_bin"] = np.ceil(
#     condensed_nf.nodes["post_syn_log_pvalue"]
# )


# %%

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
sns.histplot(np.log10(condensed_nf.nodes["pre_syn_pvalue"]))

# %%

reds = sns.color_palette("Reds_r", as_cmap=True)

norm = Normalize(vmin=-10, vmax=2)

normed_vals = norm(condensed_nf.nodes["pre_syn_log_pvalue"])

palette = dict(zip(condensed_nf.nodes["pre_syn_log_pvalue"], reds(normed_vals)))

sns.set_context("talk")
fig, axs = plt.subplots(
    5,
    3,
    figsize=(20, 10),
    sharex="col",
    constrained_layout=True,
    gridspec_kw={"width_ratios": [0.02, 0.5, 0.4]},
)


def _get_slice(items, n_items):
    if items is None:
        return slice(0, n_items)
    elif type(items) == int:
        return items
    elif len(items) == 1:
        return items[0]
    else:
        return slice(items[0], items[1])


def merge_axes(fig, axs, rows=None, cols=None):
    # TODO I could't figure out a safer way to do this without eval
    # seems like gridspec.__getitem__ only wants numpy indices in the slicing form
    # REF: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
    row_slice = _get_slice(rows, axs.shape[0])
    col_slice = _get_slice(cols, axs.shape[1])
    gs = axs.flat[0].get_gridspec()
    for ax in axs[row_slice, col_slice].ravel():
        ax.remove()
    ax = fig.add_subplot(gs[row_slice, col_slice])
    return ax


cbar_ax = merge_axes(fig, axs, cols=(0,))
tree_ax = merge_axes(fig, axs, cols=(1,))


networkplot(
    adjacency=condensed_nf.to_sparse_adjacency(),
    node_data=condensed_nf.nodes,
    node_size="descendant_pre_syn_count",
    node_hue="pre_syn_log_pvalue",
    node_sizes=(20, 100),
    palette=palette,
    edge_linewidth=3.0,
    x="pos_x",
    y="pos_y",
    ax=tree_ax,
)
tree_ax.axis("off")

# label the soma
x, y = pos[root_seg]
# ax.scatter([x], [y], marker="^", s=300, zorder=-1, color="lightgrey")
tree_ax.scatter([x], [y], marker="D", s=300, zorder=-1, color="lightgrey")

sm = plt.cm.ScalarMappable(cmap=reds, norm=norm)
sm.set_array([])
fig.colorbar(
    sm, cax=cbar_ax, orientation="vertical", shrink=0.1, label="log10(p-value)"
)
cbar_ax.yaxis.set_ticks_position("left")
cbar_ax.yaxis.set_label_position("left")

most_sig_results = results.sort_values("pre_syn_pvalue", ascending=True)[:5]

for i, (seg, result) in enumerate(most_sig_results.iterrows()):
    ax = axs[i, 2]
    table = pre_syn_tables_by_node[seg].copy()[["axon", "dendrite", "soma"]]
    anno = table.copy().dropna()
    table = table.div(table.sum(axis=1), axis=0)
    sns.heatmap(table, ax=ax, cmap="Blues", yticklabels=False, annot=anno)
    ax.set_title(f"Segment {seg} (p-value: {result['pre_syn_pvalue']:.2e})")
    if i < 4:
        ax.tick_params(axis="x", which="both", length=0)
        ax.set_xlabel("")
    if i == 2:
        ax.set_ylabel("Branch")
    else:
        ax.set_ylabel("")

# draw lines between the plots


for i, (seg, result) in enumerate(most_sig_results.iterrows()):
    start_x, start_y = pos[seg]
    end_x = -0.05
    end_y = 0.5

    con = ConnectionPatch(
        xyA=(start_x, start_y),
        xyB=(end_x, end_y),
        coordsA="data",
        coordsB="axes fraction",
        axesA=tree_ax,
        axesB=axs[i, 2],
        color="grey",
        alpha=0.5,
        zorder=-1,
    )
    tree_ax.add_artist(con)


# %%
from nglui import statebuilder as sb

state_dict = json.loads(
    sb.make_neuron_neuroglancer_link(client, root_id, return_as="json")
)
state_dict["layers"][1]["objectAlpha"] = 0.4
base_sb = sb.StateBuilder(state_dict)
sbs = []
dfs = []

# point_mapper = sb.PointMapper(point_column="ctr_pt_position")
# pre_syn_layer = sb.AnnotationLayerConfig(
#     name="pre_syn",
#     color="lightgreen",
#     mapping_rules=point_mapper,
# )
# pre_syn_sb = sb.StateBuilder(layers=[pre_syn_layer], base_state=state_dict)
# sbs.append(pre_syn_sb)
# dfs.append(pre_syns)

# post_syn_layer = sb.AnnotationLayerConfig(
#     name="post_syn",
#     color=DEFAULT_POSTSYN_COLOR,
#     mapping_rules=point_mapper,
# )
# post_syn_sb = sb.StateBuilder(layers=[post_syn_layer], base_state=state_dict)
# sbs.append(post_syn_sb)
# dfs.append(post_syns)

for pvalue_bin, data in condensed_nf.nodes.groupby("pre_syn_log_pvalue_bin"):
    point_mapper = sb.PointMapper(
        point_column="branch", split_positions=True, description_column="segment"
    )
    branch_point_layer = sb.AnnotationLayerConfig(
        name=pvalue_bin,
        color=to_hex(reds(norm(pvalue_bin))),
        mapping_rules=point_mapper,
    )
    branch_point_sb = sb.StateBuilder(
        layers=[branch_point_layer], base_state=state_dict
    )
    sbs.append(branch_point_sb)
    # branch_point_df = skeleton_nodes.query("is_branch_point").copy()
    # branch_point_df["pt"] = branch_point_df.apply(
    #     lambda x: x[["x", "y", "z"]].values / np.array([4, 4, 40]), axis=1
    # )
    dfs.append(data.query("branch_x.notnull()").reset_index())

final_sb = sb.ChainedStateBuilder(sbs)
final_sb.render_state(dfs, return_as="html")

# %%
# import cloudvolume
# from neuroglancer.coordinate_space import CoordinateSpace
# from neuroglancer.viewer_state import AnnotationPropertySpec
# from neuroglancer.write_annotations import AnnotationWriter

# coord_space = CoordinateSpace(names=["x", "y", "z"], units=["nm"] * 3, scales=[1, 1, 1])

# ann_props = []
# ann_prop = AnnotationPropertySpec(
#     id="synapse", type="uint16", description="pre_synapses"
# )
# ann_props.append(ann_prop)

# writer = AnnotationWriter(
#     coordinate_space=coord_space,
#     annotation_type="point",
#     properties=ann_props,
# )

# for i, row in pre_syns.iterrows():
#     writer.add_point(row["ctr_pt_position"], id=i)

# writer.write("test_pre_synapses")

# base_info = client.chunkedgraph.segmentation_info
# # base_info["skeletons"] = "skeleton"
# info = base_info.copy()
# info["points"] = "points"
# cv = cloudvolume.CloudVolume(
#     "precomputed://gs://allen-minnie-phase3/tempskel", mip=0, info=info, compress=False
# )
# cv.commit_info()

# # %%

# # CloudFiles("file:///tmp").puts("test_pre_synapses", "test_pre_synapses")

# # %%
# import neuroglancer
# import neuroglancer.static_file_server

# server = neuroglancer.static_file_server.StaticFileServer(
#     static_dir=".", bind_address="127.0.0.1", daemon=True
# )

# viewer = neuroglancer.Viewer()
# with viewer.txn() as s:
#     s.dimensions = coord_space
#     # s.layers["pre_syn"] = neuroglancer.PointAnnotationLayer(
#     #     source="precomputed://test_pre_synapses"
#     # )
# viewer

# %%
