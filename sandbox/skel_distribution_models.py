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
nucs.drop_duplicates(subset="pt_root_id", keep=False, inplace=True)
nucs.set_index("pt_root_id", inplace=True)

# %%
# 864691136100014453 has an interesting long range projection that gets picked up
# 864691135759685966 has a region of tangles, kinda
# 864691135234167385 another interesting long range projection
# 864691135082074359 not much going on here besides the soma
# 864691135509890057 many sigs
# 864691135082074359 not much going on here besides the soma

root_id = proofreading_table.sample(1)["pt_root_id"].values[0]
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
            lineage_pre_syns.groupby(["lineage", "post_mtype"])
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

    if len(lineage_post_syns) > 0:
        lineage_post_syns = pd.concat(lineage_post_syns)
        post_syn_lineage_counts = (
            lineage_post_syns.groupby(["lineage", "pre_mtype"])
            .size()
            .unstack()
            .fillna(0)
        )
        post_syn_tables_by_node[node.name] = post_syn_lineage_counts
        post_syn_total_counts = post_syn_lineage_counts.sum(axis=1).to_list()
        if not post_syn_lineage_counts.empty:
            post_syn_stat, post_syn_pvalue, _, _ = chi2_contingency(
                post_syn_lineage_counts
            )
        else:
            post_syn_stat = None
            post_syn_pvalue = 1.0
    else:
        post_syn_stat = None
        post_syn_pvalue = 1.0
        post_syn_total_counts = []

    rows.append(
        {
            "segment": node.name,
            "pre_syn_stat": pre_syn_stat,
            "pre_syn_pvalue": pre_syn_pvalue,
            "pre_syn_counts": pre_syn_total_counts,
            "post_syn_stat": post_syn_stat,
            "post_syn_pvalue": post_syn_pvalue,
            "post_syn_counts": post_syn_lineage_counts.sum(axis=1).to_list(),
        }
    )


results = pd.DataFrame(rows)
results.set_index("segment", inplace=True)


# %%

condensed_nf.nodes["pre_syn_pvalue"] = results["pre_syn_pvalue"]
condensed_nf.nodes["post_syn_pvalue"] = results["post_syn_pvalue"]

condensed_nf.nodes["pre_syn_log_pvalue"] = np.log10(
    condensed_nf.nodes["pre_syn_pvalue"]
)
condensed_nf.nodes["post_syn_log_pvalue"] = np.log10(
    condensed_nf.nodes["post_syn_pvalue"]
)
condensed_nf.nodes["pre_syn_log_pvalue_bin"] = np.ceil(
    condensed_nf.nodes["pre_syn_log_pvalue"]
)
condensed_nf.nodes["post_syn_log_pvalue_bin"] = np.ceil(
    condensed_nf.nodes["post_syn_log_pvalue"]
)


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
    table = pre_syn_tables_by_node[seg].copy()
    table = table.div(table.sum(axis=1), axis=0)
    sns.heatmap(table, ax=ax, cmap="Blues", yticklabels=False)
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

state_dict = json.loads(
    sb.make_neuron_neuroglancer_link(client, root_id, return_as="json")
)
state_dict["layers"][1]["objectAlpha"] = 0.2
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
