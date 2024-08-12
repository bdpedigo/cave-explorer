# %%

import os
import pickle
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from nglui import statebuilder
from nglui.segmentprops import SegmentProperties
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from pkg.constants import OUT_PATH
from pkg.plot import savefig, set_context
from pkg.utils import load_casey_palette, load_manifest, load_mtypes, start_client

# %%

file_name = "state_prediction_heuristics"

set_context()

client = start_client()
mtypes = load_mtypes(client)
manifest = load_manifest()
manifest = manifest.query("in_inhibitory_column")


# %%
inhib_mtypes = mtypes.query("classification_system == 'inhibitory_neuron'")
all_inhib_roots = inhib_mtypes.index

out_path = Path("data/synapse_pull")

files = os.listdir(out_path)

pre_synapses = []
for file in tqdm(files):
    if "pre_synapses" in file:
        chunk_pre_synapses = pd.read_csv(out_path / file)
        pre_synapses.append(chunk_pre_synapses)
pre_synapses = pd.concat(pre_synapses)

post_synapses = []
for file in tqdm(files):
    if "post_synapses" in file:
        chunk_post_synapses = pd.read_csv(out_path / file)
        post_synapses.append(chunk_post_synapses)
post_synapses = pd.concat(post_synapses)

pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(mtypes["cell_type"])

pre_synapses = pre_synapses.query("pre_pt_root_id != post_pt_root_id")
post_synapses = post_synapses.query("pre_pt_root_id != post_pt_root_id")

n_pre_synapses = (
    pre_synapses.groupby("pre_pt_root_id").size().reindex(all_inhib_roots).fillna(0)
)
n_post_synapses = (
    post_synapses.groupby("post_pt_root_id").size().reindex(all_inhib_roots).fillna(0)
)

# %%


def get_n_leaves_for_root(root_id):
    return len(client.chunkedgraph.get_leaves(root_id, stop_layer=2))


if os.path.exists(out_path / "n_leaves.csv"):
    n_leaves = pd.read_csv(out_path / "n_leaves.csv", index_col=0)["0"]
    n_leaves.name = "n_nodes"
else:
    with tqdm_joblib(
        tqdm(desc="Getting n_leaves", total=len(all_inhib_roots))
    ) as progress:
        n_leaves = Parallel(n_jobs=-1)(
            delayed(get_n_leaves_for_root)(root_id) for root_id in all_inhib_roots
        )
        n_leaves = pd.Series(n_leaves, index=all_inhib_roots)
        n_leaves.to_csv(out_path / "n_leaves.csv")

# %%
inhib_features = pd.concat([n_pre_synapses, n_post_synapses, n_leaves], axis=1)
inhib_features.columns = ["n_pre_synapses", "n_post_synapses", "n_nodes"]

# %%


def get_change_info(root_id: int, n_tries: int = 3) -> dict:
    try:
        out = client.chunkedgraph.get_change_log(root_id, filtered=True)
        out.pop("operations_ids")
        out.pop("past_ids")
        out.pop("user_info")
        return out
    except Exception:
        return get_change_info(root_id, n_tries - 1) if n_tries > 0 else {}


from tqdm_joblib import tqdm_joblib

with tqdm_joblib(
    tqdm(desc="Getting change info", total=len(inhib_features))
) as progress:
    changelog_infos = Parallel(n_jobs=-1)(
        delayed(get_change_info)(root_id) for root_id in inhib_features.index
    )

changes_df = pd.DataFrame(changelog_infos, index=inhib_features.index)
changes_df["n_operations"] = changes_df["n_mergers"] + changes_df["n_splits"]

inhib_features = inhib_features.join(changes_df)

# %%

X_new = inhib_features.copy()

# %%%

X_new["n_pre_synapses"].replace(0, 1, inplace=True)
X_new["n_post_synapses"].replace(0, 1, inplace=True)
X_new["n_nodes"].replace(0, 1, inplace=True)

X_new["n_pre_synapses"] = np.log10(X_new["n_pre_synapses"].astype(float))
X_new["n_post_synapses"] = np.log10(X_new["n_post_synapses"].astype(float))
X_new["n_nodes"] = np.log10(X_new["n_nodes"].astype(float))


# %%

model_path = OUT_PATH / file_name

with open(model_path / "state_prediction_model.pkl", "rb") as f:
    lda = pickle.load(f)

# %%
y_scores_new = lda.decision_function(X_new)
y_scores_new = pd.Series(y_scores_new, index=X_new.index)

fig, ax = plt.subplots(figsize=(6, 6))

sns.histplot(x=y_scores_new, ax=ax)

ax.set_xlabel("Log posterior ratio")

savefig("new_log_posterior_ratio", fig, file_name, doc_save=True)

# %%
inhib_features["log_posterior_ratio"] = y_scores_new
inhib_features["mtype"] = inhib_features.index.map(mtypes["cell_type"])

# %%

casey_palette = load_casey_palette()

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.histplot(
    data=inhib_features,
    x="log_posterior_ratio",
    hue="mtype",
    element="step",
    palette=casey_palette,
)
ax.set(xlabel="Log posterior ratio")
sns.move_legend(ax, "upper left")
savefig("log_posterior_ratio_count_by_mtype", fig, folder=file_name)

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.histplot(
    data=inhib_features,
    x="log_posterior_ratio",
    hue="mtype",
    stat="density",
    common_norm=False,
    element="step",
    palette=casey_palette,
)
sns.move_legend(ax, loc="upper left")
ax.set(xlabel="Log posterior ratio")
savefig("log_posterior_ratio_density_by_mtype", fig, folder=file_name)

# %%
bins = np.histogram_bin_edges(y_scores_new, bins="auto")
y_scores_new_binned = pd.cut(y_scores_new, bins=bins, include_lowest=True)
bin_counts = y_scores_new_binned.value_counts()
cumsum = bin_counts.sort_index().cumsum()
survival = len(y_scores_new) - cumsum

fig, ax = plt.subplots(figsize=(6, 6))
sns.lineplot(x=survival.index.categories.mid, y=survival.values, ax=ax)
ax.set_xlabel("Log posterior ratio")

savefig("log_posterior_ratio_survival", fig, file_name, doc_save=True)

# %%

with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(
        x=survival.index.categories.mid,
        y=survival.values,
        ax=ax,
        # color="darkblue",
        linewidth=3,
        label="Survival",
    )
    ax.set_xlabel("Log posterior ratio")
    sns.move_legend(ax, loc="lower left")

savefig("log_posterior_ratio_survival_precision_recall", fig, file_name, doc_save=True)


# %%


seg_df = inhib_features.copy()
seg_df["log_odds"] = np.nan
seg_df.loc[y_scores_new.index, "log_odds"] = y_scores_new
seg_df = seg_df.dropna()
seg_df["log_odds_rank"] = seg_df["log_odds"].rank()

# %%
n_randoms = 10
for i in range(n_randoms):
    seg_df[f"random_{i}"] = np.random.uniform(0, 1, size=len(seg_df))

seg_prop = SegmentProperties.from_dataframe(
    seg_df.reset_index(),
    id_col="pt_root_id",
    label_col="pt_root_id",
    number_cols=[
        "n_pre_synapses",
        "n_post_synapses",
        "n_nodes",
        "log_odds",
        "log_odds_quant",
        "log_odds_rank",
    ]
    + [f"random_{i}" for i in range(n_randoms)],
)

prop_id = client.state.upload_property_json(seg_prop.to_dict())
prop_url = client.state.build_neuroglancer_url(
    prop_id, format_properties=True, target_site="mainline"
)


img = statebuilder.ImageLayerConfig(
    source=client.info.image_source(),
)

seg = statebuilder.SegmentationLayerConfig(
    source=client.info.segmentation_source(),
    segment_properties=prop_url,
)

sb = statebuilder.StateBuilder(
    layers=[img, seg],
    target_site="mainline",
    view_kws={"zoom_3d": 0.001, "zoom_image": 0.00000001},
)
sb.render_state()


# %%

proofreading_df = client.materialize.query_table("proofreading_status_and_strategy")

proofreading_df.set_index("pt_root_id", inplace=True)

seg_df = seg_df.join(
    proofreading_df[["strategy_dendrite", "strategy_axon"]], how="left"
)

# %%

fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(data=seg_df, x="log_odds", hue="strategy_axon", stat="count")
ax.set_xlabel("Log posterior ratio")

savefig("proofread_log_posterior_ratio", fig, file_name, doc_save=True)

# %%
nucleus_df = client.materialize.query_table("nucleus_detection_v0")
nucleus_df = nucleus_df.set_index("pt_root_id")
nucleus_df = nucleus_df.loc[seg_df.index]

# %%
nucleus_df["x"] = nucleus_df["pt_position"].apply(lambda x: x[0]) * 4
nucleus_df["y"] = nucleus_df["pt_position"].apply(lambda x: x[1]) * 4
nucleus_df["z"] = nucleus_df["pt_position"].apply(lambda x: x[2]) * 40

# %%
if "x" in seg_df.columns:
    seg_df = seg_df.drop(columns=["x", "y", "z"]).join(nucleus_df[["x", "y", "z"]])
else:
    seg_df = seg_df.join(nucleus_df[["x", "y", "z"]])


# %%
points = seg_df[["x", "y", "z"]].values
scalars = seg_df["log_odds"].values


pv.set_jupyter_backend("client")

plotter = pv.Plotter()
cloud = pv.PolyData(points)
cloud["log_odds"] = scalars

plotter.add_mesh(
    cloud,
    scalars="log_odds",
    cmap="coolwarm",
    render_points_as_spheres=True,
    point_size=5,
)

plotter.show()

# %%

labeled_inhib_features = inhib_features.copy()
labeled_inhib_features["log_odds"] = np.nan
labeled_inhib_features.loc[y_scores_new.index, "log_odds"] = y_scores_new
labeled_inhib_features = labeled_inhib_features.dropna()

# %%


# TODO add casey's cells back in here, including those which were misclassified in the
# meta model thingy
# include them in the clustering regardless of the posterior
# see what cluster centroids look like relative to these cells
# %%


manifest = load_manifest()

ctypes = manifest["ctype"]

casey_palette = load_casey_palette()

# %%

threshold_feature = "log_odds"
method = "ward"
metric = "euclidean"
k = 40
remove_inhib = True
labels_by_threshold = {}
for threshold in np.arange(0, 11, 1).astype(int):
    pre_synapses[f"pre_{threshold_feature}"] = pre_synapses["pre_pt_root_id"].map(
        labeled_inhib_features[threshold_feature]
    )

    pre_synapses_by_threshold = pre_synapses.query(
        f"pre_{threshold_feature} > {threshold}"
    )

    projection_counts = pre_synapses_by_threshold.groupby("pre_pt_root_id")[
        "post_mtype"
    ].value_counts()

    if remove_inhib:
        projection_counts = projection_counts.drop(
            labels=["DTC", "ITC", "PTC", "STC"], level="post_mtype"
        )

    # turn counts into proportions
    projection_props = (
        projection_counts / projection_counts.groupby("pre_pt_root_id").sum()
    )

    props_by_mtype = projection_props.unstack().fillna(0)

    label_order = [
        "L2a",
        "L2b",
        "L2c",
        "L3a",
        "L3b",
        "L4a",
        "L4b",
        "L4c",
        "L5a",
        "L5b",
        "L5ET",
        "L5NP",
        "L6short-a",
        "L6short-b",
        "L6tall-a",
        "L6tall-b",
        "L6tall-c",
    ]

    props_by_mtype = props_by_mtype.loc[:, label_order]

    linkage_matrix = linkage(props_by_mtype, method=method, metric=metric)

    # leaves = leaves_list(linkage_matrix)[::-1]

    dend = dendrogram(linkage_matrix, no_plot=True, color_threshold=-np.inf)
    leaves = dend["leaves"]

    props_by_mtype = props_by_mtype.iloc[leaves]

    linkage_matrix = linkage(props_by_mtype, method=method, metric=metric)
    labels = pd.Series(
        fcluster(linkage_matrix, k, criterion="maxclust"), index=props_by_mtype.index
    )

    weighting = props_by_mtype.copy()
    weighting.columns = np.arange(len(weighting.columns))
    weighting["label"] = labels

    means = weighting.groupby("label").mean()
    label_ranking = means.mul(means.columns).sum(axis=1).sort_values()

    new_label_map = dict(zip(labels, label_ranking.index.get_indexer_for(labels)))

    set_context(font_scale=2)

    # colors = sns.color_palette("tab20", n_colors=k)
    colors = cc.glasbey_light
    palette = dict(zip(np.arange(1, k + 1), colors))

    color_labels = labels.map(palette).rename("Cluster").copy()

    color_labels = color_labels.to_frame()

    color_labels["Posterior quantile"] = labeled_inhib_features.loc[
        color_labels.index, "log_odds_quant"
    ]

    color_labels["M-type"] = (
        color_labels.index.to_series().map(mtypes["cell_type"]).map(casey_palette)
    )

    norm = Normalize(
        vmin=labeled_inhib_features.loc[color_labels.index, "log_odds_quant"].min(),
        vmax=labeled_inhib_features.loc[color_labels.index, "log_odds_quant"].max(),
    )
    colors = [(1, 1, 1), (0, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("custom", colors)

    color_labels["Posterior quantile"] = color_labels["Posterior quantile"].map(
        lambda x: cmap(norm(x))
    )

    norm = Normalize(
        vmin=0,
        # vmax=labeled_inhib_features.loc[color_labels.index, "n_operations"].max(),
        vmax=100,
    )
    colors = [(1, 1, 1), (0, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("custom", colors)

    color_labels["# edits (0-100)"] = labeled_inhib_features.loc[
        color_labels.index, "n_operations"
    ].map(lambda x: cmap(norm(x)))

    color_labels["Motif group"] = (
        color_labels.index.to_series().map(ctypes).map(palette)
    )

    props_by_mtype.to_csv(
        OUT_PATH / f"exc_projection_proportions_by_mtype_threshold={threshold}.csv"
    )

    cgrid = sns.clustermap(
        props_by_mtype.T,
        cmap="Reds",
        figsize=(25, 10),
        row_cluster=False,
        col_colors=color_labels,
        col_linkage=linkage_matrix,
        xticklabels=False,
        cbar_pos=None,
        dendrogram_ratio=(0, 0.2),
    )
    ax = cgrid.ax_heatmap

    cgrid.figure.suptitle(
        f"{threshold_feature} threshold = {threshold}", fontsize="x-large", y=1.02
    )

    rect = Rectangle(
        (0, props_by_mtype.shape[1] + 2),
        200,
        1,
        fill=True,
        edgecolor="black",
        facecolor="black",
        lw=1,
        clip_on=False,
    )
    ax.add_patch(rect)

    text = ax.text(
        200,
        props_by_mtype.shape[1] + 2.5,
        f"  200 neurons ({props_by_mtype.shape[0]} total)",
        ha="left",
        va="center",
        clip_on=False,
    )

    # move the y-axis labels to the left
    ax.yaxis.tick_left()
    ax.set(ylabel="Excitatory Neuron Class", xlabel="Inhibitory Neuron")
    ax.yaxis.set_label_position("left")

    props_by_mtype = props_by_mtype.iloc[cgrid.dendrogram_col.reordered_ind]
    props_by_mtype["label"] = labels.map(new_label_map)
    shifts = props_by_mtype["label"] != props_by_mtype["label"].shift()
    shifts.iloc[0] = False
    shifts = shifts[shifts].index
    shift_ilocs = props_by_mtype.index.get_indexer_for(shifts)
    for shift in shift_ilocs:
        ax.axvline(shift, color="black", lw=0.5)
    ax.set(xlabel="Inhibitory Neuron", ylabel="Excitatory Neuron Class")

    label_pos = labels.rename("label").to_frame().copy().loc[props_by_mtype.index]
    label_pos["pos"] = np.arange(len(label_pos))
    positions_by_label = label_pos.groupby("label")["pos"].mean()

    ax.set_xticks(positions_by_label.values)
    ax.set_xticklabels(positions_by_label.index)

    props_by_mtype.drop("label", axis=1, inplace=True)

    y_borders = [3, 5, 8, 12, 14]
    for y_border in y_borders:
        ax.axhline(y_border, color="black", lw=0.5)

    cgrid.ax_col_colors.yaxis.tick_left()

    savefig(
        f"excitatory_inhibitory_clustermap_w_tree-threshold={threshold}-k={k}-metric={metric}-method={method}",
        cgrid.figure,
        folder="state_prediction_heuristics",
        doc_save=True,
    )

    sort_bys = ["label", "log_odds_quant"]
    props_by_mtype_sorted = props_by_mtype.copy()
    props_by_mtype_sorted["label"] = labels.map(new_label_map)
    props_by_mtype_sorted["log_odds_quant"] = labeled_inhib_features.loc[
        props_by_mtype_sorted.index, "log_odds_quant"
    ]
    props_by_mtype_sorted = props_by_mtype_sorted.sort_values(sort_bys)
    props_by_mtype_sorted.drop(sort_bys, axis=1, inplace=True)
    color_labels = color_labels.loc[props_by_mtype_sorted.index]

    cgrid = sns.clustermap(
        props_by_mtype_sorted.T,
        cmap="Reds",
        figsize=(25, 10),
        row_cluster=False,
        col_colors=color_labels,
        col_cluster=False,
        xticklabels=False,
        cbar_pos=None,
        dendrogram_ratio=(0, 0),
    )
    ax = cgrid.ax_heatmap

    cgrid.figure.suptitle(
        f"{threshold_feature} threshold = {threshold}", fontsize="x-large", y=1.04
    )

    rect = Rectangle(
        (0, props_by_mtype.shape[1] + 2),
        200,
        1,
        fill=True,
        edgecolor="black",
        facecolor="black",
        lw=1,
        clip_on=False,
    )
    ax.add_patch(rect)

    text = ax.text(
        200,
        props_by_mtype.shape[1] + 2.5,
        f"  200 neurons ({props_by_mtype.shape[0]} total)",
        ha="left",
        va="center",
        clip_on=False,
    )

    # move the y-axis labels to the left
    ax.yaxis.tick_left()
    ax.set(ylabel="Excitatory Neuron Class", xlabel="Inhibitory Neuron")
    ax.yaxis.set_label_position("left")

    props_by_mtype_sorted["label"] = labels.map(new_label_map)
    shifts = props_by_mtype_sorted["label"] != props_by_mtype_sorted["label"].shift()
    shifts.iloc[0] = False
    shifts = shifts[shifts].index
    shift_ilocs = props_by_mtype_sorted.index.get_indexer_for(shifts)
    for shift in shift_ilocs:
        ax.axvline(shift, color="black", lw=0.5)
    ax.set(xlabel="Inhibitory Neuron", ylabel="Excitatory Neuron Class")

    # final touches
    label_pos = (
        labels.rename("label").to_frame().copy().loc[props_by_mtype_sorted.index]
    )
    label_pos["pos"] = np.arange(len(label_pos))
    positions_by_label = label_pos.groupby("label")["pos"].mean()

    ax.set_xticks(positions_by_label.values)
    ax.set_xticklabels(positions_by_label.index)

    props_by_mtype_sorted.drop("label", axis=1, inplace=True)

    y_borders = [3, 5, 8, 12, 14]
    for y_border in y_borders:
        ax.axhline(y_border, color="black", lw=0.5)

    cgrid.ax_col_colors.yaxis.tick_left()

    savefig(
        f"excitatory_inhibitory_clustermap_sorted-threshold={threshold}-k={k}-metric={metric}-method={method}",
        cgrid.figure,
        folder="state_prediction_heuristics",
        doc_save=True,
    )

    labels_by_threshold[threshold] = labels

# %%
props_by_mtype.to_csv(
    OUT_PATH / f"projection_proportions_by_mtype_threshold={threshold}.csv"
)

# %%

labeled_inhib_features.to_csv(OUT_PATH / "labeled_inhib_features.csv")

# %%

seg_df = inhib_features.copy()
seg_df["log_odds"] = np.nan
seg_df.loc[y_scores_new.index, "log_odds"] = y_scores_new
seg_df = seg_df.dropna()

for threshold, labels in labels_by_threshold.items():
    seg_df[f"label_at_{threshold}"] = labels.fillna(-1).astype(int)

n_randoms = 5
for i in range(n_randoms):
    seg_df[f"random_{i}"] = np.random.uniform(0, 1, size=len(seg_df))

seg_prop = SegmentProperties.from_dataframe(
    seg_df.reset_index(),
    id_col="pt_root_id",
    label_col="mtype",
    tag_value_cols=[
        f"label_at_{threshold}" for threshold in labels_by_threshold.keys()
    ],
    number_cols=[
        "n_pre_synapses",
        "n_post_synapses",
        "n_nodes",
        "log_odds",
        "log_odds_quant",
    ]
    + [f"random_{i}" for i in range(n_randoms)],
)

prop_id = client.state.upload_property_json(seg_prop.to_dict())
prop_url = client.state.build_neuroglancer_url(
    prop_id, format_properties=True, target_site="mainline"
)


img = statebuilder.ImageLayerConfig(
    source=client.info.image_source(),
)

seg = statebuilder.SegmentationLayerConfig(
    source=client.info.segmentation_source(),
    segment_properties=prop_url,
    skeleton_source="precomputed://middleauth+https://minnie.microns-daf.com/skeletoncache/api/v1/minnie65_phase3_v1/precomputed/skeleton",
)

sb = statebuilder.StateBuilder(
    layers=[img, seg],
    target_site="mainline",
    view_kws={"zoom_3d": 0.001, "zoom_image": 0.00000001},
)
sb.render_state(return_as="url")
