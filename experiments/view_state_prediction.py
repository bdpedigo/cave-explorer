# %%

import os
import pickle

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from nglui import statebuilder
from nglui.segmentprops import SegmentProperties
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from tqdm.auto import tqdm

from pkg.constants import DATA_PATH, OUT_PATH
from pkg.plot import savefig, set_context
from pkg.utils import (
    load_casey_palette,
    load_joint_table,
    load_manifest,
    load_mtypes,
    start_client,
)

# %%

fig_out_folder = "view_prediction_heuristics"

set_context()

client = start_client()
mtypes = load_mtypes(client)
manifest = load_manifest()
manifest = manifest.query("in_inhibitory_column")

# %%
joint_table = load_joint_table()

feature_path = DATA_PATH / "state_prediction_info"
feature_path = feature_path / "inhibitory_features.csv"

features = pd.read_csv(feature_path, index_col=0)
features = features.join(joint_table.reset_index().set_index("pt_root_id"), how="left")

relevant_cols = [
    "coarse_type_aibs_metamodel_celltypes_v661_corrections",
    "coarse_type_bodor_pt_target_proofread",
    "coarse_type_aibs_column_nonneuronal_ref",
    "coarse_type_allen_v1_column_types_slanted_ref",
    "coarse_type_aibs_metamodel_celltypes_v661",
]
mask = features[relevant_cols].notna()
row_ilocs, col_ilocs = np.nonzero(mask)

first_ilocs = (
    pd.DataFrame(list(zip(row_ilocs, col_ilocs)), columns=["row_iloc", "col_iloc"])
    .groupby("row_iloc")["col_iloc"]
    .first()
)

final_calls = features[relevant_cols].values[first_ilocs.index, first_ilocs.values]
final_calls = pd.Series(index=features.index[first_ilocs.index], data=final_calls)

features["soma_coarse_type"] = final_calls
features = features.query("soma_coarse_type == 'inhibitory'")

# %%

X = features.copy()[["n_pre_synapses", "n_post_synapses", "n_nodes"]]

# %%

X["n_pre_synapses"].replace(0, 1, inplace=True)
X["n_post_synapses"].replace(0, 1, inplace=True)
X["n_nodes"].replace(0, 1, inplace=True)

X["n_pre_synapses"] = np.log10(X["n_pre_synapses"].astype(float))
X["n_post_synapses"] = np.log10(X["n_post_synapses"].astype(float))
X["n_nodes"] = np.log10(X["n_nodes"].astype(float))

# %%

model_path = OUT_PATH / "train_state_prediction"

with open(model_path / "state_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# %%
y_scores_new = model.decision_function(X[model.feature_names_in_])
y_scores_new = pd.Series(y_scores_new, index=X.index)

fig, ax = plt.subplots(figsize=(6, 6))

sns.histplot(x=y_scores_new, ax=ax)

ax.set_xlabel("Log posterior ratio")

savefig("new_log_posterior_ratio", fig, folder=fig_out_folder, doc_save=True)

# %%
features["log_posterior_ratio"] = y_scores_new

# %%

casey_palette = load_casey_palette()

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.histplot(
    data=features.query('mtype.isin(["DTC", "ITC", "PTC", "STC"])'),
    x="log_posterior_ratio",
    hue="mtype",
    element="step",
    palette=casey_palette,
)
ax.set(xlabel="Log posterior ratio")
sns.move_legend(ax, "upper left")
savefig("log_posterior_ratio_count_by_mtype", fig, folder=fig_out_folder)

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.histplot(
    data=features.query('mtype.isin(["DTC", "ITC", "PTC", "STC"])'),
    x="log_posterior_ratio",
    hue="mtype",
    stat="density",
    common_norm=False,
    element="step",
    palette=casey_palette,
)
sns.move_legend(ax, loc="upper left")
ax.set(xlabel="Log posterior ratio")
savefig("log_posterior_ratio_density_by_mtype", fig, folder=fig_out_folder)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.histplot(
    data=features.query('mtype.isin(["DTC", "ITC", "PTC", "STC"])'),
    x="log_posterior_ratio",
    hue="mtype",
    stat="density",
    common_norm=False,
    element="poly",
    palette=casey_palette,
    cumulative=True,
    fill=False,
)
sns.move_legend(ax, loc="upper left")
ax.set(xlabel="Log posterior ratio")
savefig("log_posterior_ratio_density_by_mtype_cumulative", fig, folder=fig_out_folder)


# %%
bins = np.histogram_bin_edges(y_scores_new, bins="auto")
y_scores_new_binned = pd.cut(y_scores_new, bins=bins, include_lowest=True)
bin_counts = y_scores_new_binned.value_counts()
cumsum = bin_counts.sort_index().cumsum()
survival = len(y_scores_new) - cumsum

fig, ax = plt.subplots(figsize=(6, 6))
sns.lineplot(x=survival.index.categories.mid, y=survival.values, ax=ax)
ax.set_xlabel("Log posterior ratio")

savefig("log_posterior_ratio_survival", fig, folder=fig_out_folder, doc_save=True)


# %%

fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(
    data=features.query("coarse_type == 'inhibitory'"),
    x="log_posterior_ratio",
    hue="proofreading_strategy_axon",
    stat="count",
)
ax.set_xlabel("Log posterior ratio")

savefig("proofread_log_posterior_ratio", fig, folder=fig_out_folder, doc_save=True)

# %%

features["x"] = features["pt_position_x"] * 4
features["y"] = features["pt_position_y"] * 4
features["z"] = features["pt_position_z"] * 40
points = features[["x", "y", "z"]].values
scalars = features["log_posterior_ratio"].values

pv.set_jupyter_backend("client")

plotter = pv.Plotter()
cloud = pv.PolyData(points)
cloud["log_posterior_ratio"] = scalars

plotter.add_mesh(
    cloud,
    scalars="log_posterior_ratio",
    cmap="coolwarm",
    render_points_as_spheres=True,
    point_size=5,
)

plotter.show()
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


synapse_path = DATA_PATH / "state_prediction_info"

files = os.listdir(synapse_path)

pre_synapses = []
for file in tqdm(files):
    if "pre_synapses" in file:
        chunk_pre_synapses = pd.read_csv(synapse_path / file)
        pre_synapses.append(chunk_pre_synapses)


pre_synapses = pd.concat(pre_synapses)

# TODO could actually use the joint table here?
mtypes = load_mtypes(client)

pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(mtypes["cell_type"])


# %%

from matplotlib import colormaps

threshold_feature = "log_posterior_ratio"
method = "ward"
metric = "euclidean"
k = 40
remove_inhib = True
labels_by_threshold = {}
confusion_mats_by_threshold = {}


def get_thresholded_projection_proportions(
    threshold, threshold_feature="log_posterior_ratio", remove_inhib=True
):
    pre_synapses[f"pre_{threshold_feature}"] = pre_synapses["pre_pt_root_id"].map(
        features[threshold_feature]
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

    if remove_inhib:
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

    return props_by_mtype


# %%

thresholds = np.arange(-11, 11, 1).astype(int)
for threshold in thresholds:
    props_by_mtype = get_thresholded_projection_proportions(
        threshold, threshold_feature=threshold_feature, remove_inhib=remove_inhib
    )

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

    color_labels["Posterior"] = features.loc[color_labels.index, "log_posterior_ratio"]
    v = max(
        abs(features["log_posterior_ratio"].min()),
        features["log_posterior_ratio"].max(),
    )
    norm = Normalize(
        vmin=-v,
        vmax=v,
    )
    # colors = [(1, 1, 1), (0, 0, 0)]
    # cmap = LinearSegmentedColormap.from_list("custom", colors)
    cmap = colormaps["coolwarm_r"]
    color_labels["Posterior"] = color_labels["Posterior"].map(lambda x: cmap(norm(x)))

    color_labels["M-type"] = (
        color_labels.index.to_series().map(mtypes["cell_type"]).map(casey_palette)
    )

    norm = Normalize(
        vmin=0,
        # vmax=labeled_inhib_features.loc[color_labels.index, "n_operations"].max(),
        vmax=100,
    )
    colors = [(1, 1, 1), (0, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("custom", colors)

    color_labels["# edits (0-100)"] = features.loc[
        color_labels.index, "n_operations"
    ].map(lambda x: cmap(norm(x)))

    color_labels["Motif group"] = (
        color_labels.index.to_series().map(ctypes).map(palette)
    )

    color_labels["Area"] = features.loc[color_labels.index, "visual_area"]
    color_labels["Area"] = color_labels["Area"].map(
        dict(
            zip(
                np.unique(features["visual_area"]),
                sns.color_palette("tab10", n_colors=10),
            )
        )
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
        folder=fig_out_folder,
        doc_save=True,
    )
    plt.close()

    sort_bys = ["label", "log_posterior_ratio"]
    props_by_mtype_sorted = props_by_mtype.copy()
    props_by_mtype_sorted["label"] = labels.map(new_label_map)
    props_by_mtype_sorted["log_posterior_ratio"] = features.loc[
        props_by_mtype_sorted.index, "log_posterior_ratio"
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
        folder=fig_out_folder,
        doc_save=True,
    )
    plt.close()

    labels_by_threshold[threshold] = labels

    confusion_mat = pd.crosstab(labels, labels.index.map(ctypes), dropna=False)

    confusion_mats_by_threshold[threshold] = confusion_mat


# %%

features.to_csv(OUT_PATH / "labeled_inhib_features.csv")

# %%

confusion_mat = confusion_mats_by_threshold[0]

confusion_mat = confusion_mat.drop(columns=[pd.NA])
annot = confusion_mat.T
annot[annot == 0] = np.nan

fig, ax = plt.subplots(figsize=(15, 7))
sns.heatmap(
    confusion_mat.T,
    cmap="Reds",
    annot=True,
    mask=confusion_mat.T == 0,
    ax=ax,
    annot_kws=dict(size=10),
    xticklabels=True,
    yticklabels=True,
)
ax.set(xlabel="New Cluster", ylabel="Connectivity Motif")

# %%

vis_areas = joint_table.set_index("pt_root_id")["visual_area"]
v1s = vis_areas[vis_areas == "V1"].index
# %%
props_by_mtype = get_thresholded_projection_proportions(0, remove_inhib=True)
props_by_mtype = props_by_mtype.loc[props_by_mtype.index.intersection(v1s)]

from scipy.cluster.hierarchy import dendrogram

sorted_root_ids = (
    joint_table.set_index("pt_root_id")
    .loc[props_by_mtype.index]
    .sort_values("pt_position_x")
    .index
)

props_by_mtype = props_by_mtype.loc[sorted_root_ids]


linkage_matrix = linkage(props_by_mtype, method=method, metric=metric)

dend = dendrogram(linkage_matrix, no_plot=True, color_threshold=-np.inf)

k = 20
labels = pd.Series(
    fcluster(linkage_matrix, k, criterion="maxclust"), index=props_by_mtype.index
)

confusion_mat = pd.crosstab(labels, labels.index.map(ctypes), dropna=True)

annot = confusion_mat.T
annot[annot == 0] = np.nan

fig, ax = plt.subplots(figsize=(15, 7))
sns.heatmap(
    confusion_mat.T,
    cmap="Reds",
    annot=True,
    mask=confusion_mat.T == 0,
    ax=ax,
    annot_kws=dict(size=10),
    xticklabels=True,
    yticklabels=True,
)
ax.set(xlabel="New Cluster", ylabel="Connectivity Motif")

colors = sns.color_palette("tab20", n_colors=k)
palette = dict(zip(np.arange(1, k + 1), colors))
color_labels = labels.map(palette).rename("Cluster").copy().to_frame()
cgrid = sns.clustermap(
    props_by_mtype.T,
    col_linkage=linkage_matrix,
    col_colors=color_labels,
    cmap="Reds",
    row_cluster=False,
    figsize=(20, 10),
    xticklabels=False,
    tree_kws=dict(count_sort=True),
)
cgrid.ax_heatmap.yaxis.tick_left()
cgrid.ax_heatmap.yaxis.label_position = "left"
cgrid.ax_heatmap.set(ylabel="Excitatory M-type", xlabel="Inhibitory Neuron")
cgrid.ax_col_colors.yaxis.tick_left()
cgrid.ax_col_colors.set_yticklabels(cgrid.ax_col_colors.get_yticklabels(), rotation=0)


inds = cgrid.dendrogram_col.reordered_ind
sorted_labels = labels.iloc[inds].copy().rename("cluster").to_frame()
sorted_labels["iloc"] = np.arange(len(sorted_labels))
positions = sorted_labels.groupby("cluster")["iloc"].mean()

for label, position in positions.items():
    cgrid.ax_col_colors.text(
        position, 0.5, label, ha="center", va="center", fontsize=12
    )


# %%

from hyppo.ksample import KSample

props_by_mtype = get_thresholded_projection_proportions(-np.inf, remove_inhib=True)
props_by_mtype = props_by_mtype.loc[props_by_mtype.index.intersection(v1s)]

X_casey = props_by_mtype.loc[
    props_by_mtype.index.intersection(manifest.query("in_inhibitory_column").index)
]

rows = []
for threshold in tqdm(thresholds):
    props_by_mtype = get_thresholded_projection_proportions(
        threshold, remove_inhib=True
    )
    props_by_mtype = props_by_mtype.loc[props_by_mtype.index.intersection(v1s)]

    X_new = props_by_mtype.loc[props_by_mtype.index.difference(X_casey.index)]

    ks = KSample("Dcorr")

    stat, pvalue = ks.test(X_casey.values, X_new.values)

    row = {"threshold": threshold, "stat": stat, "pvalue": pvalue}
    rows.append(row)

ks_results = pd.DataFrame(rows)

# %%

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, constrained_layout=True)

sns.lineplot(data=ks_results, x="threshold", y="stat", ax=axs[0])

sns.lineplot(data=ks_results, x="threshold", y="pvalue", ax=axs[1])

# %%

from umap import UMAP

props_by_mtype = get_thresholded_projection_proportions(-10, remove_inhib=True)
props_by_mtype.loc[props_by_mtype.index.intersection(v1s)]

X = props_by_mtype.values

umap = UMAP(n_components=2, min_dist=0.2, n_neighbors=20)

embedding = umap.fit_transform(X)

embedding = pd.DataFrame(
    embedding, index=props_by_mtype.index, columns=["UMAP1", "UMAP2"]
)

features = features.drop(columns=["UMAP1", "UMAP2"]).join(embedding, how="left")


features["ctype"] = ctypes.astype(str).fillna("Unknown").str.replace("<NA>", "Unknown")
features["ctype"] = features["ctype"].fillna("Unknown")

# %%
fig, ax = plt.subplots(figsize=(10, 10))

palette = dict(
    zip(
        np.unique(features["ctype"]),
        sns.color_palette("husl", n_colors=features["ctype"].nunique()),
    )
)
palette["Unknown"] = (0.2, 0.2, 0.2)

features["is_known"] = (features["ctype"] != "Unknown").astype(int) * 5
sns.scatterplot(
    data=features.sort_values("ctype", ascending=False),
    x="UMAP1",
    y="UMAP2",
    hue="ctype",
    ax=ax,
    linewidth=1,
    palette=palette,
    size="is_known",
    sizes=(10, 100),
)
sns.move_legend(ax, loc="upper right", bbox_to_anchor=(1.3, 1))


# %%

seg_df = features.copy()
seg_df["log_odds"] = np.nan
seg_df.loc[y_scores_new.index, "log_odds"] = y_scores_new
seg_df = seg_df.dropna()
seg_df["log_odds_rank"] = seg_df["log_odds"].rank()

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

seg_df = features.copy()
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
