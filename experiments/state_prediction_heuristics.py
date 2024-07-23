# %%

import os
import pickle
from itertools import chain, combinations
from pathlib import Path

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from giskard.plot import MatrixGrid
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nglui import statebuilder
from nglui.segmentprops import SegmentProperties
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from pkg.constants import MTYPES_TABLE, OUT_PATH
from pkg.plot import savefig, set_context
from pkg.utils import load_casey_palette, load_manifest, load_mtypes, start_client

# %%

file_name = "state_prediction_heuristics"

set_context()

client = cc.CAVEclient("minnie65_phase3_v1")
mtypes = load_mtypes(client)

distance_colors = sns.color_palette("Set1", n_colors=4)
distance_palette = dict(
    zip(["euclidean", "cityblock", "jensenshannon", "cosine"], distance_colors)
)
# TODO whether to implement this as a table of tables, one massive table...
# nothing really feels satisfying here
# perhaps a table of tables will work, and it can infill the index onto those tables
# before doing a join or concat

# TODO key the elements in the sequence on something other than metaoperation_id, this
# will make it easier to join with the time-ordered dataframes which use "operation_id",
# or do things like take "bouts" for computing metrics which are not tied to a specific
# operation_id

with open(OUT_PATH / "sequence_metrics" / "all_infos.pkl", "rb") as f:
    all_infos = pickle.load(f)
with open(OUT_PATH / "sequence_metrics" / "meta_features_df.pkl", "rb") as f:
    meta_features_df = pickle.load(f)

manifest = load_manifest()
manifest = manifest.query("in_inhibitory_column")

all_infos["mtype"] = all_infos["root_id"].map(manifest["mtype"])

all_infos = all_infos.set_index(
    ["root_id", "scheme", "order_by", "random_seed", "order"]
)
all_infos = all_infos.loc[
    all_infos.index.get_level_values("root_id").intersection(manifest.index)
]


# %%


def compute_diffs_to_final(sequence_df):
    # sequence = sequence_df.index.get_level_values("sequence").unique()[0]
    final_row_idx = sequence_df.index.get_level_values("order").max()
    final_row = sequence_df.loc[final_row_idx].fillna(0).values.reshape(1, -1)
    X = sequence_df.fillna(0).values

    sample_wise_metrics = []
    for metric in ["euclidean", "cityblock", "jensenshannon", "cosine"]:
        distances = cdist(X, final_row, metric=metric)
        distances = pd.Series(
            distances.flatten(),
            name=metric,
            index=sequence_df.index.get_level_values("order"),
        )
        sample_wise_metrics.append(distances)
    sample_wise_metrics = pd.concat(sample_wise_metrics, axis=1)

    return sample_wise_metrics


meta_diff_df = pd.DataFrame(
    index=meta_features_df.index, columns=meta_features_df.columns, dtype=object
)

all_sequence_diffs = {}
for sequence_label, row in meta_features_df.iterrows():
    sequence_diffs = {}
    for metric_label, df in row.items():
        old_index = df.index
        df = df.loc[sequence_label]
        df.index = df.index.droplevel(0)
        diffs = compute_diffs_to_final(df)
        diffs.index = old_index
        sequence_diffs[metric_label] = diffs
    all_sequence_diffs[sequence_label] = sequence_diffs

meta_diff_df = pd.DataFrame(all_sequence_diffs).T
meta_diff_df.index.names = ["root_id", "scheme", "order_by", "random_seed"]

meta_diff_df

# %%
scheme = "historical"

fig, axs = plt.subplots(
    4, 6, figsize=(15, 12), sharex="col", sharey=True, constrained_layout=True
)

synapse_bins = np.linspace(0, 1000, 6).tolist()
synapse_bins.append(1000000)

for i, (mtype, group_info) in enumerate(all_infos.groupby("mtype")):
    group_info = group_info.copy()
    group_root_ids = group_info.index.get_level_values("root_id").unique()

    idx = pd.IndexSlice
    diffs = meta_diff_df.loc[idx[group_root_ids, scheme], :]["props_by_mtype"].values
    diffs = pd.concat(diffs, axis=0).copy().reset_index()
    group_info = group_info.query(f"scheme == '{scheme}'")
    group_info = group_info.reset_index()
    if scheme == "clean-and-merge":
        group_info["random_seed"] = (
            group_info["random_seed"].astype(
                "str"
            )  # replace({"None": np.nan}).astype(float)
        )
    group_info.set_index(["root_id", "random_seed", "order"], inplace=True)
    diffs.set_index(["root_id", "random_seed", "order"], inplace=True)

    diffs["n_pre_synapses"] = group_info["n_pre_synapses"]
    diffs["path_length"] = group_info["path_length"]
    diffs["n_post_synapses"] = group_info["n_post_synapses"]
    diffs["n_nodes"] = group_info["n_nodes"]
    diffs["cumulative_n_operations"] = group_info["cumulative_n_operations"]

    diffs["synapse_count_bin"] = pd.cut(
        diffs["n_pre_synapses"], synapse_bins, right=False
    )
    for j, (syn_bin, syn_group) in enumerate(diffs.groupby("synapse_count_bin")):
        sns.histplot(data=syn_group, y="euclidean", ax=axs[i, j], stat="density")
        axs[i, j].set_xticks([])
        axs[0, j].set_title(f"# out synapses\n {syn_bin}")

    ax = axs[i, 0]
    ax.text(-1, 0.5, mtype, transform=ax.transAxes, fontsize="xx-large", va="center")
    ax.set(ylabel="Euclidean distance \n to final output profile")

# %%

scheme = "historical"
group_info = all_infos.copy()

idx = pd.IndexSlice
diffs = meta_diff_df.loc[idx[:, scheme], :]["props_by_mtype"].values
diffs = pd.concat(diffs, axis=0).copy().reset_index()

group_info = group_info.copy()
group_root_ids = group_info.index.get_level_values("root_id").unique()
group_info = group_info.query(f"scheme == '{scheme}'")
group_info = group_info.reset_index()

group_info.set_index(["root_id", "random_seed", "order"], inplace=True)
diffs.set_index(["root_id", "random_seed", "order"], inplace=True)

diffs = diffs.join(
    group_info[
        [
            "n_pre_synapses",
            "path_length",
            "n_post_synapses",
            "n_nodes",
            "cumulative_n_operations",
        ]
    ]
)


# %%

fine_synapse_bins = np.linspace(0, 2000, 101)

objects = "neurons"
thresh = 0.2
metric = "euclidean"
rows = []
for bin in fine_synapse_bins:
    sub_diffs = diffs.query(f"n_pre_synapses >= {bin}")
    if objects == "neurons":
        idx = sub_diffs.groupby("root_id")["cumulative_n_operations"].idxmin()
        sub_diffs = sub_diffs.loc[idx]
    prop = (sub_diffs[metric] < thresh).mean()
    rows.append({"n_pre_synapses": bin, "prop": prop})

prop_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(6, 6))
sns.lineplot(data=prop_df, x="n_pre_synapses", y="prop", ax=ax)
ax.set(
    ylabel=f"Proportion of {objects} with\n{metric} < {thresh}",
    xlabel="Number of pre-synapses",
)

# %%
fine_synapse_bins = np.linspace(0, 2000, 201)

object = "neurons"
rows = []
for bin in fine_synapse_bins:
    sub_diffs = diffs.query(f"n_pre_synapses >= {bin}")
    if objects == "neurons":
        idx = sub_diffs.groupby("root_id")["cumulative_n_operations"].idxmin()
        sub_diffs = sub_diffs.loc[idx]
    cell_types = sub_diffs.index.get_level_values("root_id").map(manifest["mtype"])
    sub_diffs["cell_type"] = cell_types
    for cell_type, cell_group in sub_diffs.groupby("cell_type"):
        prop = (cell_group[metric] < thresh).mean()
        rows.append({"n_pre_synapses": bin, "prop": prop, "cell_type": cell_type})

prop_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(6, 6))
sns.lineplot(data=prop_df, x="n_pre_synapses", y="prop", hue="cell_type", ax=ax)
ax.set(
    ylabel=f"Proportion of {objects} with\n{metric} < {thresh}",
    xlabel="Number of pre-synapses",
)

# savefig(
#     f"prop_{objects}_below_threshold_metric={metric}_thresh={thresh}",
#     fig,
#     file_name,
#     doc_save=True,
# )

# %%

diffs["mtype"] = diffs.index.get_level_values("root_id").map(manifest["mtype"])
diffs["label"] = diffs["euclidean"] < 0.2

X = diffs[
    [
        "n_pre_synapses",
        "n_post_synapses",
        "n_nodes",
        "cumulative_n_operations",
        "path_length",
        "label",
    ]
].copy()
y = diffs["label"]

sns.PairGrid(X, hue="label").map_upper(sns.scatterplot, alpha=0.3, linewidth=0, s=1)

# %%

diffs["label"] = diffs["euclidean"] < 0.2

X = diffs[
    [
        "n_pre_synapses",
        "n_post_synapses",
        "n_nodes",
        "cumulative_n_operations",
        "path_length",
        "label",
    ]
].copy()

X["n_pre_synapses"] = np.log10(X["n_pre_synapses"])
X["n_post_synapses"] = np.log10(X["n_post_synapses"])
X["n_nodes"] = np.log10(X["n_nodes"].astype(float))
X["path_length"] = np.log10(X["path_length"].astype(float))

pg = sns.PairGrid(X, hue="label", corner=True)
pg.map_lower(
    sns.scatterplot,
    alpha=0.3,
    linewidth=0,
    s=1,
)
pg.map_diag(sns.histplot, stat="density")

fig = pg.figure
savefig("proofreading_feature_pairplot", fig, file_name, doc_save=True)

# %%

X = diffs[
    [
        "n_pre_synapses",
        "n_post_synapses",
        "n_nodes",
        "cumulative_n_operations",
        "path_length",
    ]
].copy()

X = X.query(
    "n_pre_synapses > 0 & n_post_synapses > 0 & n_nodes > 0 & cumulative_n_operations > 0 & path_length > 0"
)
X["n_pre_synapses"] = np.log10(X["n_pre_synapses"].astype(float))
X["n_post_synapses"] = np.log10(X["n_post_synapses"].astype(float))
X["n_nodes"] = np.log10(X["n_nodes"].astype(float))
X["path_length"] = np.log10(X["path_length"].astype(float))

y = diffs.loc[X.index, "euclidean"]
y = y < 0.2


lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

y_pred = lda.predict(X)

acc = (y == y_pred).mean()
print(acc)

# %%


def extract_features_and_labels(diffs, metric="euclidean", threshold=0.2):
    X = diffs[
        [
            "n_pre_synapses",
            "n_post_synapses",
            "n_nodes",
            "cumulative_n_operations",
            "path_length",
        ]
    ].copy()

    X.query(
        "n_pre_synapses > 0 & n_post_synapses > 0 & n_nodes > 0 & cumulative_n_operations > 0 & path_length > 0",
        inplace=True,
    )
    X["n_pre_synapses"] = np.log10(X["n_pre_synapses"].astype(float))
    X["n_post_synapses"] = np.log10(X["n_post_synapses"].astype(float))
    X["n_nodes"] = np.log10(X["n_nodes"].astype(float))
    X["path_length"] = np.log10(X["path_length"].astype(float))

    y = diffs.loc[X.index, metric]
    y = y < threshold
    return X, y


# %%

total_y = []
total_y_pred = []


def split_fit_predict(diffs, train_idx, test_idx):
    # separate classification by cell type
    total_y_test = []
    total_y_train = []
    total_y_pred = []
    for cell_type, cell_type_diffs in diffs.groupby("mtype"):
        sub_train_idx = train_idx.intersection(
            cell_type_diffs.index.get_level_values("root_id")
        )
        sub_test_idx = test_idx.intersection(
            cell_type_diffs.index.get_level_values("root_id")
        )
        X, y = extract_features_and_labels(cell_type_diffs)
        X_train = X.loc[sub_train_idx]
        y_train = y.loc[sub_train_idx]
        X_test = X.loc[sub_test_idx]
        y_test = y.loc[sub_test_idx]
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)

        total_y_train.append(y_train)
        total_y_test.append(y_test)
        total_y_pred.append(y_pred)

    total_y_train = np.concatenate(total_y_train)
    total_y_test = np.concatenate(total_y_test)
    total_y_pred = np.concatenate(total_y_pred)
    return total_y_train, total_y_test, total_y_pred


rows = []
splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=888)
roots = X.index.get_level_values("root_id").unique()
for i, (train_idx, test_idx) in enumerate(
    splitter.split(roots, manifest.loc[roots, "mtype"].values)
):
    train_idx = roots[train_idx]
    test_idx = roots[test_idx]

    # split classification
    y_train, y_test, y_pred = split_fit_predict(diffs, train_idx, test_idx)

    # scoring
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=True)
    rows.append({"split": i, "acc": acc, "f1": f1, "method": "split"})

    # pooled classification
    X, y = extract_features_and_labels(diffs)
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    # scoring
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=True)
    rows.append({"split": i, "acc": acc, "f1": f1, "method": "pooled"})

results_df = pd.DataFrame(rows)


# %%
fig, ax = plt.subplots(figsize=(6, 6))

sns.stripplot(data=results_df, x="method", y="f1", ax=ax)

ax.set_ylabel("F1 score (cross-validated)")

savefig("pooled-vs-split-f1", fig, file_name, doc_save=True)

# %%

fig, ax = plt.subplots(figsize=(6, 6))

sns.stripplot(data=results_df, x="method", y="acc", ax=ax)

ax.set_ylabel("Accuracy (cross-validated)")

savefig("pooled-vs-split-acc", fig, file_name, doc_save=True)

# %%

wilcoxon(
    results_df.query("method == 'split'").set_index("split")["f1"],
    results_df.query("method == 'pooled'").set_index("split")["f1"],
)
# %%
np.mean(
    results_df.query("method == 'split'").set_index("split")["f1"]
    - results_df.query("method == 'pooled'").set_index("split")["f1"]
)

# %%


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


feature_sets = list(powerset(X.columns))[1:]  # drop null set
feature_sets = [list(f) for f in feature_sets]
all_features = X.columns.tolist()

X, y = extract_features_and_labels(diffs)
rows = []
drop_cols = None  #  ["cumulative_n_operations"]

splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=888)
roots = X.index.get_level_values("root_id").unique()
for i, (train_idx, test_idx) in enumerate(
    splitter.split(roots, manifest.loc[roots, "mtype"].values)
):
    train_idx = roots[train_idx]
    test_idx = roots[test_idx]

    for feature_set in feature_sets:
        this_X = X[feature_set]

        X_train = this_X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = this_X.loc[test_idx]
        y_test = y.loc[test_idx]

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=True)

        row = {"split": i, "acc": acc, "f1": f1}
        for feature in all_features:
            row[feature] = feature in feature_set
        rows.append(row)

results_df = pd.DataFrame(rows)

# %%


class UpsetCatplot:
    def __init__(self, fig, ax, upset_ax):
        self.fig = fig
        self.ax = ax
        self.upset_ax = upset_ax

    def set_ylabel(self, label, **kwargs):
        self.ax.set_ylabel(label, **kwargs)

    # TODO: allow for passing a dictionary to map old to new in case order changed
    def set_upset_ticklabels(self, ticklabels, **kwargs):
        self.upset_ax.set_yticklabels(ticklabels, **kwargs)

    def set_title(self, title, **kwargs):
        self.ax.set_title(title, **kwargs)


def upset_catplot(
    data,
    x=None,
    y=None,
    hue=None,
    kind="bar",
    estimator=None,
    estimator_width=0.2,
    estimator_labels=False,
    estimator_format="{estimate:.2f}",
    upset_ratio=0.3,
    upset_pad=0.7,
    upset_size=None,
    upset_linewidth=3,
    ax=None,
    fig=None,
    figsize=(8, 6),
    s=100,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = plt.figure()

    data = data.copy()
    # data = data.groupby(x).mean().reset_index()
    groupby = data.groupby(x, sort=False)
    group_data = groupby.mean().reset_index()
    # combos = groupby.groups.keys()
    # combos = pd.DataFrame(combos, columns=x).set_index(x)

    # create a dummy variable for seaborn-style plotting
    data["all_x_vars"] = np.nan
    for combo, index in groupby.groups.items():
        data.loc[index, "all_x_vars"] = str(combo)

    if kind == "bar":
        sns.barplot(data=data, x="all_x_vars", y=y, hue=hue, ax=ax, **kwargs)
    elif kind == "strip":
        sns.stripplot(data=data, x="all_x_vars", y=y, hue=hue, ax=ax, **kwargs)

    # TODO : could add other seaborn "kind"s

    # TODO : possibly more control over how this gets plotted
    # E.g. right now this would look ugly with barplot
    if estimator is not None:
        # for i, estimate in enumerate(group_estimates):
        for i, row in group_data.iterrows():
            estimate = row[y]

            x_low = i - estimator_width
            x_high = i + estimator_width
            ax.plot([x_low, x_high], [estimate, estimate], color="black", zorder=9)
            if estimator_labels:
                pad = 0.02
                ax.text(
                    x_high + pad,
                    estimate,
                    estimator_format.format(estimate=estimate),
                    va="center",
                    zorder=10,
                )

    ax.set(xlabel="")

    divider = make_axes_locatable(ax)
    upset_ax = divider.append_axes(
        "bottom", size=f"{upset_ratio*100}%", sharex=ax, pad=0
    )

    combos = group_data.set_index(x)

    plot_upset_indicators(
        combos,
        ax=upset_ax,
        height_pad=upset_pad,
        element_size=upset_size,
        linewidth=upset_linewidth,
        s=s,
    )

    return UpsetCatplot(fig, ax, upset_ax)


def plot_upset_indicators(
    intersections,
    ax=None,
    facecolor="black",
    element_size=None,
    with_lines=True,
    horizontal=True,
    height_pad=0.7,
    linewidth=2,
    s=100,
):
    # REF: https://github.com/jnothman/UpSetPlot/blob/e6f66883e980332452041cd1a6ba986d6d8d2ae5/upsetplot/plotting.py#L428
    """Plot the matrix of intersection indicators onto ax"""
    data = intersections
    index = data.index
    index = index.reorder_levels(index.names[::-1])
    n_cats = index.nlevels

    idx = np.flatnonzero(index.to_frame()[index.names].values)  # [::-1]
    c = np.array(["lightgrey"] * len(data) * n_cats, dtype="O")
    c[idx] = facecolor
    x = np.repeat(np.arange(len(data)), n_cats)
    y = np.tile(np.arange(n_cats), len(data))
    if s is None:
        if element_size is not None:
            s = (element_size * 0.35) ** 2
        else:
            # TODO: make s relative to colw
            s = 200
    ax.scatter(x, y, c=c.tolist(), linewidth=0, s=s)

    if with_lines:
        line_data = (
            pd.Series(y[idx], index=x[idx]).groupby(level=0).aggregate(["min", "max"])
        )
        ax.vlines(
            line_data.index.values,
            line_data["min"],
            line_data["max"],
            lw=linewidth,
            colors=facecolor,
        )

    tick_axis = ax.yaxis
    tick_axis.set_ticks(np.arange(n_cats))
    tick_axis.set_ticklabels(index.names, rotation=0 if horizontal else -90)
    # ax.xaxis.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    if not horizontal:
        ax.yaxis.set_ticks_position("top")
    ax.set_frame_on(False)
    ax.set_ylim((-height_pad, n_cats - 1 + height_pad))
    ax.set_xticks([])


value = "f1"
results_df[f"mean_{value}"] = results_df.groupby(all_features).transform("mean")[value]
results_df = results_df.sort_values([f"mean_{value}", "split"], ascending=False)
upsetplot = upset_catplot(
    results_df,
    x=all_features,
    y=value,
    kind="strip",
    estimator=np.mean,
    estimator_width=0.3,
    s=80,
    alpha=0.3,
)
upsetplot.ax.set_ylabel("F1 score")

savefig("feature_set_f1_scores", upsetplot.fig, file_name, doc_save=True)

# %%
final_feature_set = ["n_pre_synapses", "n_post_synapses", "n_nodes"]

X, y = extract_features_and_labels(diffs)
X = X[final_feature_set]

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
y_pred = lda.predict(X)
tn, fp, fn, tp = confusion_matrix(y, y_pred, normalize="all").ravel()


X_trans = lda.transform(X)

fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(x=X_trans.ravel(), hue=y, ax=ax, stat="density")
sns.move_legend(ax, loc="upper left", title="Euc. distance < 0.2")
savefig("lda_transformed", fig, file_name, doc_save=True)


# %%

y_scores = lda.decision_function(X)

fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(y_scores, ax=ax)


qt = QuantileTransformer(n_quantiles=100, output_distribution="uniform")
qt.fit(y_scores.reshape(-1, 1))

# %%

fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(x=y_scores.ravel(), hue=y, ax=ax, stat="density")
sns.move_legend(ax, loc="upper left", title="Euc. distance < 0.2")
ax.set(xlabel="Log posterior ratio")
ax.axvline(0, color="black", lw=1, ls="--")
savefig("lda_decision_function", fig, file_name, doc_save=True)

# %%

y_posteriors = lda.predict_proba(X)[:, 1]

fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(y_posteriors, ax=ax)

# %%
precisions, recalls, thresholds = precision_recall_curve(
    y, y_scores, drop_intermediate=True
)

i = 200
precisions = precisions[:-i]
recalls = recalls[:-i]
thresholds = thresholds[:-i]
thresholds = np.concatenate([thresholds, [thresholds.max() + 1]])

threshold_df = pd.DataFrame(
    {"threshold": thresholds, "precision": precisions, "recall": recalls}
)

fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(
    data=threshold_df,
    x="recall",
    y="precision",
    hue="threshold",
    palette="coolwarm",
    ax=ax,
    linewidth=0,
)
ax.set(xlabel="Recall", ylabel="Precision")
sns.move_legend(ax, loc="lower left", title="Log-odds\nthreshold")

savefig("precision_recall_curve", fig, file_name, doc_save=True)

# %%


# def custom_score(y_observed, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_observed, y_pred, normalize="all").ravel()
#     return tp - 1 * fp


# from sklearn.metrics import make_scorer

# custom_scorer = make_scorer(
#     custom_score, response_method="predict", greater_is_better=True
# )
# tuned_classifier = TunedThresholdClassifierCV(lda, cv=5, scoring=custom_scorer).fit(
#     X, y
# )

# print(f"Tuned decision threshold: {tuned_classifier.best_threshold_:.3f}")
# print(f"Custom score: {custom_score(y_test, tuned_classifier.predict(X_test)):.2f}")


# %%


client = start_client()

mtypes = load_mtypes(client)
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

# %%
pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(mtypes["cell_type"])

# %%

pre_synapses = pre_synapses.query("pre_pt_root_id != post_pt_root_id")
post_synapses = post_synapses.query("pre_pt_root_id != post_pt_root_id")

# %%

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
        n_leaves = Parallel(n_jobs=12)(
            delayed(get_n_leaves_for_root)(root_id) for root_id in all_inhib_roots
        )
        n_leaves = pd.Series(n_leaves, index=all_inhib_roots)
        n_leaves.to_csv(out_path / "n_leaves.csv")

# %%
inhib_features = pd.concat([n_pre_synapses, n_post_synapses, n_leaves], axis=1)
inhib_features.columns = ["n_pre_synapses", "n_post_synapses", "n_nodes"]
X_new = inhib_features.copy()
X_new = X_new.query("n_pre_synapses > 0 & n_post_synapses > 0 & n_nodes > 0")
X_new["n_pre_synapses"] = np.log10(X_new["n_pre_synapses"].astype(float))
X_new["n_post_synapses"] = np.log10(X_new["n_post_synapses"].astype(float))
X_new["n_nodes"] = np.log10(X_new["n_nodes"].astype(float))

# %%
y_scores_new = lda.decision_function(X_new)
y_scores_new = pd.Series(y_scores_new, index=X_new.index)

y_scores_quant_new = qt.transform(y_scores_new.values.reshape(-1, 1))
y_scores_quant_new = pd.Series(y_scores_quant_new.flatten(), index=X_new.index)

fig, ax = plt.subplots(figsize=(6, 6))

sns.histplot(x=y_scores_new, ax=ax)

ax.set_xlabel("Log posterior ratio")

savefig("new_log_posterior_ratio", fig, file_name, doc_save=True)

# %%
X_trans_new = lda.transform(X_new)

remapped_y = np.vectorize({True: "old", False: "old"}.get)(y)
labels = np.array(remapped_y.tolist() + ["new"] * len(X_new))

X_trans_combined = np.concatenate((X_trans, X_trans_new), axis=0).ravel()

fig, ax = plt.subplots(figsize=(6, 6))

sns.histplot(x=X_trans_new.ravel(), ax=ax)

# %%


seg_df = inhib_features.copy()
seg_df["log_odds"] = np.nan
seg_df.loc[y_scores_new.index, "log_odds"] = y_scores_new
seg_df = seg_df.dropna()
seg_df["log_odds_quant"] = y_scores_quant_new


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

seg_df = seg_df.join(proofreading_df[["strategy_dendrite", "strategy_axon"]])

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
labeled_inhib_features["log_odds_quant"] = y_scores_quant_new

# %%
pre_synapses["pre_log_odds_quant"] = pre_synapses["pre_pt_root_id"].map(
    labeled_inhib_features["log_odds_quant"]
)

# TODO add casey's cells back in here, including those which were misclassified in the
# meta model thingy
# include them in the clustering regardless of the posterior
# see what cluster centroids look like relative to these cells


# %%

threshold = 0.2
pre_synapses_by_threshold = pre_synapses.query(f"pre_log_odds_quant > {threshold}")


projection_counts = pre_synapses_by_threshold.groupby("pre_pt_root_id")[
    "post_mtype"
].value_counts()

remove_inhib = True
if remove_inhib:
    projection_counts = projection_counts.drop(
        labels=["DTC", "ITC", "PTC", "STC"], level="post_mtype"
    )

# turn counts into proportions
projection_props = projection_counts / projection_counts.groupby("pre_pt_root_id").sum()

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

sns.clustermap(props_by_mtype.T, xticklabels=False, row_cluster=False, figsize=(20, 10))

# %%

method = "ward"
metric = "euclidean"
k = 40
linkage_matrix = linkage(props_by_mtype, method=method, metric=metric)

leaves = leaves_list(linkage_matrix)

props_by_mtype = props_by_mtype.iloc[leaves]

linkage_matrix = linkage(props_by_mtype, method=method, metric=metric)
labels = pd.Series(
    fcluster(linkage_matrix, k, criterion="maxclust"), index=props_by_mtype.index
)

# %%
weighting = props_by_mtype.copy()
weighting.columns = np.arange(len(weighting.columns))
weighting["label"] = labels

means = weighting.groupby("label").mean()
label_ranking = means.mul(means.columns).sum(axis=1).sort_values()

new_label_map = dict(zip(labels, label_ranking.index.get_indexer_for(labels)))

# %%
manifest = load_manifest()
ctypes = manifest["ctype"]


# %%

# linkage_matrix = linkage_matrix[leaves]
set_context(font_scale=2)

colors = sns.color_palette("tab20", n_colors=k)
palette = dict(zip(np.arange(1, k + 1), colors))
color_labels = labels.map(palette).rename("Cluster")

color_labels = color_labels.to_frame()
color_labels["Posterior quantile"] = labeled_inhib_features.loc[
    color_labels.index, "log_odds_quant"
]

norm = Normalize(vmin=0.7, vmax=1)

# colormap that is increasing reds from 0 to 1
colors = [(1, 1, 1), (0, 0, 0)]
cmap = LinearSegmentedColormap.from_list("custom", colors)

color_labels["Posterior quantile"] = color_labels["Posterior quantile"].map(
    lambda x: cmap(norm(x))
)

color_labels["Motif group"] = color_labels.index.to_series().map(ctypes).map(palette)


cgrid = sns.clustermap(
    props_by_mtype.T,
    cmap="Reds",
    figsize=(30, 10),
    row_cluster=False,
    col_colors=color_labels,
    col_linkage=linkage_matrix,
    xticklabels=False,
)
ax = cgrid.ax_heatmap

# move the y-axis labels to the left
ax.yaxis.tick_left()
ax.set(ylabel="Excitatory Neuron Class", xlabel="Inhibitory Neuron")
ax.yaxis.set_label_position("left")

props_by_mtype["label"] = labels.map(new_label_map)
shifts = props_by_mtype["label"] != props_by_mtype["label"].shift()
shifts.iloc[0] = False
shifts = shifts[shifts].index
shift_ilocs = props_by_mtype.index.get_indexer_for(shifts)
for shift in shift_ilocs:
    ax.axvline(shift, color="black", lw=0.5)
ax.set(xlabel="Inhibitory Neuron", ylabel="Excitatory Neuron Class")

props_by_mtype.drop("label", axis=1, inplace=True)

y_borders = [3, 5, 8, 12, 14]
for y_border in y_borders:
    ax.axhline(y_border, color="black", lw=0.5)

plt.savefig("excitatory_inhibitory_clustermap_w_tree.png", bbox_inches="tight")


# %%

casey_palette = load_casey_palette()

# %%
mtypes = client.materialize.query_table(MTYPES_TABLE).set_index("pt_root_id")

sort_bys = ["label", "log_odds_quant"]
props_by_mtype["label"] = labels.map(new_label_map)
props_by_mtype["log_odds_quant"] = labeled_inhib_features.loc[
    props_by_mtype.index, "log_odds_quant"
]
props_by_mtype = props_by_mtype.sort_values(sort_bys)
props_by_mtype.drop(sort_bys, axis=1, inplace=True)
# fig, ax = plt.subplots(figsize=(25, 10))
mg = MatrixGrid(figsize=(25, 10))
ax = mg.ax
sns.heatmap(props_by_mtype.T, cmap="Reds", xticklabels=False, ax=ax)


props_by_mtype["label"] = labels.map(new_label_map)
shifts = props_by_mtype["label"] != props_by_mtype["label"].shift()
shifts.iloc[0] = False
shifts = shifts[shifts].index
shift_ilocs = props_by_mtype.index.get_indexer_for(shifts)
for shift in shift_ilocs:
    ax.axvline(shift, color="black", lw=1)
ax.set(xlabel="Inhibitory Neuron", ylabel="Excitatory Neuron Class")

# props_by_mtype.drop("label", axis=1, inplace=True)


y_borders = [3, 5, 8, 12, 14]
for y_border in y_borders:
    ax.axhline(y_border, color="black", lw=1)

# add
posterior_ax = mg.append_axes("top", size="5%", pad=0.05)
sns.heatmap(
    labeled_inhib_features.loc[props_by_mtype.index, "log_odds_quant"].values.reshape(
        1, -1
    ),
    ax=posterior_ax,
    # cmap=cmap,
    cmap="Greys",
    cbar=False,
    vmin=threshold,
    vmax=1,
)
posterior_ax.set(xticks=[], yticks=[])
posterior_ax.set_ylabel("Posterior quantile", rotation=0, ha="right", va="center")


mtype_ax = mg.append_axes("top", size="5%", pad=0.05)

# add the mtype labels
palette = casey_palette.copy()
plot_frame = mtypes.loc[props_by_mtype.index]["cell_type"].to_frame().copy()
unique_values = np.unique(plot_frame.values)
value_map = dict(zip(unique_values, np.arange(len(unique_values))))
plot_frame = plot_frame.replace(value_map)
colors = [palette[i] for i in unique_values]
cmap = ListedColormap(colors)
sns.heatmap(
    plot_frame.values.reshape(1, -1),
    ax=mtype_ax,
    cmap=cmap,
    cbar=False,
    yticklabels=False,
    xticklabels=False,
)
mtype_ax.set_ylabel("Mtype", rotation=0, ha="right", va="center")
mtype_ax.set_xlabel("")

# final touches
label_pos = labels.rename("label").to_frame().copy().loc[props_by_mtype.index]
label_pos["pos"] = np.arange(len(label_pos))
positions_by_label = label_pos.groupby("label")["pos"].mean()

ax.set_xticks(positions_by_label.values)
ax.set_xticklabels(positions_by_label.index)

mg.clear_subaxis_ticks()

# %%
from nglui.segmentprops import SegmentProperties

seg_df = inhib_features.copy()
seg_df["log_odds"] = np.nan
seg_df.loc[y_scores_new.index, "log_odds"] = y_scores_new
seg_df = seg_df.dropna()
seg_df["log_odds_quant"] = y_scores_quant_new
seg_df["label"] = labels.fillna(-1).astype(int)

n_randoms = 5
for i in range(n_randoms):
    seg_df[f"random_{i}"] = np.random.uniform(0, 1, size=len(seg_df))

seg_prop = SegmentProperties.from_dataframe(
    seg_df.reset_index(),
    id_col="pt_root_id",
    label_col="label",
    tag_value_cols=["label"],
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

from nglui import statebuilder

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
