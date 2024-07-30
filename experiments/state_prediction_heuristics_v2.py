# %%

import os
import pickle
from itertools import chain, combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nglui import statebuilder
from nglui.segmentprops import SegmentProperties
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from pkg.constants import OUT_PATH
from pkg.plot import savefig, set_context
from pkg.utils import load_manifest, load_mtypes, start_client

# %%

file_name = "state_prediction_heuristics"

set_context()

client = start_client()
mtypes = load_mtypes(client)


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

n_filtered = (
    all_infos.loc[pd.IndexSlice[:, "historical"], :]["is_filtered"]
    .fillna(False)
    .groupby("root_id")
    .cumsum()
)

all_infos["n_filtered_edits"] = n_filtered

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
    diffs["n_filtered_edits"] = group_info["n_filtered_edits"]

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
            "n_filtered_edits",
            "is_filtered",
        ]
    ]
)

diffs["is_filtered"] = diffs["is_filtered"].fillna(True)
diffs = diffs.query("is_filtered")

# %%


thresh = 0.2
metric = "euclidean"

diffs["mtype"] = diffs.index.get_level_values("root_id").map(manifest["mtype"])
diffs["label"] = diffs[metric] < thresh

X = diffs[
    [
        "n_pre_synapses",
        "n_post_synapses",
        "n_nodes",
        "path_length",
        "label",
        "n_filtered_edits",
    ]
].copy()

X["n_pre_synapses"] = X["n_pre_synapses"].replace(0, 1)
X["n_post_synapses"] = X["n_post_synapses"].replace(0, 1)
X["n_pre_synapses"] = np.log10(X["n_pre_synapses"])
X["n_post_synapses"] = np.log10(X["n_post_synapses"])
X["n_nodes"] = np.log10(X["n_nodes"].astype(float))
X["path_length"] = np.log10(X["path_length"].astype(float))
X["n_filtered_edits"] = np.log10((X["n_filtered_edits"] + 1).astype(float))

X = X.fillna(0)

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
fig, ax = plt.subplots(figsize=(6, 6))

sns.scatterplot(
    data=X,
    x="n_pre_synapses",
    y="n_post_synapses",
    hue="label",
    ax=ax,
    alpha=0.7,
    linewidth=0,
    s=2,
)

# %%

y = X["label"]
X = X.drop(columns="label")

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

y_pred = lda.predict(X)
# %%


def compute_scores(X, y, y_pred):
    counts = X.groupby("root_id").size()
    inv_counts = 1 / counts
    sample_weight = X.index.get_level_values("root_id").map(inv_counts)

    precision = precision_score(y, y_pred, sample_weight=sample_weight, pos_label=True)
    recall = recall_score(y, y_pred, sample_weight=sample_weight, pos_label=True)
    f1 = f1_score(y, y_pred, sample_weight=sample_weight, pos_label=True)
    coehns = cohen_kappa_score(y, y_pred, sample_weight=sample_weight)
    matthews = matthews_corrcoef(y, y_pred, sample_weight=sample_weight)

    scores = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coehns": coehns,
        "matthews": matthews,
    }
    return scores


print(compute_scores(X, y, y_pred))


# %%


def extract_features_and_labels(diffs, metric="euclidean", threshold=0.2):
    X = diffs[
        [
            "n_pre_synapses",
            "n_post_synapses",
            "n_nodes",
            "path_length",
            "label",
            "n_filtered_edits",
        ]
    ].copy()

    X["n_pre_synapses"] = X["n_pre_synapses"].replace(0, 1)
    X["n_post_synapses"] = X["n_post_synapses"].replace(0, 1)
    X["n_pre_synapses"] = np.log10(X["n_pre_synapses"])
    X["n_post_synapses"] = np.log10(X["n_post_synapses"])
    X["n_nodes"] = np.log10(X["n_nodes"].astype(float))
    X["path_length"] = np.log10(X["path_length"].astype(float))
    X["n_filtered_edits"] = np.log10((X["n_filtered_edits"] + 1).astype(float))

    X = X.fillna(0)

    y = diffs.loc[X.index, metric]
    y = y < threshold
    return X, y


# %%


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


all_features = [
    "n_pre_synapses",
    "n_post_synapses",
    "n_nodes",
    "path_length",
    "n_filtered_edits",
]

feature_sets = list(powerset(all_features))[1:]  # drop null set
feature_sets = [list(f) for f in feature_sets]

X, y = extract_features_and_labels(diffs)
rows = []
drop_cols = None  #  ["cumulative_n_operations"]


quantile_lda = Pipeline(
    [
        ("quantile", QuantileTransformer(output_distribution="normal")),
        ("lda", LinearDiscriminantAnalysis()),
    ]
)

models = {
    "lda": LinearDiscriminantAnalysis(),
    "quantile-lda": quantile_lda,
    "rf": RandomForestClassifier(max_depth=5),
    "lr": LogisticRegression(),
}

splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=888)
roots = X.index.get_level_values("root_id").unique()
for i, (train_idx, test_idx) in tqdm(
    enumerate(splitter.split(roots, manifest.loc[roots, "mtype"].values)),
    total=splitter.get_n_splits(),
):
    train_idx = roots[train_idx]
    test_idx = roots[test_idx]

    for feature_set in feature_sets:
        this_X = X[feature_set]

        X_train = this_X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = this_X.loc[test_idx]
        y_test = y.loc[test_idx]

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            row = compute_scores(X_test, y_test, y_pred)
            row["split"] = i
            row["model_name"] = model_name

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


# value = "coehns"
# results_df[f"mean_{value}"] = results_df.groupby(all_features).transform("mean")[value]
# results_df = results_df.sort_values([f"mean_{value}", "split"], ascending=False)
# upsetplot = upset_catplot(
#     results_df,
#     x=all_features,
#     y=value,
#     hue="model_name",
#     kind="strip",
#     # estimator=np.mean,
#     # estimator_width=0.3,
#     s=80,
#     alpha=0.3,
# )
# upsetplot.ax.set_ylabel("F1 score")

# savefig("feature_set_f1_scores", upsetplot.fig, file_name, doc_save=True)

# %%
results_df.groupby(all_features + ["model_name"])["f1"].mean().sort_values(
    ascending=False
).head(20)

# %%
final_feature_set = ["n_pre_synapses", "n_post_synapses", "n_nodes"]

X, y = extract_features_and_labels(diffs)
X = X[final_feature_set]

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
y_pred = lda.predict(X)

# %%

y_scores = lda.decision_function(X)
y_scores = pd.Series(y_scores, index=X.index)

fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(y_scores, ax=ax)

qt = QuantileTransformer(n_quantiles=100, output_distribution="uniform")
qt.fit(y_scores.values.reshape(-1, 1))

# %%

fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(x=y_scores.values.ravel(), hue=y, ax=ax, stat="density")
sns.move_legend(ax, loc="upper left", title="Euc. distance < 0.2")
ax.set(xlabel="Log posterior ratio")
ax.axvline(0, color="black", lw=1, ls="--")
savefig("lda_decision_function", fig, file_name, doc_save=True)

# %%
counts = X.groupby("root_id").size()
inv_counts = 1 / counts
sample_weight = X.index.get_level_values("root_id").map(inv_counts)

precisions, recalls, thresholds = precision_recall_curve(
    y, y_scores, drop_intermediate=True, sample_weight=sample_weight
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


fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(x=y_scores.values, hue=y.values, ax=ax)
ax.set(ylabel="Count (neuron-states)")
right_ax = plt.twinx(ax)

colors = sns.color_palette("tab10", n_colors=6)
sns.lineplot(
    x=threshold_df["threshold"],
    y=threshold_df["precision"],
    ax=right_ax,
    label="Precision",
    color=colors[3],
)
sns.lineplot(
    x=threshold_df["threshold"],
    y=threshold_df["recall"],
    ax=right_ax,
    label="Recall",
    color=colors[4],
)

threshold_df["f1"] = (
    2
    * threshold_df["precision"]
    * threshold_df["recall"]
    / (threshold_df["precision"] + threshold_df["recall"])
)
sns.lineplot(
    x=threshold_df["threshold"],
    y=threshold_df["f1"],
    ax=right_ax,
    label="F1",
    color=colors[5],
)

right_ax.set_ylabel("Precision/Recall/F1")
right_ax.set_ylim(0, 1)

sns.move_legend(ax, "center left", title="Euc. dist\n< 0.2")
sns.move_legend(right_ax, "upper right")

ax.spines[["right"]].set_visible(True)

savefig("precision_recall_overlay", fig, folder=file_name)

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
pre_synapses.query("pre_pt_root_id == 864691135697284250")

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
y_scores_new = lda.decision_function(X_new)
y_scores_new = pd.Series(y_scores_new, index=X_new.index)

y_scores_quant_new = qt.transform(y_scores_new.values.reshape(-1, 1))
y_scores_quant_new = pd.Series(y_scores_quant_new.flatten(), index=X_new.index)

fig, ax = plt.subplots(figsize=(6, 6))

sns.histplot(x=y_scores_new, ax=ax)

ax.set_xlabel("Log posterior ratio")

savefig("new_log_posterior_ratio", fig, file_name, doc_save=True)

# %%
inhib_features["log_posterior_ratio"] = y_scores_new
inhib_features["mtype"] = inhib_features.index.map(mtypes["cell_type"])

# %%
from pkg.utils import load_casey_palette

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

    right_ax = plt.twinx(ax)
    sns.lineplot(
        data=threshold_df,
        x="threshold",
        y="recall",
        ax=right_ax,
        color="red",
        label="Recall",
    )
    sns.lineplot(
        data=threshold_df,
        x="threshold",
        y="precision",
        ax=right_ax,
        color="purple",
        label="Precision",
    )
    sns.lineplot(
        data=threshold_df, x="threshold", y="f1", ax=right_ax, color="green", label="F1"
    )
    sns.move_legend(right_ax, loc="upper right")
    right_ax.set_ylabel("Estimated Precision/Recall/F1")

    ax.spines["right"].set_visible(True)


savefig("log_posterior_ratio_survival_precision_recall", fig, file_name, doc_save=True)


# %%

bins = np.histogram_bin_edges(y_scores_quant_new, bins="auto")
y_scores_quant_new_binned = pd.cut(y_scores_quant_new, bins=bins, include_lowest=True)
bin_counts = y_scores_quant_new_binned.value_counts()
cumsum = bin_counts.sort_index().cumsum()
survival = len(y_scores_quant_new) - cumsum

with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(x=survival.index.categories.mid, y=survival.values, ax=ax)
    ax.set_xlabel("Posterior ratio quantile")

savefig("posterior_ratio_quantile_survival", fig, file_name, doc_save=True)

# %%


seg_df = inhib_features.copy()
seg_df["log_odds"] = np.nan
seg_df.loc[y_scores_new.index, "log_odds"] = y_scores_new
seg_df = seg_df.dropna()
seg_df["log_odds_quant"] = y_scores_quant_new
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
labeled_inhib_features["log_odds_quant"] = y_scores_quant_new

# %%


# TODO add casey's cells back in here, including those which were misclassified in the
# meta model thingy
# include them in the clustering regardless of the posterior
# see what cluster centroids look like relative to these cells
# %%


from pkg.utils import load_casey_palette, load_mtypes

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
    from scipy.cluster.hierarchy import dendrogram

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

    import colorcet as cc

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

    from matplotlib.patches import Rectangle

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
from nglui.segmentprops import SegmentProperties

seg_df = inhib_features.copy()
seg_df["log_odds"] = np.nan
seg_df.loc[y_scores_new.index, "log_odds"] = y_scores_new
seg_df = seg_df.dropna()
seg_df["log_odds_quant"] = y_scores_quant_new

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

from nglui import statebuilder

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
