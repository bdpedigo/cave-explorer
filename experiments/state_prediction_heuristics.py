# %%

import pickle

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist

from pkg.constants import OUT_PATH
from pkg.plot import set_context
from pkg.utils import load_manifest, load_mtypes

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

diffs["synapse_count_bin"] = pd.cut(diffs["n_pre_synapses"], synapse_bins, right=False)

# %%
fig, ax = plt.subplots(figsize=(6, 6))

sns.histplot(
    data=diffs.query("n_pre_synapses < 1000"),
    x="n_pre_synapses",
    y="euclidean",
    ax=ax,
    stat="density",
    bins=100,
)

# %%
fig, ax = plt.subplots(figsize=(6, 6))

sns.histplot(
    data=diffs.query("n_pre_synapses > 1000"),
    x="n_pre_synapses",
    y="euclidean",
    ax=ax,
    stat="density",
    bins=100,
)
# %%

fig, ax = plt.subplots(figsize=(6, 6))

sns.histplot(
    data=diffs.query("n_pre_synapses > 1000"),
    x="euclidean",
    ax=ax,
    stat="density",
    bins=100,
)
# %%

sample = diffs.query("n_pre_synapses >= 1000")
(sample["euclidean"] < 0.2).mean()


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

# savefig(
#     f"prop_{objects}_below_threshold_metric={metric}_thresh={thresh}",
#     fig,
#     file_name,
#     doc_save=True,
# )

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

min_inds = (
    diffs.query("n_pre_synapses > 1200 & mtype == 'ITC'")
    .groupby("root_id")["cumulative_n_operations"]
    .idxmin()
)

diffs.loc[min_inds]

# %%
pop_good = diffs.query("mtype == 'ITC' & euclidean < 0.2")
pop_bad = diffs.query("mtype == 'ITC' & euclidean >= 0.2")

fig, ax = plt.subplots(figsize=(6, 6))

sns.histplot(
    data=pop_good, x="n_pre_synapses", ax=ax, stat="density", bins=100, color="green"
)
sns.histplot(
    data=pop_bad, x="n_pre_synapses", ax=ax, stat="density", bins=100, color="red"
)

# %%

fig, ax = plt.subplots(figsize=(6, 6))

sns.histplot(data=pop_good, x="n_nodes", ax=ax, stat="density", bins=100, color="green")
sns.histplot(data=pop_bad, x="n_nodes", ax=ax, stat="density", bins=100, color="red")

# %%
fig, ax = plt.subplots(figsize=(6, 6))
bins = 40
sns.histplot(
    data=pop_good, x="n_post_synapses", ax=ax, stat="density", color="green", bins=bins
)
sns.histplot(
    data=pop_bad, x="n_post_synapses", ax=ax, stat="density", color="red", bins=bins
)

# %%
fig, ax = plt.subplots(figsize=(6, 6))

feature = "cumulative_n_operations"

sns.histplot(data=pop_good, x=feature, ax=ax, stat="density", color="green", bins=100)
sns.histplot(data=pop_bad, x=feature, ax=ax, stat="density", color="red", bins=100)

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
y = diffs["label"]

sns.PairGrid(X, hue="label").map_upper(sns.scatterplot, alpha=0.3, linewidth=0, s=1)

# %%

diffs["label"] = diffs["euclidean"] < 0.1

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

sns.PairGrid(X, hue="label").map_upper(sns.scatterplot, alpha=0.3, linewidth=0, s=1)

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

y_pred = lda.predict(X)

acc = (y == y_pred).mean()
print(acc)

X_trans = lda.transform(X)

fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(x=X_trans.ravel(), hue=y, ax=ax, stat="density")
sns.move_legend(ax, loc="upper left", title="Euc. distance < 0.2")

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
from sklearn.metrics import classification_report

total_y = []
total_y_pred = []
# by cell type
for cell_type, cell_type_diffs in diffs.groupby("mtype"):
    X, y = extract_features_and_labels(cell_type_diffs)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    y_pred = lda.predict(X)
    acc = (y == y_pred).mean()
    print(f"{cell_type}: {acc}")
    total_y.append(y)
    total_y_pred.append(y_pred)

total_y = np.concatenate(total_y)
total_y_pred = np.concatenate(total_y_pred)
acc = (total_y == total_y_pred).mean()
print(f"By cell type: {acc}")
print(classification_report(total_y, total_y_pred))

print()
# overall
X, y = extract_features_and_labels(diffs)
X = X.drop(columns="cumulative_n_operations")
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
y_pred = lda.predict(X)
acc = (y == y_pred).mean()
print(f"Overall: {acc}")

print(classification_report(y, y_pred))

# %%
# power set of possible features
from itertools import chain, combinations

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


feature_sets = list(powerset(X.columns))[1:]  # drop null set
feature_sets = [list(f) for f in feature_sets]
all_features = X.columns.tolist()

X, y = extract_features_and_labels(diffs)
rows = []
drop_cols = None  #  ["cumulative_n_operations"]
splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

# upset_ax = divider.append_axes("bottom", size=f"{upset_ratio*100}%", sharex=ax)


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
    figsize=(8, 6),
    s=100,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

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
    print(index)
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
    ax=ax,
    kind="strip",
    estimator=np.mean,
    estimator_width=0.3,
    s=80,
    alpha=0.3,
)
upsetplot.ax.set_ylabel("F1 score")
# yticklabels = upsetplot.upset_ax.get_yticklabels()
# upsetplot.upset_ax.set_yticklabels(yticklabels)
from pkg.plot import savefig

savefig("feature_set_f1_scores", upsetplot.fig, file_name, doc_save=True)

# %%
final_feature_set = ["n_pre_synapses", "n_post_synapses", "n_nodes"]

X, y = extract_features_and_labels(diffs)
X = X[final_feature_set]

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
y_pred = lda.predict(X)


from sklearn.metrics import PrecisionRecallDisplay

y_score = lda.decision_function(X)

display = PrecisionRecallDisplay.from_predictions(
    y, y_score, name="LDA", plot_chance_level=False
)

# %%
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import FixedThresholdClassifier, TunedThresholdClassifierCV

fixed_classifier = FixedThresholdClassifier(lda, threshold=0.1)
fixed_classifier.fit(X, y)
y_pred = fixed_classifier.predict(X)
tn, fp, fn, tp = confusion_matrix(y, y_pred, normalize="all").ravel()

y_scores = lda.decision_function(X)
precisions, recalls, thresholds = precision_recall_curve(
    y, y_scores, drop_intermediate=True
)

i = 200
precisions = precisions[:-i]
recalls = recalls[:-i]
thresholds = thresholds[:-i]
thresholds = np.concatenate([thresholds, [thresholds.max() + 1]])

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(recalls, precisions)
ax.set(xlabel="Recall", ylabel="Precision")

threshold_df = pd.DataFrame(
    {"threshold": thresholds, "precision": precisions, "recall": recalls}
)

fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(
    data=threshold_df, x="recall", y="precision", hue="threshold", ax=ax, linewidth=0
)

# %%


def custom_score(y_observed, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_observed, y_pred, normalize="all").ravel()
    return tp - 1 * fp


from sklearn.metrics import make_scorer

custom_scorer = make_scorer(
    custom_score, response_method="predict", greater_is_better=True
)
tuned_classifier = TunedThresholdClassifierCV(lda, cv=5, scoring=custom_scorer).fit(
    X, y
)

print(f"Tuned decision threshold: {tuned_classifier.best_threshold_:.3f}")
print(f"Custom score: {custom_score(y_test, tuned_classifier.predict(X_test)):.2f}")

# %%

from pkg.constants import MTYPES_TABLE
from pkg.utils import start_client

client = start_client()

# %%
mtypes = client.materialize.query_table(MTYPES_TABLE)
mtypes = mtypes.query("classification_system == 'inhibitory_neuron'")


# %%

degrees = client.materialize.query_view(
    "synapses_pni_2_in_out_degree", random_sample=10
)
