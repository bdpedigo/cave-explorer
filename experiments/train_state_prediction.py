# %%

import pickle
from itertools import chain, combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

from pkg.constants import OUT_PATH
from pkg.plot import savefig, set_context
from pkg.utils import load_manifest, load_mtypes, start_client

# %%

file_name = "train_state_prediction"

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

precision_recall_df = pd.concat(meta_features_df["pre_precision_recall"].tolist())

# %%

scheme = "historical"
group_info = all_infos.copy()

idx = pd.IndexSlice
diffs = meta_diff_df.loc[idx[:, scheme], :]["props_by_mtype"].values
diffs = pd.concat(diffs, axis=0).copy().reset_index()

precision_recall_df = precision_recall_df.loc[idx[:, scheme], :].copy()

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

diffs = diffs.join(
    precision_recall_df.reset_index()
    .set_index(["root_id", "random_seed", "order"])
    .drop(columns=["scheme", "order_by"])
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

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

sns.scatterplot(
    data=diffs, x="euclidean", y="precision", s=1, alpha=0.3, linewidth=0, ax=axs[0]
)
axs[0].axvline(0.2, color="black", lw=1, ls="--")
sns.scatterplot(
    data=diffs, x="euclidean", y="recall", s=1, alpha=0.3, linewidth=0, ax=axs[1]
)
axs[1].axvline(0.2, color="black", lw=1, ls="--")

savefig("precision_recall_scatter_vs_distance", fig, file_name, doc_save=True)

# %%

filt = diffs[diffs["euclidean"] < 0.2].copy()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

filt["weight"] = 1 / filt.groupby("root_id").transform("size")

sns.histplot(
    data=filt, x="precision", weights="weight", ax=axs[0], bins=20, stat="percent"
)

sns.histplot(
    data=filt, x="recall", weights="weight", ax=axs[1], bins=20, stat="percent"
)

fig.suptitle("States with euclidean distance < 0.2")

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

compare_models = False
if compare_models:
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
                pd.Series(y[idx], index=x[idx])
                .groupby(level=0)
                .aggregate(["min", "max"])
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

    results_df.groupby(all_features + ["model_name"])["f1"].mean().sort_values(
        ascending=False
    ).head(20)

# %%
final_feature_set = ["n_pre_synapses", "n_post_synapses", "n_nodes"]

X, y = extract_features_and_labels(diffs)
X = X[final_feature_set]

final_model = LinearDiscriminantAnalysis()
final_model.fit(X, y)
y_pred = final_model.predict(X)

# %%

y_scores = final_model.decision_function(X)
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
# save final model
# TODO refactor this to use the joblib/skops thingy

save_path = OUT_PATH / file_name

with open(save_path / "state_prediction_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

# %%
