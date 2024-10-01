# %%

import pickle

import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist

from pkg.constants import OUT_PATH
from pkg.plot import savefig, set_context
from pkg.utils import load_casey_palette, load_manifest, load_mtypes

# %%

set_context()

client = cc.CAVEclient("minnie65_phase3_v1")
mtypes = load_mtypes(client)

distance_colors = sns.color_palette("Set1", n_colors=5)
distance_palette = dict(
    zip(
        ["euclidean", "cityblock", "jensenshannon", "cosine", "hamming"],
        distance_colors,
    )
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


# %%


def compute_diffs_to_final(sequence_df):
    # sequence = sequence_df.index.get_level_values("sequence").unique()[0]
    final_row_idx = sequence_df.index.get_level_values("order").max()
    final_row = sequence_df.loc[final_row_idx].fillna(0).values.reshape(1, -1)
    X = sequence_df.fillna(0).values

    sample_wise_metrics = []
    for metric in ["euclidean", "cityblock", "jensenshannon", "cosine", "hamming"]:
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
ctype_hues = load_casey_palette()

root_ids = (
    meta_diff_df.index.get_level_values("root_id").unique().intersection(manifest.index)
)
meta_diff_df = meta_diff_df.loc[root_ids]
# root_mtypes = root_ids.map(manifest["target_id"]).map(column_mtypes)
# root_mtypes = root_ids.map(manifest["target_id"]).map(column_mtypes)
root_mtypes = manifest.loc[root_ids, "mtype"]

from pkg.constants import COLUMN_MTYPES_TABLE

column_mtypes = client.materialize.query_table(COLUMN_MTYPES_TABLE)

# TODO need to redo this mapping via targets
column_mtypes = column_mtypes.set_index("target_id")["cell_type"]


# %%
root_id_ctype_hues = root_mtypes.map(ctype_hues)
root_id_ctype_hues.index = root_id_ctype_hues.index.astype(str)

# %%
XLABEL = "# operations"


# %%


# TODO make it so that the X-axes align? i think this just means getting rid of some
# unconsequential edits from the log in the historical


fig, axs = plt.subplots(
    3, 4, figsize=(16, 12), constrained_layout=True, sharey="row", sharex=False
)


info = all_infos.set_index(["root_id", "scheme", "order_by", "random_seed", "order"])

idx = pd.IndexSlice

# distance = "jensenshannon"
distance = "euclidean"

name_map = {
    "props_by_mtype": "Proportion of outputs\n by M-type",
    "spatial_props": "Out synapse probability\n by radial distance",
    "spatial_props_by_mtype": "Out synapse probability\n by radial distance\nand M-type",
}
scheme_map = {
    "historical": "Historical",
    "lumped-time": "Lumped-time",
    "clean-and-merge-time": "Clean-and-merge\nordered by time",
    "clean-and-merge-random": "Clean-and-merge\nordered randomly",
}

for i, feature in enumerate(
    ["props_by_mtype", "spatial_props", "spatial_props_by_mtype"]
):
    for j, scheme in enumerate(
        ["historical", "lumped-time", "clean-and-merge-time", "clean-and-merge-random"]
    ):
        if scheme == "historical":
            historical_diff_df = meta_diff_df.loc[idx[:, "historical", :, :]]
        elif scheme == "lumped-time":
            historical_diff_df = meta_diff_df.loc[idx[:, "lumped-time", :, :]]
        elif scheme == "clean-and-merge-time":
            historical_diff_df = meta_diff_df.loc[idx[:, "clean-and-merge", "time", :]]
        elif scheme == "clean-and-merge-random":
            historical_diff_df = meta_diff_df.loc[
                idx[:, "clean-and-merge", "random", :]
            ]

        historical_diff_df = pd.concat(historical_diff_df[feature].to_list())
        historical_diff_df = historical_diff_df.reset_index(drop=False)
        cumulative_n_operations = historical_diff_df.set_index(
            ["root_id", "scheme", "order_by", "random_seed", "order"]
        ).index.map(info["cumulative_n_operations"])
        historical_diff_df["root_id_str"] = historical_diff_df["root_id"].astype(str)
        historical_diff_df["cumulative_n_operations"] = cumulative_n_operations
        historical_diff_df["mtype"] = historical_diff_df["root_id"].map(
            manifest["mtype"]
        )
        ax = axs[i, j]
        sns.lineplot(
            data=historical_diff_df,
            x="cumulative_n_operations",
            y=distance,
            ax=ax,
            legend=False,
            linewidth=1,
            hue="root_id_str",
            alpha=0.5,
            palette=root_id_ctype_hues.to_dict(),
            # units="root_id_str",
            # estimator=None,
        )
        if i == 0:
            ax.set_title(scheme_map[scheme])
        if i == 2:
            ax.set_xlabel(XLABEL)
        else:
            ax.set_xlabel("")
        if j == 0:
            ax.text(
                -0.45,
                0.5,
                name_map[feature],
                transform=ax.transAxes,
                ha="right",
                va="center",
                rotation=0,
            )
            ax.set_ylabel(distance.capitalize())


savefig(
    f"diffs-from-final-by-scheme-distance={distance}",
    fig,
    folder="sequence_output_metrics",
)

# %%

example_root_ids = manifest.query("is_sample").index

# %%
# plotting the historical ordering for example cells

feature = "props_by_mtype"
for scheme in ["historical", "lumped-time"]:
    for root_id in example_root_ids:
        historical_df: pd.DataFrame
        historical_df = meta_features_df.loc[idx[root_id, scheme, :, :]][feature].iloc[
            0
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        historical_df_long = (
            historical_df.melt(ignore_index=False, value_name="probability")
            .reset_index()
            .fillna(0)
        )
        sns.lineplot(
            data=historical_df_long,
            x="order",
            y="probability",
            hue="post_mtype",
            ax=ax,
            legend=False,
            palette=ctype_hues,
        )
        ax.set(ylabel="Proportion of outputs", xlabel=XLABEL)
        savefig(
            f"{scheme}-ordering-{feature}-root_id={root_id}",
            fig,
            folder="sequence_output_metrics",
            doc_save=True,
            group=f"{scheme}-ordering-{feature}",
            caption=root_id,
        )


# %%

# look at distance from final for example cells in historical ordering

feature = "props_by_mtype"
scheme = "historical"
for scheme in ["historical", "lumped-time"]:
    for root_id in example_root_ids:
        historical_diff_df: pd.DataFrame
        historical_diff_df = meta_diff_df.loc[idx[root_id, scheme, :, :]][feature].iloc[
            0
        ]
        if scheme == "historical":
            historical_diff_df = historical_diff_df.droplevel(
                ["root_id", "scheme", "order_by", "random_seed", "operation_id"]
            ).reset_index(drop=False)
        else:
            historical_diff_df = historical_diff_df.droplevel(
                ["root_id", "scheme", "order_by", "random_seed", "metaoperation_id"]
            ).reset_index(drop=False)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        for metric in ["euclidean", "cityblock", "jensenshannon", "cosine"]:
            sns.lineplot(
                data=historical_diff_df,
                x="order",
                y=metric,
                ax=ax,
                label=metric,
                color=distance_palette[metric],
                linewidth=2,
            )
            ax.set(ylabel="Distance to final", xlabel=XLABEL)
            ax.legend

        savefig(
            f"{scheme}-ordering-{feature}-distance-root_id={root_id}",
            fig,
            folder="sequence_output_metrics",
            doc_save=True,
            group=f"{scheme}-ordering-{feature}-distance",
            caption=root_id,
        )

# %%

# look at a summary plot showing distance from final ('euclidean') for all cells in
# historical ordering, all on the same axes

distance = "euclidean"
feature = "props_by_mtype"
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
historical_diff_df = (
    pd.concat(meta_diff_df.loc[idx[:, "historical", :, :], feature].to_list())
    .droplevel(["scheme", "order_by", "random_seed", "operation_id"])
    .reset_index(drop=False)
)


sns.lineplot(
    data=historical_diff_df,
    x="order",
    y=distance,
    units="root_id",
    estimator=None,
    ax=ax,
    legend=False,
    alpha=0.5,
    linewidth=0.5,
    color="black",
    zorder=-1,
)
sns.lineplot(
    data=historical_diff_df,
    x="order",
    y=distance,
    ax=ax,
    legend=False,
    color="red",
    zorder=2,
)
ax.set(ylabel="Distance to final", xlabel=XLABEL)

savefig(
    f"historical-ordering-props_by_mtype-distance={distance}-summary",
    fig,
    folder="sequence_output_metrics",
    doc_save=True,
)

# %%
feature = "partners"
scheme = "historical"
for root_id in example_root_ids:
    historical_diff_df: pd.DataFrame
    historical_diff_df = meta_diff_df.loc[idx[root_id, scheme, :, :]][feature].iloc[0]
    historical_diff_df = historical_diff_df.droplevel(
        ["root_id", "scheme", "order_by", "random_seed", "operation_id"]
    ).reset_index(drop=False)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for metric in ["hamming"]:
        sns.lineplot(
            data=historical_diff_df,
            x="order",
            y=metric,
            ax=ax,
            label=metric,
            color=distance_palette[metric],
            linewidth=2,
        )
        ax.set(ylabel="Distance to final", xlabel=XLABEL)
        ax.legend

    # savefig(
    #     f"{scheme}-ordering-{feature}-distance-root_id={root_id}",
    #     fig,
    #     folder="sequence_output_metrics",
    #     doc_save=True,
    #     group=f"{scheme}-ordering-{feature}-distance",
    #     caption=root_id,
    # )


# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.histplot(data=historical_diff_df, y="euclidean", x="order", ax=ax)

# %%
historical_diff_df["bin"] = pd.cut(historical_diff_df[distance], bins=40)

# %%
diff_histogram_df = (
    historical_diff_df.groupby("order")["bin"].value_counts().unstack().T
)
norm_diff_histogram_df = diff_histogram_df.div(diff_histogram_df.sum(axis=0), axis=1)
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.heatmap(norm_diff_histogram_df, ax=ax)
ax.invert_yaxis()


# %%
# plotting the historical ordering for example cells

feature = "props_by_mtype"
scheme = "clean-and-merge"
for root_id in [864691135213953920]:
    feature_df: pd.DataFrame = pd.concat(
        meta_features_df.loc[idx[root_id, scheme, :, :]][feature].to_list()
    )
    feature_df = feature_df.fillna(0)
    feature_df_long = feature_df.melt(
        ignore_index=False, value_name="probability"
    ).reset_index()
    cumulative_n_operations = feature_df.droplevel(["metaoperation_id"]).index.map(
        info["cumulative_n_operations"]
    )
    cumulative_n_operations = cumulative_n_operations.to_series()
    cumulative_n_operations.index = feature_df.index
    feature_df_long["cumulative_n_operations"] = feature_df_long.set_index(
        ["root_id", "scheme", "order_by", "random_seed", "metaoperation_id", "order"]
    ).index.map(cumulative_n_operations)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.lineplot(
        data=feature_df_long.query("order_by == 'random'"),
        x="cumulative_n_operations",
        y="probability",
        units="random_seed",
        estimator=None,
        hue="post_mtype",
        ax=ax,
        legend=False,
        palette=ctype_hues,
        linewidth=0.4,
        alpha=0.25,
    )
    sns.lineplot(
        data=feature_df_long.query("order_by == 'time'"),
        x="cumulative_n_operations",
        y="probability",
        hue="post_mtype",
        ax=ax,
        legend=False,
        palette=ctype_hues,
        linewidth=2,
        alpha=1,
        zorder=2,
    )
    ax.set(ylabel="Proportion of outputs", xlabel=XLABEL)
    savefig(
        f"{scheme}-ordering-{feature}-root_id={root_id}",
        fig,
        folder="sequence_output_metrics",
        doc_save=True,
        group=f"{scheme}-ordering-{feature}",
        caption=root_id,
    )

# %%


for root_id in example_root_ids:
    diff_df: pd.DataFrame = pd.concat(
        meta_diff_df.loc[idx[root_id, "clean-and-merge", :, :]][feature].to_list()
    )
    cumulative_n_operations = diff_df.droplevel(["metaoperation_id"]).index.map(
        info["cumulative_n_operations"]
    )
    cumulative_n_operations = cumulative_n_operations.to_series()
    cumulative_n_operations.index = diff_df.index
    diff_df_long = diff_df.melt(
        ignore_index=False, value_name="distance", var_name="distance_type"
    ).reset_index(drop=False)
    diff_df_long["cumulative_n_operations"] = diff_df_long.set_index(
        ["root_id", "scheme", "order_by", "random_seed", "metaoperation_id", "order"]
    ).index.map(cumulative_n_operations)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.lineplot(
        data=diff_df_long.query("order_by == 'random'"),
        x="cumulative_n_operations",
        units="random_seed",
        estimator=None,
        y="distance",
        hue="distance_type",
        ax=ax,
        linewidth=0.4,
        alpha=0.25,
        palette=distance_palette,
        legend=False,
    )
    sns.lineplot(
        data=diff_df_long.query("order_by == 'time'"),
        x="cumulative_n_operations",
        units="random_seed",
        estimator=None,
        y="distance",
        hue="distance_type",
        ax=ax,
        linewidth=2,
        alpha=1,
        palette=distance_palette,
        legend=True,
        zorder=2,
    )
    ax.set(ylabel="Distance to final", xlabel=XLABEL)
    sns.move_legend(ax, "upper right", title="Distance type")

    savefig(
        f"{scheme}-ordering-{feature}-distance-root_id={root_id}",
        fig,
        folder="sequence_output_metrics",
        doc_save=True,
        group=f"{scheme}-ordering-{feature}-distance",
        caption=root_id,
    )

# %%

# look at a summary plot showing distance from final ('euclidean') for all cells in
# historical ordering, all on the same axes

distance = "euclidean"
feature = "props_by_mtype"
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
diff_df = pd.concat(
    meta_diff_df.loc[idx[:, "clean-and-merge", :, :], feature].to_list()
)
cumulative_n_operations = diff_df.droplevel(["metaoperation_id"]).index.map(
    info["cumulative_n_operations"]
)
cumulative_n_operations = cumulative_n_operations.to_series()
cumulative_n_operations.index = diff_df.index
diff_df_long = diff_df.melt(
    ignore_index=False, value_name="distance", var_name="distance_type"
).reset_index(drop=False)
diff_df_long["cumulative_n_operations"] = diff_df_long.set_index(
    ["root_id", "scheme", "order_by", "random_seed", "metaoperation_id", "order"]
).index.map(cumulative_n_operations)
diff_df_long["sequence"] = list(
    zip(
        diff_df_long["root_id"],
        diff_df_long["random_seed"],
    )
)
sns.lineplot(
    data=diff_df_long.query("distance_type == @distance").query('order_by == "random"'),
    x="cumulative_n_operations",
    y="distance",
    units="sequence",
    estimator=None,
    ax=ax,
    legend=False,
    alpha=0.2,
    linewidth=0.2,
    color="black",
    zorder=-1,
)
sns.lineplot(
    data=diff_df_long.query("distance_type == @distance").query('order_by == "random"'),
    x="cumulative_n_operations",
    y="distance",
    ax=ax,
    legend=False,
    color="red",
    zorder=2,
)
ax.set(ylabel="Distance to final", xlabel=XLABEL)

savefig(
    f"clean-and-merge-ordering-props_by_mtype-distance={distance}-summary",
    fig,
    folder="sequence_output_metrics",
    doc_save=True,
)


# %%

# plotting everything in a 3x3

distance = "euclidean"
distance_name_map = {
    "euclidean": "Distance to final\n(euclidean)",
    "cityblock": "Distance to final\n(manhattan)",
    "jensenshannon": "Distance to final\n(Jensen-Shannon)",
    "cosine": "Distance to final\n(cosine)",
}
for root_id in example_root_ids:
    fig, axs = plt.subplots(
        3, 3, figsize=(16, 12), constrained_layout=True, sharey="row", sharex=False
    )
    for i, feature in enumerate(
        ["props_by_mtype", "spatial_props", "spatial_props_by_mtype"]
    ):
        for j, scheme in enumerate(
            ["historical", "clean-and-merge-time", "clean-and-merge-random"]
        ):
            if scheme == "historical":
                historical_diff_df = meta_diff_df.loc[idx[root_id, "historical", :, :]]
            elif scheme == "clean-and-merge-time":
                historical_diff_df = meta_diff_df.loc[
                    idx[root_id, "clean-and-merge", "time", :]
                ]
            elif scheme == "clean-and-merge-random":
                historical_diff_df = meta_diff_df.loc[
                    idx[root_id, "clean-and-merge", "random", :]
                ]

            historical_diff_df = pd.concat(historical_diff_df[feature].to_list()).copy()
            historical_diff_df = historical_diff_df.reset_index(drop=False)
            cumulative_n_operations = historical_diff_df.set_index(
                ["root_id", "scheme", "order_by", "random_seed", "order"]
            ).index.map(info["cumulative_n_operations"])
            historical_diff_df["root_id_str"] = historical_diff_df["root_id"].astype(
                str
            )
            historical_diff_df[
                "cumulative_n_operations"
            ] = cumulative_n_operations.copy()

            historical_diff_df["mtype"] = historical_diff_df["root_id"].map(
                column_mtypes
            )

            ax = axs[i, j]
            if scheme == "clean-and-merge-random":
                sns.lineplot(
                    data=historical_diff_df,
                    x="cumulative_n_operations",
                    y=distance,
                    ax=ax,
                    legend=False,
                    linewidth=1,
                    hue="root_id_str",
                    alpha=0.75,
                    palette=root_id_ctype_hues.to_dict(),
                    units="random_seed",
                    estimator=None,
                )
            sns.lineplot(
                data=historical_diff_df,
                x="cumulative_n_operations",
                y=distance,
                ax=ax,
                legend=False,
                linewidth=2,
                hue="root_id_str",
                alpha=1,
                palette=root_id_ctype_hues.to_dict(),
            )
            if i == 0:
                ax.set_title(scheme_map[scheme])
            if i == 2:
                ax.set_xlabel(XLABEL)
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.text(
                    -0.45,
                    0.5,
                    name_map[feature],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=0,
                )
                ax.set_ylabel(distance_name_map[distance])

    savefig(
        f"diffs-from-final-by-scheme-distance={distance}-root_id={root_id}",
        fig,
        folder="sequence_output_metrics",
        doc_save=True,
    )

# %%

# histograms of number of edits to reach threshold
metric = "euclidean"

for i, feature in enumerate(["props_by_mtype"]):
    for j, scheme in enumerate(["clean-and-merge-time"]):
        if scheme == "historical":
            diff_df = meta_diff_df.loc[idx[:, "historical", :, :]]
        elif scheme == "clean-and-merge-time":
            diff_df = meta_diff_df.loc[idx[:, "clean-and-merge", "time", :]]
        elif scheme == "clean-and-merge-random":
            diff_df = meta_diff_df.loc[idx[:, "clean-and-merge", "random", :]]
        diff_df = pd.concat(diff_df[feature].to_list()).copy()
        diff_df = diff_df.reset_index(drop=False)
        cumulative_n_operations = diff_df.set_index(
            ["root_id", "scheme", "order_by", "random_seed", "order"]
        ).index.map(info["cumulative_n_operations"])
        diff_df["root_id_str"] = diff_df["root_id"].astype(str)
        diff_df["cumulative_n_operations"] = cumulative_n_operations.copy()

        diff_df["mtype"] = (
            diff_df["root_id"].map(manifest["target_id"]).map(column_mtypes)
        )

# %%

for delta in [0.5, 0.3, 0.1, 0.05]:
    diff_df["pass_threshold"] = diff_df[metric] < delta

    final_efforts = diff_df.groupby(["root_id"])["cumulative_n_operations"].max()

    diff_df["prop_effort"] = (
        diff_df["cumulative_n_operations"] / final_efforts[diff_df["root_id"]].values
    )

    diff_df = diff_df.sort_values("euclidean", ascending=False)

    first_pass_threshold = (
        diff_df.query("pass_threshold")
        .groupby(["root_id", "scheme", "order_by", "random_seed"], dropna=False)
        .first()
    )
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    hist_kws = dict(kde=True, element="step", stat="proportion", common_norm=False)

    sns.histplot(
        data=first_pass_threshold,
        x="cumulative_n_operations",
        hue="mtype",
        ax=axs[0],
        legend=False,
        **hist_kws,
    )

    sns.histplot(
        data=first_pass_threshold,
        x="prop_effort",
        hue="mtype",
        ax=axs[1],
        **hist_kws,
    )

    fig.suptitle(
        f"Threshold = {delta}, metric = {metric},\nfeature = {feature}, scheme={scheme}",
        y=1.02,
    )

    savefig(
        f"histogram-threshold-by-type-delta={delta}-metric={metric}-feature={feature}-scheme={scheme}",
        fig,
        folder="sequence_output_metrics",
        doc_save=True,
        group="histogram-threshold-by-type",
    )

# %%

scheme = "historical"
distance = "hamming"
feature = "partners"
op = "operation_id"
order_by = ""
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
diff_df = pd.concat(meta_diff_df.loc[idx[:, scheme, :, :], feature].to_list())
cumulative_n_operations = diff_df.droplevel([op]).index.map(
    info["cumulative_n_operations"]
)
cumulative_n_operations = cumulative_n_operations.to_series()
cumulative_n_operations.index = diff_df.index
diff_df_long = diff_df.melt(
    ignore_index=False, value_name="distance", var_name="distance_type"
).reset_index(drop=False)
diff_df_long["cumulative_n_operations"] = diff_df_long.set_index(
    ["root_id", "scheme", "order_by", "random_seed", op, "order"]
).index.map(cumulative_n_operations)
diff_df_long["sequence"] = list(
    zip(
        diff_df_long["root_id"],
        diff_df_long["random_seed"],
    )
)
diff_df_long["mtype"] = diff_df_long["root_id"].map(manifest["mtype"])
plot_df = (
    diff_df_long.query("distance_type == @distance")
    # .query('order_by == "random"')
    .copy()
)

max_n_ops = plot_df["cumulative_n_operations"].max()

import numpy as np

roots = plot_df["root_id"].unique()
seeds = plot_df["random_seed"].unique()
n_ops = np.arange(0, max_n_ops + 1)

plot_df.set_index(["root_id", "random_seed", "cumulative_n_operations"], inplace=True)

new_index = pd.MultiIndex.from_product([roots, seeds, n_ops], names=plot_df.index.names)

plot_df = plot_df.reindex(new_index)

plot_df = plot_df.ffill()

plot_df = plot_df.reset_index(drop=False)

sns.lineplot(
    data=plot_df,
    x="cumulative_n_operations",
    y="distance",
    hue="mtype",
    units="sequence",
    estimator=None,
    ax=ax,
    legend=False,
    alpha=0.2,
    linewidth=0.2,
    color="black",
    zorder=-1,
)
sns.lineplot(
    data=plot_df,
    x="cumulative_n_operations",
    y="distance",
    ax=ax,
    legend=False,
    color="red",
    zorder=2,
)
ax.set(ylabel="Distance to final", xlabel=XLABEL)

# savefig(
#     f"clean-and-merge-ordering-props_by_mtype-distance={distance}-summary",
#     fig,
#     folder="sequence_output_metrics",
#     doc_save=True,
# )

# %%
fig, axs = plt.subplots(5, 5, figsize=(25, 25), sharex=True, sharey=True)

for i, (root, data) in enumerate(diff_df_long.groupby("root_id")):
    data = data.query("distance_type == @distance")
    (
        x,
        y,
    ) = np.unravel_index(i, (5, 5))
    ax = axs[x, y]
    sns.lineplot(
        data=data,
        x="cumulative_n_operations",
        y="distance",
        hue="mtype",
        palette=ctype_hues,
        legend=False,
        ax=ax,
    )
    if i >= 24:
        break

# %%

feature = "props_by_mtype"
scheme = "clean-and-merge"
order_by = "time"
op = "metaoperation_id"

diff_df = meta_diff_df.loc[idx[:, scheme, order_by, :]][feature]
diff_df = pd.concat(diff_df.to_list()).copy()
cumulative_n_operations = diff_df.droplevel(op).index.map(
    info["cumulative_n_operations"]
)
diff_df["cumulative_n_operations"] = cumulative_n_operations

diff_df = diff_df.reset_index(drop=False)


diff_df["final_n_operations"] = diff_df.groupby("root_id")[
    "cumulative_n_operations"
].transform("max")

max_n_ops = diff_df["cumulative_n_operations"].max()

n_ops = np.arange(0, max_n_ops + 1)

roots = diff_df["root_id"].unique()
seeds = diff_df["random_seed"].unique()
diff_df.set_index(["root_id", "random_seed", "cumulative_n_operations"], inplace=True)

new_index = pd.MultiIndex.from_product(
    [roots, seeds, n_ops], names=["root_id", "random_seed", "cumulative_n_operations"]
)

diff_df = diff_df.reindex(new_index)

diff_df = diff_df.ffill()

diff_df = diff_df.reset_index(drop=False)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.lineplot(
    data=diff_df,
    x="cumulative_n_operations",
    y="cityblock",
    color="red",
    linewidth=3,
    ax=ax,
    zorder=10,
)
sns.lineplot(
    data=diff_df,
    x="cumulative_n_operations",
    y="cityblock",
    units="root_id",
    estimator=None,
    alpha=0.2,
    color="black",
    ax=ax,
    zorder=-1,
)

ax.set(xlim=(0, 200), xlabel="Number of operations", ylabel="Distance from final state")


diff_df = diff_df.copy()
diff_df["p_operations"] = (
    diff_df["cumulative_n_operations"] / diff_df["final_n_operations"]
)
diff_df = diff_df.query("p_operations <= 1.0")

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.lineplot(
    data=diff_df,
    x="p_operations",
    y="cityblock",
    # color="red",
    # linewidth=3,
    units="root_id",
    estimator=None,
    alpha=0.2,
    ax=ax,
    # zorder=10,
)
sns.lineplot(
    data=diff_df,
    x="p_operations",
    y="cityblock",
    color="red",
    linewidth=3,
    # units="root_id",
    # estimator=None,
    # alpha=0.2,
    ax=ax,
    zorder=10,
)

# %%
diff_df


# %%

# %%
red_lines = ax.get_children()[:2]

lines = ax.get_lines()[1:]
for line in lines:
    line.set_visible(False)


def update(i):
    if i <= 40:
        for line in lines[:i]:
            line.set_visible(True)
            line.set_alpha(0.2)
        for line in red_lines:
            line.set_visible(False)
        lines[i].set_visible(True)
        lines[i].set_alpha(1)
    if i > 40:
        for line in lines:
            line.set_visible(True)
            line.set_alpha(0.2)
        for line in red_lines:
            line.set_visible(True)


# ani = animation.FuncAnimation(
#     fig, update, frames=range(0, 80), interval=100, repeat=False
# )

# writer = animation.PillowWriter(fps=5)

# ani.save("test.gif", writer=writer)


# %%


def compute_diffs_to_final(sequence_df):
    # sequence = sequence_df.index.get_level_values("sequence").unique()[0]
    final_row_idx = sequence_df.index.get_level_values("order").max()
    final_row = sequence_df.loc[final_row_idx].fillna(0).values.reshape(1, -1)
    X = sequence_df.fillna(0).values

    sample_wise_metrics = []
    for metric in ["euclidean", "cityblock", "jensenshannon", "cosine", "hamming"]:
        distances = cdist(X, final_row, metric=metric)
        distances = pd.Series(
            distances.flatten(),
            name=metric,
            index=sequence_df.index.get_level_values("order"),
        )
        sample_wise_metrics.append(distances)
    sample_wise_metrics = pd.concat(sample_wise_metrics, axis=1)

    return sample_wise_metrics


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

# %%
feature_dfs = meta_features_df.loc[idx[:, scheme, order_by, :]][feature]
feature_df = pd.concat(feature_dfs.to_list()).copy()
cumulative_n_operations = feature_df.droplevel(op).index.map(
    info["cumulative_n_operations"]
)
feature_df["cumulative_n_operations"] = cumulative_n_operations

max_ops_by_cell = feature_df.groupby("root_id")["cumulative_n_operations"].max()

feature_df["prop_operations"] = feature_df[
    "cumulative_n_operations"
] / feature_df.index.get_level_values("root_id").map(max_ops_by_cell)

# %%

from sklearn.metrics import pairwise_distances

fig, axs = plt.subplots(
    2,
    4,
    figsize=(20, 10),
    sharey=True,
    constrained_layout=True,
    gridspec_kw=dict(hspace=0.1),
)

x = "cumulative_n_operations"
thresholds_by_x = {
    "cumulative_n_operations": [25, 50, 100, 200],
    "prop_operations": [0.1, 0.2, 0.5, 0.8],
}

for i, x in enumerate(["cumulative_n_operations", "prop_operations"]):
    for j, threshold in enumerate(thresholds_by_x[x]):
        ax = axs[i, j]
        for root_id in feature_df.index.get_level_values("root_id").unique():
            root_feature_df = feature_df.loc[root_id]
            X = (
                root_feature_df.fillna(0)
                .drop(columns=["cumulative_n_operations", "prop_operations"])
                .values
            )
            dists = pairwise_distances(X, metric="cityblock")
            # dists = pd.DataFrame(
            #     dists, index=root_feature_df.index, columns=root_feature_df.index
            # )
            idx = (root_feature_df[x] - threshold).abs().idxmin()
            iloc = root_feature_df.index.get_loc(idx)
            ds = dists[iloc, : iloc + 1]
            xs = root_feature_df[x].iloc[: iloc + 1]
            ax.scatter(xs, ds, color="black", alpha=0.2)
            ax.set(title=f"Threshold = {threshold}", ylabel="Distance to 'final' (L1)")
        if i == 1:
            ax.set(xlabel="Proportion of operations")
        else:
            ax.set(xlabel="Number of operations")

# ax.set(xlabel="Number of operations", ylabel="Cityblock distance to 'final' state")

# %%


def diag_indices(n, k=0):
    # get the kth diagonal, k>0 means upper, k<0 means lower
    rows, cols = np.diag_indices(n)
    if k > 0:
        return rows[:-k], cols[k:]
    elif k < 0:
        return rows[-k:], cols[:k]
    else:
        return rows, cols


set_context()

xs = []
ys = []
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for i, x in enumerate(["cumulative_n_operations", "prop_operations"]):
    for root_id in feature_df.index.get_level_values("root_id").unique():
        root_feature_df = feature_df.loc[root_id]
        X = (
            root_feature_df.fillna(0)
            .drop(columns=["cumulative_n_operations", "prop_operations"])
            .values
        )
        dists = pairwise_distances(X, metric="cityblock")

        inds_i, inds_j = diag_indices(X.shape[0], k=1)
        vals = dists[inds_i, inds_j]

        ax.scatter(
            root_feature_df["cumulative_n_operations"].iloc[1:],
            vals,
            color="black",
            alpha=0.1,
        )
        xs.extend(root_feature_df["cumulative_n_operations"].iloc[1:].to_list())
        ys.extend(vals)

ax.set(
    ylabel=r"$d(i-1, i)$ (L1 distance)", xlabel="Number of operations", xlim=(0, 200)
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

sns.histplot(x=xs, y=ys, bins=50, ax=ax)


# %%
sns.heatmap(X)

# %%

# %%

scheme = "clean-and-merge"

distance = "hamming"
feature = "partners"
op = "metaoperation_id"
order_by = "random"

idx = pd.IndexSlice
features_dfs = meta_features_df.loc[idx[:, scheme, order_by, :]]["props_by_mtype"]
feature_df = pd.concat(features_dfs.to_list()).copy().fillna(0)
cumulative_n_operations = feature_df.droplevel(op).index.map(
    info["cumulative_n_operations"]
)
feature_df["cumulative_n_operations"] = cumulative_n_operations

max_ops_by_cell = feature_df.groupby("root_id")["cumulative_n_operations"].max()

feature_df["prop_operations"] = feature_df[
    "cumulative_n_operations"
] / feature_df.index.get_level_values("root_id").map(max_ops_by_cell)

feature_df["prop_operations_bin"] = pd.cut(
    feature_df["prop_operations"], bins=20
).cat.codes

from scipy.cluster.hierarchy import fcluster, linkage

# fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)
# %%
labels_by_trial = []
for i, level in enumerate(np.linspace(0.1, 1, 20)):
    for j, p_neurons in enumerate(np.linspace(0.1, 1, 20)):
        for sample in range(25):
            feature_df["prop_op_dist"] = np.abs(feature_df["prop_operations"] - level)

            selected_feature_df = feature_df.copy()
            selected_feature_df = selected_feature_df.sample(frac=1, replace=True)
            selections = selected_feature_df.groupby("root_id")["prop_op_dist"].idxmin()

            selected_df = feature_df.loc[selections]
            selected_df = selected_df.sample(frac=p_neurons)
            n_edits = selected_df["cumulative_n_operations"].sum()
            selected_df = selected_df.drop(
                columns=[
                    "cumulative_n_operations",
                    "prop_operations",
                    "prop_operations_bin",
                    "prop_op_dist",
                ]
            )

            linkage_matrix = linkage(selected_df.values)
            labels = fcluster(linkage_matrix, 18, criterion="maxclust")
            labels = pd.Series(
                labels,
                index=selected_df.index.get_level_values("root_id"),
                name="label",
            )
            labels = labels.to_frame()
            labels["level"] = level
            labels["p_neurons"] = p_neurons
            labels["sample"] = sample
            labels["n_edits"] = n_edits

            labels_by_trial.append(labels)

        # ax = axs.flat[i]

        # sns.heatmap(
        #     selected_df.iloc[np.argsort(labels)].T,
        #     ax=ax,
        #     cmap="viridis",
        #     xticklabels=False,
        #     yticklabels=False,
        # )

        # sns.clustermap(
        #     selected_df.iloc[np.argsort(labels)].T,
        #     cmap="viridis",
        #     xticklabels=False,
        #     yticklabels=False,
        #     row_cluster=False,
        #     figsize=(20, 10),
        # )


# %%
final_labels = labels
from sklearn.metrics import adjusted_rand_score

ari_rows = []
for labels in labels_by_trial:
    intersect_index = labels.index.intersection(final_labels.index)
    y_trial = labels.loc[intersect_index, "label"].values
    y_final = final_labels.loc[intersect_index, "label"].values

    ari = adjusted_rand_score(y_trial, y_final)

    ari_rows.append(
        {
            "level": labels["level"].iloc[0],
            "p_neurons": labels["p_neurons"].iloc[0],
            "sample": labels["sample"].iloc[0],
            "n_edits": labels["n_edits"].iloc[0],
            "ari": ari,
        }
    )

# %%
ari_df = pd.DataFrame(ari_rows)

square_ari = ari_df.pivot_table(
    index="level", columns="p_neurons", values="ari", aggfunc="mean"
)
sns.heatmap(square_ari)

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
square_n_edits = ari_df.pivot_table(
    index="level", columns="p_neurons", values="n_edits", aggfunc="mean"
)
sns.heatmap(square_n_edits, ax=ax, square=True)
plt.contour(
    square_n_edits,
    levels=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    colors="white",
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

sns.heatmap(square_ari, ax=ax, square=True)
ax.contour(
    square_n_edits,
    levels=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    colors="white",
    origin="lower",
    extend="both",
)


# %%
ari_df["effort_bin"] = pd.cut(ari_df["n_edits"], bins=20).cat.codes

fig, axs = plt.subplots(4, 5, figsize=(20, 15))

for effort_bin, ari_data in ari_df.groupby("effort_bin"):
    x = ari_data["level"]
    y = ari_data["p_neurons"]
    z = ari_data["ari"]
    ax = axs.flat[effort_bin]
    sns.lineplot(x=x, y=z, ax=ax)

# %%
