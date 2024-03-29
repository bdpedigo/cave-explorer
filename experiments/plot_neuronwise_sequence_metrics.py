# %%

import pickle

import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist

from pkg.constants import COLUMN_MTYPES_TABLE, OUT_PATH
from pkg.plot import savefig, set_context
from pkg.utils import load_casey_palette, load_manifest, load_mtypes

# %%

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
ctype_hues = load_casey_palette()
column_mtypes = client.materialize.query_table(COLUMN_MTYPES_TABLE)

# TODO need to redo this mapping via targets
column_mtypes = column_mtypes.set_index("target_id")["cell_type"]

root_ids = meta_diff_df.index.get_level_values("root_id").to_series()

root_mtypes = root_ids.map(manifest["target_id"]).map(column_mtypes)

# %%
root_id_ctype_hues = root_mtypes.map(ctype_hues)
root_id_ctype_hues.index = root_id_ctype_hues.index.astype(str)

# %%

# TODO make it so that the X-axes align? i think this just means getting rid of some
# unconsequential edits from the log in the historical


fig, axs = plt.subplots(
    3, 3, figsize=(16, 12), constrained_layout=True, sharey="row", sharex=False
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
    "clean-and-merge-time": "Clean-and-merge\nordered by time",
    "clean-and-merge-random": "Clean-and-merge\nordered randomly",
}

for i, feature in enumerate(
    ["props_by_mtype", "spatial_props", "spatial_props_by_mtype"]
):
    for j, scheme in enumerate(
        ["historical", "clean-and-merge-time", "clean-and-merge-random"]
    ):
        if scheme == "historical":
            historical_diff_df = meta_diff_df.loc[idx[:, "historical", :, :]]
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
        historical_diff_df["mtype"] = historical_diff_df["root_id"].map(column_mtypes)
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
            ax.set_xlabel("# operations")
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
scheme = "historical"
for root_id in example_root_ids:
    historical_df: pd.DataFrame
    historical_df = meta_features_df.loc[idx[root_id, scheme, :, :]][feature].iloc[0]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    historical_df_long = historical_df.melt(
        ignore_index=False, value_name="probability"
    ).reset_index()
    sns.lineplot(
        data=historical_df_long,
        x="order",
        y="probability",
        hue="post_mtype",
        ax=ax,
        legend=False,
        palette=ctype_hues,
    )
    ax.set(ylabel="Proportion of outputs", xlabel="Operation")
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
for root_id in example_root_ids:
    historical_diff_df: pd.DataFrame
    historical_diff_df = meta_diff_df.loc[idx[root_id, scheme, :, :]][feature].iloc[0]
    historical_diff_df = historical_diff_df.droplevel(
        ["root_id", "scheme", "order_by", "random_seed", "operation_id"]
    ).reset_index(drop=False)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for metric in ["euclidean", "cityblock", "jensenshannon", "cosine"]:
        sns.lineplot(
            data=historical_diff_df,
            x="order",
            y=metric,
            ax=ax,
            label=metric,
        )
        ax.set(ylabel="Distance to final", xlabel="Operation")
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
ax.set(ylabel="Distance to final", xlabel="Operation")

savefig(
    f"historical-ordering-props_by_mtype-distance={distance}-summary",
    fig,
    folder="sequence_output_metrics",
    doc_save=True,
)


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
for root_id in example_root_ids:
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
    ax.set(ylabel="Proportion of outputs", xlabel="Operations")
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
        legend=False,
        zorder=2,
    )
    ax.set(ylabel="Distance to final", xlabel="Operation")


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
                ax.set_xlabel("# operations")
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
