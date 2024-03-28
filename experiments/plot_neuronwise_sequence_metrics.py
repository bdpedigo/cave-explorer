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

# # %%
# sub_df = pd.concat(
#     meta_features_df.query("root_id == @root_id & scheme == 'historical'")[
#         "spatial_props"
#     ].values
# )
# cols = sub_df.columns
# mids = [
#     interval.mid for interval in sub_df.columns.get_level_values("radial_to_nuc_bin")
# ]

# fig, ax = plt.subplots(1, 1, figsize=(6, 5))
# colors = sns.color_palette("coolwarm_r", n_colors=sub_df.shape[0])

# for i, (operation_id, row) in enumerate(sub_df.iterrows()):
#     sns.lineplot(
#         y=row.values,
#         x=mids,
#         ax=ax,
#         alpha=0.5,
#         linewidth=0.5,
#         color=colors[i],
#         legend=False,
#     )


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

root_ids = manifest.query("is_sample").index

# %%
# plotting the historical ordering for example cells

feature = "props_by_mtype"
scheme = "historical"
for root_id in root_ids:
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

# plotting everything in a 3x3

distance = "euclidean"
distance_name_map = {
    "euclidean": "Distance to final\n(euclidean)",
    "cityblock": "Distance to final\n(manhattan)",
    "jensenshannon": "Distance to final\n(Jensen-Shannon)",
    "cosine": "Distance to final\n(cosine)",
}
for root_id in root_ids:
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
