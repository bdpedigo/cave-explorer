# %%

import pickle

import caveclient as cc
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist

from pkg.constants import OUT_PATH
from pkg.plot import set_context
from pkg.utils import load_manifest, load_mtypes

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

all_infos["mtype"] = all_infos["root_id"].map(manifest["mtype"])

all_infos = all_infos.set_index(
    ["root_id", "scheme", "order_by", "random_seed", "order"]
)


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
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from pkg.plot import set_context

features = [
    "n_pre_synapses",
    # "path_length",
    # "n_post_synapses",
    # "n_nodes",
    # "cumulative_n_operations",
]
set_context()
fig, axs = plt.subplots(
    4, 2, figsize=(10, 12), sharex="col", sharey=True, constrained_layout=True
)
all_infos["dummy"] = 0
scheme = "historical"
for i, (mtype, group_info) in enumerate(
    all_infos.query("mtype == 'PTC'").groupby("mtype")
):
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

    sns.histplot(data=diffs, x="n_pre_synapses", y="euclidean", ax=axs[i, 0])
    sns.histplot(data=diffs, x="path_length", y="euclidean", ax=axs[i, 1])

    ax = axs[i, 0]
    ax.text(-0.5, 0.5, mtype, transform=ax.transAxes, fontsize="xx-large", va="center")
    ax.set(ylabel="Euclidean distance \n to final output profile")

    ax = axs[3, 0]
    ax.set_xlabel("Number of pre-synapses")
    ax = axs[3, 1]
    ax.set_xlabel("Total Path length")

    print(f"{mtype}: number of neurons = {len(group_root_ids)}")
    for i in range(10):
        train_root_ids, test_root_ids = train_test_split(group_root_ids, test_size=0.2)

        train_diffs = diffs.loc[train_root_ids]
        test_diffs = diffs.loc[test_root_ids]

        X_train = train_diffs[features]
        y_train = train_diffs["euclidean"]
        X_test = test_diffs[features]
        y_test = test_diffs["euclidean"]

        # model = LinearRegression()
        model = RandomForestRegressor(n_estimators=500, max_depth=2)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        y_pred = pd.Series(index=y_test.index, data=y_pred)
        print(f"{mtype}: train_score={train_score}, test_score={test_score}")

    print()


fig, ax = plt.subplots(figsize=(6, 6))

sns.scatterplot(x=y_test, y=y_pred, ax=ax)

ax.set(xlabel="True Euclidean distance", ylabel="Predicted Euclidean distance")

# %%
fig, ax = plt.subplots(figsize=(6, 6))

sns.scatterplot(x=X_test["n_pre_synapses"], y=y_pred, ax=ax)


# %%
import seaborn as sns

sns.histplot(x=group_info["n_pre_synapses"])

# %%

ptcs = mtypes.query("cell_type == 'PTC'").sample(100)

# %%

import numpy as np
from tqdm.auto import tqdm

chunk_size = 100

ids_by_chunk = np.array_split(ptcs.index, len(ptcs) // chunk_size)

syns_by_chunk = []
for chunk in tqdm(ids_by_chunk):
    syns = client.materialize.synapse_query(
        pre_ids=chunk,
    )
    syns_by_chunk.append(syns)

# %%
# pre_syn_counts = pd.Series(index=ptcs.index, dtype=int, name="n_pre_synapses")
# for root in tqdm(ptcs.index):
#     count = client.materialize.query_table(
#         table="synapses_pni_2",
#         filter_in_dict={"pre_pt_root_id": root},
#         get_counts=True,
#     ).loc[0, "count"]
#     pre_syn_counts[root] = count

# %%

all_syns = pd.concat(syns_by_chunk, axis=0)
# %%
sns.histplot(all_syns.groupby("pre_pt_root_id").size())
