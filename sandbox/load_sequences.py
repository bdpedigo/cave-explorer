# %%
import os

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"
os.environ["LAZYCLOUD_RECOMPUTE"] = "False"
os.environ["LAZYCLOUD_USE_CLOUD"] = "True"

import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cloudfiles import CloudFiles
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from pkg.constants import OUT_PATH
from pkg.neuronframe import NeuronFrameSequence, load_neuronframe
from pkg.plot import savefig
from pkg.sequence import create_merge_and_clean_sequence
from pkg.utils import load_casey_palette, load_mtypes

# %%
cloud_bucket = "allen-minnie-phase3"
folder = "edit_sequences"

cf = CloudFiles(f"gs://{cloud_bucket}/{folder}")

files = list(cf.list())
files = pd.DataFrame(files, columns=["file"])

# pattern is root_id=number as the beginning of the file name
# extract the number from the file name and store it in a new column
files["root_id"] = files["file"].str.split("=").str[1].str.split("-").str[0].astype(int)
files["order_by"] = files["file"].str.split("=").str[2].str.split("-").str[0]
files["random_seed"] = files["file"].str.split("=").str[3].str.split("-").str[0]


file_counts = files.groupby("root_id").size()
has_all = file_counts[file_counts == 12].index

files_finished = files.query("root_id in @has_all")

# %%

client = cc.CAVEclient("minnie65_phase3_v1")
mtypes = load_mtypes(client)

# %%


def compute_target_stats(seq: NeuronFrameSequence):
    post_mtype_stats = seq.synapse_groupby_metrics(by="post_mtype", which="pre")
    bouts = seq.sequence_info["has_merge"].fillna(False).cumsum()
    bouts.name = "bout"
    bout_exemplars = (
        seq.sequence_info.index.to_series().groupby(bouts).apply(lambda x: x.iloc[-1])
    )
    # bout_info = seq.sequence_info.loc[bout_exemplars.values]
    bout_post_mtype_stats = post_mtype_stats.query(
        "metaoperation_id.isin(@bout_exemplars)"
    )
    return bout_post_mtype_stats


# %%

recompute = False
if recompute:
    root_ids = files_finished["root_id"].unique()
    all_targets_stats = {}
    all_infos = {}
    pbar = tqdm(total=len(root_ids), desc="Computing target stats...")
    i = 0
    for root_id, rows in files_finished.groupby("root_id"):
        neuron = load_neuronframe(root_id, client)
        neuron.pre_synapses["post_mtype"] = neuron.pre_synapses["post_pt_root_id"].map(
            mtypes["cell_type"]
        )
        for keys, sub_rows in rows.groupby(["order_by", "random_seed"]):
            order_by, random_seed = keys
            if order_by == "time" or order_by == "random":
                sequence = create_merge_and_clean_sequence(
                    neuron, root_id, order_by=order_by, random_seed=random_seed
                )

                target_stats = compute_target_stats(sequence)
                target_stats["root_id"] = root_id
                target_stats["order_by"] = order_by
                target_stats["random_seed"] = random_seed
                all_targets_stats[(root_id, order_by, random_seed)] = target_stats.drop(
                    ["pre_synapses", "post_synapses", "applied_edits"], axis=1
                )

                info = sequence.sequence_info
                info["root_id"] = root_id
                info["order_by"] = order_by
                info["random_seed"] = random_seed
                all_infos[(root_id, order_by, random_seed)] = info.drop(
                    ["pre_synapses", "post_synapses", "applied_edits"], axis=1
                )
        i += 1
        # if i > 5:
        #     break
        pbar.update(1)

    pbar.close()

    save_path = OUT_PATH / "load_sequences"

    all_target_stats = pd.concat(all_targets_stats.values())
    all_target_stats["cumulative_n_operations"].fillna(0, inplace=True)
    all_target_stats["root_id_str"] = all_target_stats["root_id"].astype(str)
    all_target_stats["sequence"] = (
        all_target_stats["root_id_str"]
        + "-"
        + all_target_stats["order_by"]
        + "-"
        + all_target_stats["random_seed"].astype(str)
    )
    all_target_stats.to_csv(save_path / "all_target_stats.csv")

    all_infos = pd.concat(all_infos.values())
    all_infos["root_id_str"] = all_infos["root_id"].astype(str)
    all_infos["sequence"] = (
        all_infos["root_id_str"]
        + "-"
        + all_infos["order_by"]
        + "-"
        + all_infos["random_seed"].astype(str)
    )
    all_infos = all_infos.reset_index(drop=False).set_index(["sequence", "order"])
    all_infos.to_csv(save_path / "all_infos.csv")

all_target_stats = pd.read_csv(
    OUT_PATH / "load_sequences" / "all_target_stats.csv", index_col=0
)
all_target_stats["metaoperation_id"] = all_target_stats["metaoperation_id"].astype(
    "Int64"
)
all_infos = pd.read_csv(OUT_PATH / "load_sequences" / "all_infos.csv", index_col=[0, 1])

# %%

query_neurons = client.materialize.query_table("connectivity_groups_v507")
ctype_map = query_neurons.set_index("pt_root_id")["cell_type"]


# %%


def process_sequence_diffs(sequence_df):
    sequence = sequence_df.index.get_level_values("sequence").unique()[0]
    final_row_idx = sequence_df.index.get_level_values("order").max()
    final_row = sequence_df.loc[(sequence, final_row_idx)].values.reshape(1, -1)
    X = sequence_df.values

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


target_stats_by_state = all_target_stats.pivot(
    index=["sequence", "order"], columns="post_mtype", values="prop"
).fillna(0)

diffs_from_final = target_stats_by_state.groupby("sequence").apply(
    process_sequence_diffs
)
diffs_from_final = diffs_from_final.join(all_infos)
# %%
diffs_from_final

# %%
ctype_hues = load_casey_palette()
sns.set_context("talk")

# %%

y = "cosine"
fig, axs = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True, sharex=True)
for i, y in enumerate(["euclidean", "cityblock", "jensenshannon", "cosine"]):
    sns.lineplot(
        data=diffs_from_final.loc["864691134886015738-time-None"],
        x="cumulative_n_operations",
        y=y,
        estimator=None,
        alpha=1,
        linewidth=2,
        legend=False,
        ax=axs.flat[i],
    )

# %%
y = "cityblock"
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.lineplot(
    data=diffs_from_final.query("order_by == 'random'"),
    x="cumulative_n_operations",
    y=y,
    hue="root_id_str",
    units="sequence",
    estimator=None,
    alpha=0.5,
    linewidth=0.25,
    ax=ax,
    legend=False,
)
sns.lineplot(
    data=diffs_from_final.query("order_by == 'time'"),
    x="cumulative_n_operations",
    y=y,
    hue="root_id_str",
    alpha=1,
    linewidth=1,
    ax=ax,
    legend=False,
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.lineplot(
    data=diffs_from_final,
    x="cumulative_n_operations",
    y=y,
    hue="root_id_str",
    linewidth=1,
    ax=ax,
    legend=False,
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.kdeplot(
    data=diffs_from_final,
    x="cumulative_n_operations",
    y=y,
    kde=True,
    ax=ax,
    clip=((0, 1000), (0, 2)),
)
sns.histplot(
    data=diffs_from_final,
    x="cumulative_n_operations",
    y=y,
    kde=True,
    ax=ax,
)
# %%
column_mtypes = client.materialize.query_table("allen_column_mtypes_v2")
column_mtypes.set_index("pt_root_id", inplace=True)
diffs_from_final["mtype"] = diffs_from_final["root_id"].map(column_mtypes["cell_type"])

# %%
fig, axs = plt.subplots(4, 4, figsize=(10, 10), sharey="row", sharex=True)

sns.set_context("talk")
for i, y in enumerate(["euclidean", "cityblock", "jensenshannon", "cosine"]):
    for j, group in enumerate(diffs_from_final["mtype"].unique()):
        ax = axs[i, j]
        sns.lineplot(
            data=diffs_from_final.query("mtype == @group").query(
                "order_by == 'random'"
            ),
            x="cumulative_n_operations",
            y=y,
            linewidth=0.1,
            units="sequence",
            estimator=None,
            alpha=0.5,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(y.capitalize())
        ax.set_xticks([0, 100, 200, 300])
        if i == 0:
            ax.set_title(group)
        ax.spines[["top", "right"]].set_visible(False)

fig.text(0.54, 0.0, "Cumulative operations", ha="center")
plt.tight_layout()
savefig("diffs-from-final-access-order-random-by-mtype", fig, folder="load_sequences")

# TODO need a better way to visualize this, some kind of smoothing or binning

# %%

import numpy as np

bins = np.arange(0, 300, 25)
for i in range(len(bins) - 1):
    start = bins[i]
    stop = bins[i + 1]
    query_data = diffs_from_final.query(
        "cumulative_n_operations < @stop & cumulative_n_operations >= @start"
    )
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.histplot(
        data=query_data,
        x="euclidean",
        hue="mtype",
        stat="density",
        common_norm=False,
        alpha=0.3,
        linewidth=0.5,
        bins=15,
        ax=ax,
    )
    sns.kdeplot(
        data=query_data,
        x="euclidean",
        hue="mtype",
        common_norm=False,
        linewidth=3,
        clip=(0, 4),
        ax=ax,
    )

# %%
bins = np.arange(0, 300, 25)
diffs_from_final["cumulative_n_operations_bin"] = pd.cut(
    diffs_from_final["cumulative_n_operations"], bins
)

fig, ax = plt.subplots(1, 1, figsize=(12, 5))

sns.violinplot(
    data=diffs_from_final,
    x="cumulative_n_operations_bin",
    y="euclidean",
    hue="mtype",
    inner="quartile",
    linewidth=0.5,
    ax=ax,
    scale="count",
)
ax.set_ylabel("L2 distance from final")
ax.set_xlabel("Cumulative operations")
ax.get_legend().set_title("M-type")
# rotate x labels
for item in ax.get_xticklabels():
    item.set_rotation(45)

ax.axhline(0, color="black", linewidth=1)
savefig(
    "diffs-from-final-access-order-random-by-mtype-violin", fig, folder="load_sequences"
)

# %%
bins = np.arange(0, 300, 25)
diffs_from_final["cumulative_n_operations_bin"] = pd.cut(
    diffs_from_final["cumulative_n_operations"], bins
)

fig, ax = plt.subplots(1, 1, figsize=(12, 5))

sns.stripplot(
    data=diffs_from_final,
    x="cumulative_n_operations_bin",
    y="euclidean",
    hue="mtype",
    ax=ax,
    dodge=True,
    s=0.5,
    alpha=0.5,
    jitter=0.4,
    linewidth=0,
    zorder=2,
    legend=False,
    color="black",
)
sns.violinplot(
    data=diffs_from_final,
    x="cumulative_n_operations_bin",
    y="euclidean",
    hue="mtype",
    inner="quartile",
    linewidth=1,
    ax=ax,
    scale="count",
    zorder=1,
)

# rotate x labels
for item in ax.get_xticklabels():
    item.set_rotation(45)

ax.axhline(0, color="black", linewidth=1)
ax.set_ylabel("L2 distance from final")
ax.set_xlabel("Cumulative operations")

# %%
pd.Series(diffs_from_final["root_id"].unique()).map(column_mtypes["cell_type"])

# %%

import numpy as np

bins = np.arange(0, 500, 50)


# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
root_id = all_target_stats["root_id"].unique()[10]
root_target_stats = all_target_stats.query("root_id == @root_id").reset_index()
sns.lineplot(
    data=root_target_stats.query("order_by == 'time'"),
    x="cumulative_n_operations",
    y="prop",
    hue="post_mtype",
    palette=ctype_hues,
    ax=ax,
    legend=False,
    linewidth=3,
)
sns.lineplot(
    data=root_target_stats.query("order_by == 'random'"),
    x="cumulative_n_operations",
    y="prop",
    hue="post_mtype",
    palette=ctype_hues,
    ax=ax,
    legend=False,
    units="random_seed",
    estimator=None,
    linewidth=0.5,
    alpha=0.5,
)
ax.set_xlabel("Cumulative number of operations")
ax.set_ylabel("Proportion of synapses")
ax.spines[["top", "right"]].set_visible(False)
savefig(f"target-stats-random-vs-time-ordered-root_id={root_id}", fig)


# %%
# TODO pivot or pivot table here
X_df = all_target_stats.pivot_table(
    index=["root_id", "order_by", "random_seed", "metaoperation_id"],
    columns="post_mtype",
    values="prop",
).fillna(0)
print(X_df.shape)
# %%
all_target_stats


# %%
X = X_df.values

hue = X_df.index.get_level_values("root_id").map(ctype_map).astype(str)
hue.name = "C-Type"

# %%
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)

# TODO melt this back to the same format as all_target_stats and join with that

# %%
X_pca_df = pd.DataFrame(
    X_pca, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"], index=X_df.index
).reset_index(drop=False)
X_pca_df
# %%
all_infos = all_infos.reset_index(drop=False)
all_infos = all_infos.set_index(
    ["root_id", "order_by", "random_seed", "metaoperation_id"]
)
all_infos
# %%
X_pca_df = X_pca_df.set_index(
    ["root_id", "order_by", "random_seed", "metaoperation_id"]
)
# %%
X_pca_df = X_pca_df.join(all_infos)
# %%
X_pca_df = X_pca_df.reset_index(drop=False)
# %%
X_pca_df["ctype"] = X_pca_df["root_id"].map(ctype_map).astype(str)

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.scatterplot(
    data=X_pca_df,
    x="PC1",
    y="PC2",
    hue="ctype",
    s=1,
    linewidth=0,
    alpha=0.3,
    legend=False,
)

# %%

kmeans = KMeans(n_clusters=3).fit(X_pca)

# %%
centers = kmeans.cluster_centers_

# %%
native_centers = pca.inverse_transform(centers)

native_centers_df = pd.DataFrame(
    native_centers, columns=X_df.columns, index=[f"Cluster {i}" for i in range(3)]
)
sns.heatmap(native_centers_df, cmap="RdBu_r", center=0, annot=False)

# %%
X_pca_df["root_id_str"] = X_pca_df["root_id"].astype(str)
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

x = 1
y = 2
sns.scatterplot(
    data=X_pca_df,
    x=f"PC{x}",
    y=f"PC{y}",
    s=1,
    linewidth=0,
    alpha=0.3,
    legend=False,
    color="lightgrey",
)

sns.scatterplot(
    data=X_pca_df.query("ctype == '1'"),
    x=f"PC{x}",
    y=f"PC{y}",
    s=2,
    hue="root_id_str",
    legend=False,
    ax=ax,
)

centers_df = pd.DataFrame(
    centers,
    columns=[f"PC{i}" for i in range(1, 7)],
)
for i, row in centers_df.iterrows():
    ax.text(row[f"PC{x}"], row[f"PC{y}"], i, fontsize=12, ha="center", va="bottom")
    ax.scatter(row[f"PC{x}"], row[f"PC{y}"], s=100, color="black", marker="*")

# %%
X_pca_df["instance"] = (
    X_pca_df["root_id_str"]
    + "-"
    + X_pca_df["order_by"]
    + "-"
    + X_pca_df["random_seed"].astype(str)
)
# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

group = 15
x = 1
y = 2
sns.scatterplot(
    data=X_pca_df,
    x=f"PC{x}",
    y=f"PC{y}",
    s=1,
    linewidth=0,
    alpha=0.3,
    legend=False,
    color="lightgrey",
)


sns.lineplot(
    data=X_pca_df.query(f"ctype=='{group}'"),
    x=f"PC{x}",
    y=f"PC{y}",
    hue="root_id_str",
    legend=False,
    units="instance",
    estimator=None,
    linewidth=0.5,
)

lasts = X_pca_df.query(f"ctype=='{group}'")
last_idxs = lasts.groupby(["root_id_str"])["cumulative_n_operations"].idxmax()
last_idxs
sns.scatterplot(
    data=X_pca_df.loc[last_idxs],
    x=f"PC{x}",
    y=f"PC{y}",
    hue="root_id_str",
    legend=False,
    s=50,
    marker="^",
    linewidth=1,
    edgecolor="black",
    zorder=2,
)

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.scatterplot(
    data=X_pca_df,
    x=f"PC{x}",
    y=f"PC{y}",
    hue="cumulative_n_operations",
    s=1,
    linewidth=0,
    alpha=0.3,
    legend=True,
    palette="RdBu_r",
)
