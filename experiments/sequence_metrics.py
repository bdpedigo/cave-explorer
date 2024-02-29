# %%
import os

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"
os.environ["LAZYCLOUD_RECOMPUTE"] = "False"
os.environ["LAZYCLOUD_USE_CLOUD"] = "True"

import pickle
import time

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cloudfiles import CloudFiles
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm

from pkg.neuronframe import NeuronFrame, load_neuronframe
from pkg.paths import OUT_PATH
from pkg.plot import savefig
from pkg.sequence import create_merge_and_clean_sequence, create_time_ordered_sequence
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

files["scheme"] = "historical"
files.loc[files["order_by"].notna(), "scheme"] = "clean-and-merge"

files_finished = files.query("root_id in @has_all")


# %%


BINS = np.linspace(0, 1_000_000, 31)


def annotate_pre_synapses(neuron: NeuronFrame, mtypes: pd.DataFrame) -> None:
    # annotating with classes
    neuron.pre_synapses["post_mtype"] = neuron.pre_synapses["post_pt_root_id"].map(
        mtypes["cell_type"]
    )

    # locations of the post-synaptic soma
    post_locs = (
        neuron.pre_synapses["post_pt_root_id"]
        .map(mtypes["pt_position"])
        .dropna()
        .to_frame(name="post_nuc_loc")
    )
    post_locs["post_nuc_x"] = post_locs["post_nuc_loc"].apply(lambda x: x[0])
    post_locs["post_nuc_y"] = post_locs["post_nuc_loc"].apply(lambda x: x[1])
    post_locs["post_nuc_z"] = post_locs["post_nuc_loc"].apply(lambda x: x[2])
    neuron.pre_synapses = neuron.pre_synapses.join(post_locs)

    # euclidean distance to post-synaptic soma
    nuc_loc = neuron.nodes.loc[neuron.nucleus_id, ["x", "y", "z"]]
    X = neuron.pre_synapses[["post_nuc_x", "post_nuc_y", "post_nuc_z"]].dropna()
    euclidean_distances = pairwise_distances(
        X, nuc_loc.values.reshape(1, -1), metric="euclidean"
    )
    euclidean_distances = pd.Series(
        euclidean_distances.flatten(), index=X.index, name="euclidean"
    )

    # radial (x-z only) distance to post-synaptic soma
    X_radial = neuron.pre_synapses[["post_nuc_x", "post_nuc_z"]].dropna()
    nuc_loc_radial = nuc_loc[["x", "z"]]
    radial_distances = pairwise_distances(
        X_radial, nuc_loc_radial.values.reshape(1, -1), metric="euclidean"
    )
    radial_distances = pd.Series(
        radial_distances.flatten(), index=X_radial.index, name="radial"
    )
    distance_df = pd.concat([euclidean_distances, radial_distances], axis=1)
    neuron.pre_synapses = neuron.pre_synapses.join(distance_df)

    neuron.pre_synapses["radial_to_nuc_bin"] = pd.cut(
        neuron.pre_synapses["radial"], BINS
    )

    return None


def annotate_mtypes(neuron: NeuronFrame, mtypes: pd.DataFrame):
    mtypes["post_mtype"] = mtypes["cell_type"]
    mtypes["x"] = mtypes["pt_position"].apply(lambda x: x[0])
    mtypes["y"] = mtypes["pt_position"].apply(lambda x: x[1])
    mtypes["z"] = mtypes["pt_position"].apply(lambda x: x[2])
    nuc_loc = neuron.nodes.loc[neuron.nucleus_id, ["x", "y", "z"]]
    distance_to_nuc = pairwise_distances(
        mtypes[["x", "y", "z"]], nuc_loc.values.reshape(1, -1), metric="euclidean"
    )
    mtypes["euclidean_to_nuc"] = distance_to_nuc

    nuc_loc = neuron.nodes.loc[neuron.nucleus_id, ["x", "z"]]
    distance_to_nuc = pairwise_distances(
        mtypes[["x", "z"]], nuc_loc.values.reshape(1, -1), metric="euclidean"
    )
    mtypes["radial_to_nuc"] = distance_to_nuc

    mtypes["radial_to_nuc_bin"] = pd.cut(mtypes["radial_to_nuc"], BINS)

    return None


def compute_spatial_target_proportions(synapses_df, mtypes=None, by=None):
    if by is not None:
        spatial_by = ["radial_to_nuc_bin", by]
    else:
        spatial_by = ["radial_to_nuc_bin"]

    cells_hit = synapses_df.groupby(spatial_by)["post_pt_root_id"].nunique()

    cells_available = mtypes.groupby(spatial_by).size()

    p_cells_hit = cells_hit / cells_available

    return p_cells_hit


def compute_target_counts(synapses_df: pd.DataFrame, by=None):
    result = synapses_df.groupby(by).size()
    return result


def compute_target_proportions(synapses_df: pd.DataFrame, by=None):
    result = synapses_df.groupby(by).size()
    result = result / result.sum()
    return result


def apply_metadata(df, key):
    index_name = df.index.name
    df["root_id"] = key[0]
    df["scheme"] = key[1]
    df["order_by"] = key[2]
    df["random_seed"] = key[3]
    df["order"] = np.arange(len(df))
    df.reset_index(drop=False, inplace=True)
    df.set_index(
        ["root_id", "scheme", "order_by", "random_seed", index_name, "order"],
        inplace=True,
    )
    return df


# %%

client = cc.CAVEclient("minnie65_phase3_v1")
mtypes = load_mtypes(client)

if False:
    root_id = 864691134886015738
    neuron = load_neuronframe(root_id, client)
    annotate_pre_synapses(neuron, mtypes)
    annotate_mtypes(neuron, mtypes)
    order_by = "time"
    random_seed = None
    sequence = create_merge_and_clean_sequence(
        neuron, root_id, order_by=order_by, random_seed=random_seed
    )

# %%

total_time = time.time()

load_neuron_time = 0
annotate_time = 0
load_sequence_time = 0
counts_time = 0
props_time = 0
spatial_props_time = 0
spatial_props_by_mtype_time = 0
sequence_time = 0

recompute = True
save = False
if recompute:
    root_ids = files_finished["root_id"].unique()[:10]
    all_infos = []
    all_sequence_features = {}
    pbar = tqdm(total=len(root_ids), desc="Computing target stats...")
    for root_id, rows in files_finished.query("root_id.isin(@root_ids)").groupby(
        "root_id"
    ):
        currtime = time.time()
        neuron = load_neuronframe(root_id, client)
        load_neuron_time += time.time() - currtime

        currtime = time.time()
        annotate_pre_synapses(neuron, mtypes)
        annotate_mtypes(neuron, mtypes)
        annotate_time += time.time() - currtime

        for keys, sub_rows in rows.groupby(
            ["scheme", "order_by", "random_seed"], dropna=False
        ):
            scheme, order_by, random_seed = keys

            currtime = time.time()
            if scheme == "clean-and-merge":
                sequence = create_merge_and_clean_sequence(
                    neuron, root_id, order_by=order_by, random_seed=random_seed
                )
                sequence = sequence.select_by_bout("has_merge", keep="last")
            elif scheme == "historical":
                sequence = create_time_ordered_sequence(neuron, root_id)
            else:
                raise ValueError(f"Scheme {scheme} not recognized.")
            load_sequence_time += time.time() - currtime

            sequence_key = (root_id, scheme, order_by, random_seed)

            currtime = time.time()
            sequence_feature_dfs = {}
            counts_by_mtype = sequence.apply_to_synapses_by_sample(
                compute_target_counts, which="pre", by="post_mtype"
            )
            counts_by_mtype = apply_metadata(counts_by_mtype, sequence_key)
            sequence_feature_dfs["counts_by_mtype"] = counts_by_mtype
            counts_time += time.time() - currtime

            currtime = time.time()
            props_by_mtype = sequence.apply_to_synapses_by_sample(
                compute_target_proportions, which="pre", by="post_mtype"
            )
            props_by_mtype = apply_metadata(props_by_mtype, sequence_key)
            sequence_feature_dfs["props_by_mtype"] = props_by_mtype
            props_time += time.time() - currtime

            currtime = time.time()
            spatial_props = sequence.apply_to_synapses_by_sample(
                compute_spatial_target_proportions, which="pre", mtypes=mtypes
            )
            spatial_props = apply_metadata(spatial_props, sequence_key)
            sequence_feature_dfs["spatial_props"] = spatial_props
            spatial_props_time += time.time() - currtime

            currtime = time.time()
            spatial_props_by_mtype = sequence.apply_to_synapses_by_sample(
                compute_spatial_target_proportions,
                which="pre",
                mtypes=mtypes,
                by="post_mtype",
            )
            spatial_props_by_mtype = apply_metadata(
                spatial_props_by_mtype, sequence_key
            )
            sequence_feature_dfs["spatial_props_by_mtype"] = spatial_props_by_mtype
            spatial_props_by_mtype_time += time.time() - currtime

            all_sequence_features[sequence_key] = sequence_feature_dfs

            currtime = time.time()
            info = sequence.sequence_info
            info["root_id"] = root_id
            info["scheme"] = scheme
            info["order_by"] = order_by
            info["random_seed"] = random_seed
            all_infos.append(
                info.drop(["pre_synapses", "post_synapses", "applied_edits"], axis=1)
            )
            sequence_time += time.time() - currtime

        pbar.update(1)

    pbar.close()

    all_infos = pd.concat(all_infos)

    meta_features_df = pd.DataFrame(all_sequence_features).T
    meta_features_df.index.names = ["root_id", "scheme", "order_by", "random_seed"]

    if save:
        with open(OUT_PATH / "sequence_metrics" / "all_infos.pkl", "wb") as f:
            pickle.dump(all_infos, f)
        with open(OUT_PATH / "sequence_metrics" / "meta_features_df.pkl", "wb") as f:
            pickle.dump(meta_features_df, f)

else:
    with open(OUT_PATH / "sequence_metrics" / "all_infos.pkl", "rb") as f:
        all_infos = pickle.load(f)
    with open(OUT_PATH / "sequence_metrics" / "meta_features_df.pkl", "rb") as f:
        meta_features_df = pickle.load(f)

total_time = time.time() - total_time

print(f"Total time: {total_time:.2f} seconds")
print(f"Load neuron proportion: {load_neuron_time / total_time:.2f}")
print(f"Annotate proportion: {annotate_time / total_time:.2f}")
print(f"Load sequence proportion: {load_sequence_time / total_time:.2f}")
print(f"Counts proportion: {counts_time / total_time:.2f}")
print(f"Props proportion: {props_time / total_time:.2f}")
print(f"Spatial props proportion: {spatial_props_time / total_time:.2f}")
print(
    f"Spatial props by mtype proportion: {spatial_props_by_mtype_time / total_time:.2f}"
)
print(f"Sequence proportion: {sequence_time / total_time:.2f}")


# %%


# TODO whether to implement this as a table of tables, one massive table...
# nothing really feels satisfying here
# perhaps a table of tables will work, and it can infill the index onto those tables
# before doing a join or concat

# TODO key the elements in the sequence on something other than metaoperation_id, this
# will make it easier to join with the time-ordered dataframes which use "operation_id",
# or do things like take "bouts" for computing metrics which are not tied to a specific
# operation_id


# %%
sub_df = pd.concat(
    meta_features_df.query("root_id == @root_id & scheme == 'historical'")[
        "spatial_props"
    ].values
)
cols = sub_df.columns
mids = [
    interval.mid for interval in sub_df.columns.get_level_values("radial_to_nuc_bin")
]

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
colors = sns.color_palette("coolwarm_r", n_colors=sub_df.shape[0])

for i, (operation_id, row) in enumerate(sub_df.iterrows()):
    sns.lineplot(
        y=row.values,
        x=mids,
        ax=ax,
        alpha=0.5,
        linewidth=0.5,
        color=colors[i],
        legend=False,
    )


# %%


def process_sequence_diffs(sequence_df):
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
        df = df.loc[sequence_label]
        df.index = df.index.droplevel(0)
        diffs = process_sequence_diffs(df)
        sequence_diffs[metric_label] = diffs
    all_sequence_diffs[sequence_label] = sequence_diffs

meta_diff_df = pd.DataFrame(all_sequence_diffs).T
        # meta_diff_df.loc[sequence_label, metric_label] = diffs

meta_diff_df
# %%


# %%
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

query_neurons = client.materialize.query_table("connectivity_groups_v795")
ctype_map = query_neurons.set_index("pt_root_id")["cell_type"]

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
