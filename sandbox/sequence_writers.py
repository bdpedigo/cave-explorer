# %%
import os

os.environ["LAZYCLOUD_USE_CLOUD"] = "True"
os.environ["LAZYCLOUD_RECOMPUTE"] = "False"
os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from scipy.spatial.distance import cdist

from pkg.neuronframe import (
    NeuronFrameSequence,
    load_neuronframe,
)
from pkg.paths import FIG_PATH, OUT_PATH
from pkg.plot import animate_neuron_edit_sequence, savefig
from pkg.sequence import create_merge_and_clean_sequence, create_time_ordered_sequence
from pkg.utils import load_casey_palette, load_mtypes

pv.set_jupyter_backend("client")

# %%


client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
query_neurons.sort_values("id", inplace=True)

prefix = ""
path = OUT_PATH / "access_time_ordered"
completes_neuron = False

ctype_hues = load_casey_palette()

root_id = query_neurons["pt_root_id"].values[19]

# root_id = 864691135737446276
full_neuron = load_neuronframe(root_id, client)

# TODO put this in the lazy loader when running the full thing
full_neuron.apply_edge_lengths(inplace=True)

mtypes = load_mtypes(client)

pre_synapses = full_neuron.pre_synapses
pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(mtypes["cell_type"])

setback = 2_000_000
# %%


time_ordered_sequence = create_time_ordered_sequence(full_neuron, root_id=root_id)


# %%
merge_and_clean_sequence = create_merge_and_clean_sequence(
    full_neuron, root_id=root_id, order_by="time"
)
# %%
merge_and_clean_sequence1 = create_merge_and_clean_sequence(
    full_neuron, root_id=root_id, order_by="random", random_seed=8888
)
merge_and_clean_sequence2 = create_merge_and_clean_sequence(
    full_neuron, root_id=root_id, order_by="random", random_seed=8888
)

# %%
df1 = merge_and_clean_sequence1.sequence_info
df2 = merge_and_clean_sequence2.sequence_info

df1.equals(df2)

# %%
path = str(FIG_PATH / "animations" / f"merge_and_clean_by_time-root_id={root_id}.gif")

animate_neuron_edit_sequence(
    path,
    merge_and_clean_sequence.resolved_sequence,
    n_rotation_steps=2,
    setback=setback,
    azimuth_step_size=1,
    fps=20,
)

# %%

rng = np.random.default_rng(8888)
n_trials = 10
for i in range(n_trials):
    seed = rng.integers(np.iinfo(np.uint64).max, dtype=np.uint64)
    rand_merge_and_clean_sequence = create_merge_and_clean_sequence(
        full_neuron, order_by="random", random_seed=seed
    )

# %%
import pickle

with open("test.pickle", "wb") as f:
    pickle.dump(rand_merge_and_clean_sequence.sequence_info, f)


# %%
np.random.default_rng(np.iinfo(np.uint64).max)
# %%
path = str(FIG_PATH / "animations" / f"merge_and_clean_by_random-root_id={root_id}.gif")

animate_neuron_edit_sequence(
    path,
    rand_merge_and_clean_sequence.resolved_sequence,
    n_rotation_steps=2,
    setback=setback,
    azimuth_step_size=1,
    fps=20,
)
# %%
merge_and_clean_sequence.sequence_info

# %%
rand_merge_and_clean_sequence.sequence_info

info = merge_and_clean_sequence.sequence_info
seq = merge_and_clean_sequence
# %%
out_dict = merge_and_clean_sequence.to_dict()
# %%
seq = NeuronFrameSequence.from_dict_and_neuron(out_dict, full_neuron)


# %%
seq.sequence_info

# %%
seq.synapse_groupby_count(by="post_mtype", which="pre")

# %%
post_mtype_stats = seq.synapse_groupby_metrics(by="post_mtype", which="pre")

# %%
bouts = seq.sequence_info["has_merge"].fillna(False).cumsum()
bouts.name = "bout"

# %%
bout_exemplars = (
    seq.sequence_info.index.to_series().groupby(bouts).apply(lambda x: x.iloc[-1])
)

bout_exemplars
bout_info = seq.sequence_info.loc[bout_exemplars.values]

sub_post_mtype_stats = post_mtype_stats.query("metaoperation_id.isin(@bout_exemplars)")


# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.scatterplot(
    data=post_mtype_stats,
    x="cumulative_n_operations",
    y="prop",
    hue="post_mtype",
    palette=ctype_hues,
    ax=ax,
    legend=False,
    s=10,
)
sns.scatterplot(
    data=sub_post_mtype_stats,
    x="cumulative_n_operations",
    y="prop",
    hue="post_mtype",
    palette=ctype_hues,
    ax=ax,
    legend=False,
    s=50,
)
sns.lineplot(
    data=sub_post_mtype_stats,
    x="cumulative_n_operations",
    y="prop",
    hue="post_mtype",
    palette=ctype_hues,
    ax=ax,
    legend=False,
)

for metaoperation_id, row in post_mtype_stats.iterrows():
    if row["has_merge"]:
        ax.axvline(
            row["cumulative_n_operations"],
            color="lightgrey",
            linestyle="-",
            alpha=0.5,
            linewidth=1,
            zorder=-1,
        )


# %%
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
sns.scatterplot(
    data=sub_post_mtype_stats,
    x="cumulative_n_operations",
    y="prop",
    hue="post_mtype",
    palette=ctype_hues,
    ax=ax,
    legend=False,
    s=15,
)
sns.lineplot(
    data=sub_post_mtype_stats,
    x="cumulative_n_operations",
    y="prop",
    hue="post_mtype",
    palette=ctype_hues,
    ax=ax,
    legend=True,
)
handles, labels = ax.get_legend_handles_labels()
ax.get_legend().remove()
ax.legend(
    handles=handles[:],
    labels=labels[:],
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    title="Post-synaptic M-type",
    fontsize="small",
    ncol=2,
)
ax.set(xlabel="Cumulative # of operations", ylabel="Proportion of output synapses")

# %%
# get the last row from each bout as a representation
bout_info = neuron_sequence.sequence_info.groupby(bouts).apply(lambda x: x.iloc[-1])
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.scatterplot(
    data=bout_info,
    x="order",
    y="path_length",
    hue="has_merge",
    palette="tab10",
    ax=ax,
    legend=False,
    s=10,
)


# %%
final_probs = post_mtype_probs.iloc[-1]

# euclidean distance
# euc_diffs = (((post_mtype_probs - final_probs) ** 2).sum(axis=1)) ** 0.5

sample_wise_metrics = []
for metric in ["euclidean", "cityblock", "jensenshannon", "cosine"]:
    distances = cdist(
        post_mtype_probs.values, final_probs.values.reshape(1, -1), metric=metric
    )
    distances = pd.Series(
        distances.flatten(), name=metric, index=post_mtype_probs.index
    )
    sample_wise_metrics.append(distances)
sample_wise_metrics = pd.concat(sample_wise_metrics, axis=1)
sample_wise_metrics[operation_key] = sample_wise_metrics.index.map(
    resolved_synapses[operation_key]
)
sample_wise_metrics = sample_wise_metrics.join(edits, on=operation_key)

# TODO might as well also do the same join as the above to the added metaedits


# %%


# %%
post_mtype_stats, sample_wise_metrics = compute_synapse_metrics(
    full_neuron, edits, resolved_synapses, operation_key
)

# %%
metrics = ["euclidean", "cityblock", "jensenshannon", "cosine"]
n_col = len(metrics)

fig, axs = plt.subplots(1, n_col, figsize=(5 * n_col, 5))

for i, metric in enumerate(metrics):
    sns.lineplot(
        data=sample_wise_metrics,
        x="sample",
        y=metric,
        ax=axs[i],
    )
    axs[i].set_xlabel("Metaoperation added")
    axs[i].set_ylabel(f"{metric} distance")
    axs[i].spines[["top", "right"]].set_visible(False)


# %%
save = False

sns.set_context("talk")

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.lineplot(
    data=post_mtype_stats,
    x="sample",
    y="count",
    hue="post_mtype",
    legend=False,
    palette=ctype_hues,
    ax=ax,
)
ax.set_xlabel("Metaoperation added")
ax.set_ylabel("# output synapses")
ax.spines[["top", "right"]].set_visible(False)
if save:
    savefig(
        f"output_synapses_access_time_ordered-root_id={root_id}",
        fig,
        folder="access_time_ordered",
    )

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.lineplot(
    data=post_mtype_stats,
    x="sample",
    y="prop",
    hue="post_mtype",
    legend=False,
    palette=ctype_hues,
    ax=ax,
)
ax.set_xlabel("Metaoperation added")
ax.set_ylabel("Proportion of output synapses")
ax.spines[["top", "right"]].set_visible(False)

if save:
    savefig(
        f"output_proportion_access_time_ordered-root_id={root_id}",
        fig,
        folder="access_time_ordered",
    )


fig, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.lineplot(
    data=post_mtype_stats,
    x="sample",
    y="centroid_distance_to_nuc_um",
    hue="post_mtype",
    legend=False,
    palette=ctype_hues,
    ax=ax,
)
ax.set_xlabel("Metaoperation added")
ax.set_ylabel("Distance to nucleus (nm)")
ax.spines[["top", "right"]].set_visible(False)

if save:
    savefig(
        f"distance_access_time_ordered-root_id={root_id}",
        fig,
        folder="access_time_ordered",
    )

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.lineplot(
    data=diffs,
    x="sample",
    y="diff",
)
ax.set_xlabel("Metaoperation added")
ax.set_ylabel("Distance from final")
ax.spines[["top", "right"]].set_visible(False)

if save:
    savefig(
        f"distance_from_final_access_time_ordered-root_id={root_id}",
        fig,
        folder="access_time_ordered",
    )

if save:
    resolved_synapses.to_csv(path / f"resolved_synapses-root_id={root_id}.csv")

    post_mtype_stats.to_csv(path / f"post_mtype_stats_tidy-root_id={root_id}.csv")

    diffs.to_csv(path / f"diffs-root_id={root_id}.csv")

# %%
