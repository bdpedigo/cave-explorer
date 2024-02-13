# %%
import os

os.environ["LAZYCLOUD_USE_CLOUD"] = "True"
os.environ["LAZYCLOUD_RECOMPUTE"] = "False"
os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv
import seaborn as sns
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from pkg.neuronframe import (
    NeuronFrameSequence,
    load_neuronframe,
)
from pkg.paths import FIG_PATH, OUT_PATH
from pkg.plot import animate_neuron_edit_sequence, savefig
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

root_id = query_neurons["pt_root_id"].values[15]

full_neuron = load_neuronframe(root_id, client)

# TODO put this in the lazy loader when running the full thing
full_neuron.apply_edge_lengths(inplace=True)

mtypes = load_mtypes(client)

pre_synapses = full_neuron.pre_synapses
# map post synapses to their mtypes
pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(mtypes["cell_type"])


# %%

# TODO I think this can be re-written using the same logic as the below; a priority-
# sorted list of operations, and then (optionally) some logic about whether to apply
# them in order or anything like that.

# simple time-ordered case
neuron_sequence = NeuronFrameSequence(
    full_neuron, prefix="", edit_label_name="operation_id"
)
neuron_sequence.edits.sort_values("time", inplace=True)

for i in tqdm(range(len(neuron_sequence.edits))):
    operation_id = neuron_sequence.edits.index[i]
    neuron_sequence.apply_edits(operation_id)

if neuron_sequence.is_completed:
    print("Neuron is completed")
else:
    print("Neuron is not completed")

# %%

path = str(FIG_PATH / "animations" / f"all_edits_by_time-root_id={root_id}.gif")

animate_neuron_edit_sequence(
    path, neuron_sequence.resolved_sequence, n_rotation_steps=5
)

# %%

# this is actually "merge and clean"
# from the current available operations, apply all splits,
# then apply the soonest merge
# then see if we can apply any more splits (recurse here)
prefix = "meta"
neuron_sequence = NeuronFrameSequence(
    full_neuron, prefix=prefix, edit_label_name="metaoperation_id"
)

neuron_sequence.edits.sort_values(["has_merge", "time"], inplace=True)


i = 0
next_operation = True
pbar = tqdm(total=len(neuron_sequence.edits), desc="Applying edits...")
while next_operation is not None:
    possible_edit_ids = neuron_sequence.find_incident_edits()
    if len(possible_edit_ids) == 0:
        print("No more possible operations to apply")
        next_operation = None
    else:
        next_operation = possible_edit_ids[0]
        neuron_sequence.apply_edits(next_operation)
    i += 1
    pbar.update(1)
pbar.close()

if neuron_sequence.is_completed:
    print("Neuron is completed")
else:
    print("Neuron is not completed")

# %%

path = str(
    FIG_PATH
    / "animations"
    / f"all_edits_by_access-prefix={prefix}-root_id={root_id}.gif"
)

animate_neuron_edit_sequence(
    path, neuron_sequence.resolved_sequence, n_rotation_steps=5, setback=-3_000_000
)

# %%


# %%
bouts = neuron_sequence.sequence_info["has_merge"].fillna(False).cumsum()
bouts.name = "bout"

bout_exemplars = (
    neuron_sequence.sequence_info.index.to_series()
    .groupby(bouts)
    .apply(lambda x: x.iloc[-1])
)

bout_exemplars
bout_info = neuron_sequence.sequence_info.loc[bout_exemplars.values]

sub_post_mtype_stats = post_mtype_stats_tidy.query(
    "metaoperation_id.isin(@bout_exemplars)"
)


# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.scatterplot(
    data=post_mtype_stats_tidy,
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

for metaoperation_id, row in post_mtype_stats_tidy.iterrows():
    if row["has_merge"]:
        ax.axvline(
            row["cumulative_n_operations"],
            color="lightgrey",
            linestyle="-",
            alpha=0.5,
            linewidth=1,
            zorder=-1,
        )

plt.show()


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
post_mtype_stats_tidy, sample_wise_metrics = compute_synapse_metrics(
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
    data=post_mtype_stats_tidy,
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
    data=post_mtype_stats_tidy,
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
    data=post_mtype_stats_tidy,
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

    post_mtype_stats_tidy.to_csv(path / f"post_mtype_stats_tidy-root_id={root_id}.csv")

    diffs.to_csv(path / f"diffs-root_id={root_id}.csv")

# %%
