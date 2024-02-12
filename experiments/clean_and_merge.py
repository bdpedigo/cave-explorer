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
from pkg.utils import find_closest_point, load_casey_palette, load_mtypes

pv.set_jupyter_backend("client")

# %%


def apply_operations(
    full_neuron,
    applied_op_ids,
    resolved_synapses,
    neuron_list,
    operation_key,
    iteration_key,
):
    current_neuron = full_neuron.set_edits(applied_op_ids, inplace=False, prefix=prefix)

    if full_neuron.nucleus_id in current_neuron.nodes.index:
        current_neuron.select_nucleus_component(inplace=True)
    else:
        print("WARNING: Using closest point to nucleus...")
        point_id = find_closest_point(
            current_neuron.nodes,
            full_neuron.nodes.loc[full_neuron.nucleus_id, ["x", "y", "z"]],
        )
        current_neuron.select_component_from_node(
            point_id, inplace=True, directed=False
        )

    current_neuron.remove_unused_synapses(inplace=True)

    neuron_list[iteration_key] = current_neuron
    resolved_synapses[iteration_key] = {
        "resolved_pre_synapses": current_neuron.pre_synapses.index.to_list(),
        "resolved_post_synapses": current_neuron.post_synapses.index.to_list(),
        operation_key: applied_op_ids[-1] if i > 0 else None,
    }

    return current_neuron


client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
query_neurons.sort_values("id", inplace=True)

prefix = ""
path = OUT_PATH / "access_time_ordered"
completes_neuron = False

ctype_hues = load_casey_palette()

root_id = query_neurons["pt_root_id"].values[14]

full_neuron = load_neuronframe(root_id, client)

# TODO put this in the lazy loader when running the full thing
full_neuron.apply_edge_lengths(inplace=True)

mtypes = load_mtypes(client)

pre_synapses = full_neuron.pre_synapses
# map post synapses to their mtypes
pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(mtypes["cell_type"])


# %%

# simple time-ordered case
neuron_sequence = NeuronFrameSequence(
    full_neuron, prefix="", edit_label_name="operation_id"
)
edits = neuron_sequence.edits.sort_values("time")

for i in tqdm(range(len(edits))):
    operation_id = neuron_sequence.edits.index[i]
    neuron_sequence.apply_edits(operation_id)

# %%
neuron_sequence.edits

# %%
neuron_sequence.is_completed

# %%

path = str(FIG_PATH / "animations" / f"all_edits_by_time-root_id={root_id}.gif")

animate_neuron_edit_sequence(
    path, neuron_sequence.resolved_sequence, n_rotation_steps=5
)

# %%


# from the current available operations, apply all splits,
# then apply the soonest merge
# then see if we can apply any more splits (recurse here)
prefix = "meta"
neuron_sequence = NeuronFrameSequence(
    full_neuron, prefix=prefix, edit_label_name="metaoperation_id"
)

neuron_sequence.edits

# %%

edits = neuron_sequence.edits.sort_values(["has_merge", "time"])

added_key = f"{prefix}operation_added"
removed_key = f"{prefix}operation_removed"

include_added = True
include_removed = False

i = 0
next_operation = True

pbar = tqdm(total=len(edits), desc="Applying edits...")
while next_operation is not None:
    # TODO make much of the below a few steps coming from methods in the
    # NeuronFrameSequence
    current_neuron = neuron_sequence.current_resolved_neuron
    full_neuron = neuron_sequence.base_neuron
    applied_edit_ids = neuron_sequence.applied_edit_ids

    # find operations that are internal to the current neuron
    internal_edges = current_neuron.edges
    internal_edit_ids = set()
    if include_added:
        added_edit_ids = set(internal_edges[added_key].unique()) - {-1}
        internal_edit_ids = internal_edit_ids | added_edit_ids
    if include_removed:
        removed_edit_ids = set(internal_edges[removed_key].unique()) - {-1}
        internal_edit_ids = internal_edit_ids | removed_edit_ids
    n_internal = len(internal_edit_ids)

    possible_edit_ids = neuron_sequence.unapplied_edits.index.intersection(
        internal_edit_ids, sort=False
    )
    n_possible_internal = len(possible_edit_ids)

    # if no internal operations, find an external one
    if n_possible_internal == 0:
        out_edges = full_neuron.edges.query(
            "source.isin(@current_neuron.nodes.index) | target.isin(@current_neuron.nodes.index)"
        )
        out_edges = out_edges.drop(current_neuron.edges.index)

        possible_edit_ids = out_edges[f"{prefix}operation_added"].unique()

        possible_edit_ids = edits.index[edits.index.isin(possible_edit_ids)]
        possible_edit_ids = possible_edit_ids[~possible_edit_ids.isin(applied_edit_ids)]

    if len(possible_edit_ids) == 0:
        print("No possible operations to apply")
        next_operation = None
    else:
        next_operation = possible_edit_ids[0]
        neuron_sequence.apply_edits(next_operation)

    i += 1
    pbar.update(1)

pbar.close()


# %%
if neuron_sequence.is_completed:
    print("Neuron is completed")
else:
    print("Neuron is not completed")
    current = neuron_sequence.current_resolved_neuron
    final = neuron_sequence.final_neuron
    print(current.node_agreement(final), "current nodes in final.")
    print(final.node_agreement(current), "final nodes in current.")


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
neuron_sequence

# %%
# sns.scatterplot(
#     data=neuron_sequence.sequence_info,
#     hue='has_merge',
#     x="order",
#     y="n_nodes",
#     palette="tab10",
# )

sns.scatterplot(
    data=neuron_sequence.sequence_info,
    hue="has_merge",
    x="order",
    y="path_length",
    palette="tab10",
)

# %%
counts_table = neuron_sequence.synapse_groupby_count(by="post_mtype", which="pre")
counts_table

# %%
edit_label_name = neuron_sequence.edit_label_name
# %%
# wrangle counts and probs

var_name = "post_mtype"
post_mtype_stats_tidy = counts_table.reset_index().melt(
    var_name=var_name, value_name="count", id_vars=edit_label_name
)
post_mtype_stats_tidy

post_mtype_probs = counts_table / counts_table.sum(axis=1).values[:, None]
post_mtype_probs.fillna(0, inplace=True)
post_mtype_probs_tidy = post_mtype_probs.reset_index().melt(
    var_name=var_name, value_name="prob", id_vars=edit_label_name
)
post_mtype_stats_tidy["prob"] = post_mtype_probs_tidy["prob"]

post_mtype_stats_tidy

# post_mtype_stats_tidy[operation_key] = post_mtype_stats_tidy["sample"].map(
#     resolved_synapses[operation_key]
# )


post_mtype_stats_tidy = post_mtype_stats_tidy.join(
    neuron_sequence.sequence_info, on=edit_label_name
)
post_mtype_stats_tidy


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
    x="order",
    y="count",
    hue="post_mtype",
    palette=ctype_hues,
    ax=ax,
    legend=False,
    s=10,
)

for metaoperation_id, row in post_mtype_stats_tidy.iterrows():
    if row["has_merge"]:
        ax.axvline(
            row["order"],
            color="lightgrey",
            linestyle="-",
            alpha=0.5,
            linewidth=1,
            zorder=-1,
        )

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.scatterplot(
    data=post_mtype_stats_tidy,
    x="cumulative_n_operations",
    y="prob",
    hue="post_mtype",
    palette=ctype_hues,
    ax=ax,
    legend=False,
    s=10,
)
sns.scatterplot(
    data=sub_post_mtype_stats,
    x="cumulative_n_operations",
    y="prob",
    hue="post_mtype",
    palette=ctype_hues,
    ax=ax,
    legend=False,
    s=50,
)
sns.lineplot(
    data=sub_post_mtype_stats,
    x="cumulative_n_operations",
    y="prob",
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
    y="prob",
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
