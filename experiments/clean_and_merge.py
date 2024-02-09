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

from pkg.edits import count_synapses_by_sample
from pkg.neuronframe import (
    NeuronFrameSequence,
    load_neuronframe,
    verify_neuron_matches_final,
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

root_id = query_neurons["pt_root_id"].values[13]

full_neuron = load_neuronframe(root_id, client)

# %%

# simple time-ordered case
neuron_sequence = NeuronFrameSequence(
    full_neuron, prefix="", edit_label_name="operation_id"
)
edits = neuron_sequence.edits.sort_values("time")

for i in tqdm(range(len(edits))):
    operation_id = neuron_sequence.edits.index[i]
    neuron_sequence.apply_edits(operation_id)

#%%
neuron_sequence.is_completed

# %%

path = str(FIG_PATH / "animations" / f"all_edits_by_time-root_id={root_id}.gif")

animate_neuron_edit_sequence(
    path, neuron_sequence.resolved_sequence, n_rotation_steps=5
)

# %%


def select_next_operation(full_neuron, current_neuron, applied_op_ids, possible_op_ids):
    # select the next operation to apply
    # this looks at all of the edges that are connected to the current neuron
    # and then finds the set of operations that are "touched" by this one
    # then it selects the first one of those that hasn't been applied yet, in time
    # order
    out_edges = full_neuron.edges.query(
        "source.isin(@current_neuron.nodes.index) | target.isin(@current_neuron.nodes.index)"
    )
    out_edges = out_edges.drop(current_neuron.edges.index)

    candidate_operations = out_edges[operation_key].unique()

    # TODO this is hard coded
    ordered_ops = possible_op_ids[possible_op_ids.isin(candidate_operations)]

    # HACK?
    # TODO should this be applied merges, or applied ops?
    ordered_ops = ordered_ops[~ordered_ops.isin(applied_merges)]

    if len(ordered_ops) == 0:
        return False
    else:
        applied_op_ids.append(ordered_ops[0])
        applied_merges.append(ordered_ops[0])
        return True


# from the current available operations, apply all splits,
# then apply the soonest merge
# then see if we can apply any more splits (recurse here)
neuron_sequence = NeuronFrameSequence(
    full_neuron, prefix="", edit_label_name="operation_id"
)
neuron_sequence.edits = neuron_sequence.edits.sort_values(["is_merge", "time"])

edits = neuron_sequence.edits

added_key = f"{prefix}operation_added"
removed_key = f"{prefix}operation_removed"

include_added = True
include_removed = False

i = 0
next_operation = True
while next_operation is not None:
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
    # )
    n_possible_internal = len(possible_edit_ids)
    # if n_internal != n_possible_internal:
    #     print(f"WARNING: {n_internal - n_possible_internal} internal edits are missing")

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
        # print(f"Possible operations to apply: {possible_edit_ids}")
        # print(f"Applying operation: {possible_edit_ids[0]}")
        # print(f"Operation is merge: {edits.loc[possible_edit_ids[0], 'is_merge']}")
        next_operation = possible_edit_ids[0]
        neuron_sequence.apply_edits(next_operation, label=i)

    i += 1

# %%
neuron_sequence.final_neuron

# %%
neuron_sequence.current_resolved_neuron

# %%

path = str(FIG_PATH / "animations" / f"all_edits_by_access-root_id={root_id}.gif")

animate_neuron_edit_sequence(
    path, neuron_sequence.resolved_sequence, n_rotation_steps=5, setback=-3_000_000
)

# neuron_sequence.current_resolved_neuron.plot_pyvista()
# %%
# if n_possible_internal == 0:
# now find the next merge to apply, basically

# # TODO this is hard coded
# ordered_ops = possible_op_ids[possible_op_ids.isin(candidate_operations)]

# # HACK?
# # TODO should this be applied merges, or applied ops?
# ordered_ops = ordered_ops[~ordered_ops.isin(applied_merges)]

# %%
plotter = pv.Plotter()

skeleton = full_neuron.to_skeleton_polydata()
plotter.add_mesh(skeleton, color="black", line_width=1)
merge_poly = full_neuron.to_merge_polydata()
if len(merge_poly.points) > 0:
    plotter.add_mesh(merge_poly, color="purple", line_width=5)

plotter.show()

# %%
current_neuron.to_edit_polydata()

# %%

if prefix == "meta":
    edits = full_neuron.metaedits
else:
    edits = full_neuron.edits
    edits["has_split"] = ~edits["is_merge"]
    edits["has_merge"] = edits["is_merge"]

edits = edits.sort_values("time")
# edits["any_merge"] = edits["is_merges"].apply(any)

split_metaedits = edits.query("has_split")
merge_metaedits = edits.query("has_merge")
merge_op_ids = merge_metaedits.index
split_op_ids = split_metaedits.index
applied_op_ids = list(split_op_ids)

# edge case where the neuron's ultimate soma location is itself a merge node
operation_key = f"{prefix}operation_added"
if full_neuron.nodes.loc[full_neuron.nucleus_id, operation_key] != -1:
    applied_op_ids.append(full_neuron.nodes.loc[full_neuron.nucleus_id, operation_key])


neurons = {}
resolved_synapses = {}
applied_merges = []


for i in tqdm(
    range(len(merge_op_ids) + 1), desc="Applying edits and resolving synapses..."
):
    # TODO consider doing this in a way such that I can keep track of how many different
    # split edits are applied with each merge
    # i think this would just mean recursively adding the split edits if available, and
    # then keeping track of merges when they pop up.
    current_neuron = apply_operations(
        full_neuron,
        applied_op_ids,
        resolved_synapses,
        neurons,
        operation_key,
        i,
    )

    # TODO write this in a way where this part can be swapped in and out
    more_operations = select_next_operation(
        full_neuron, current_neuron, applied_op_ids, merge_op_ids
    )
    if not more_operations:
        break

# print(pl.camera_position)


print(f"No remaining merges, stopping ({i / len(merge_op_ids):.2f})")

resolved_synapses = pd.DataFrame(resolved_synapses).T

if completes_neuron:
    verify_neuron_matches_final(full_neuron, current_neuron)

# %%
path = str(FIG_PATH / "animations" / f"edits-root_id={root_id}.gif")

from pkg.plot import animate_neuron_edit_sequence

animate_neuron_edit_sequence(path, neurons)


# %%
def compute_synapse_metrics(
    full_neuron,
    edits,
    resolved_synapses,
    operation_key,
):
    mtypes = load_mtypes(client)

    pre_synapses = full_neuron.pre_synapses
    # map post synapses to their mtypes
    pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(
        mtypes["cell_type"]
    )

    # post_synapses = full_neuron.post_synapses
    # post_synapses["pre_mtype"] = post_synapses["pre_pt_root_id"].map(
    #     mtypes["cell_type"]
    # )

    # find the synapses per sample
    resolved_pre_synapses = resolved_synapses["resolved_pre_synapses"]
    post_mtype_counts = count_synapses_by_sample(
        pre_synapses, resolved_pre_synapses, "post_mtype"
    )

    # wrangle counts and probs
    counts_table = post_mtype_counts
    var_name = "post_mtype"
    post_mtype_stats_tidy = counts_table.reset_index().melt(
        var_name=var_name, value_name="count", id_vars="sample"
    )
    post_mtype_probs = counts_table / counts_table.sum(axis=1).values[:, None]
    post_mtype_probs.fillna(0, inplace=True)
    post_mtype_probs_tidy = post_mtype_probs.reset_index().melt(
        var_name=var_name, value_name="prob", id_vars="sample"
    )
    post_mtype_stats_tidy["prob"] = post_mtype_probs_tidy["prob"]
    post_mtype_stats_tidy[operation_key] = post_mtype_stats_tidy["sample"].map(
        resolved_synapses[operation_key]
    )
    post_mtype_stats_tidy = post_mtype_stats_tidy.join(edits, on=operation_key)

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

    return post_mtype_stats_tidy, sample_wise_metrics


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
