# %%
import os
import time

import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from networkframe import NetworkFrame
from tqdm.auto import tqdm

from pkg.edits import (
    collate_edit_info,
    get_initial_network,
    get_operation_metaoperation_map,
    lazy_load_network_edits,
    pseudo_apply_edit,
)
from pkg.morphology import apply_nucleus, apply_synapses, find_component_by_l2_id
from pkg.plot import savefig
from pkg.utils import get_positions

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
query_neurons.sort_values("id", inplace=True)


# %%

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

# %%

# 10 looked "unstable"
# 11 looked "stable"

root_id = query_neurons["pt_root_id"].values[13]

nuc_row = client.materialize.query_table(
    "nucleus_detection_v0", filter_equal_dict={"pt_root_id": root_id}
)

# %%
(
    networkdeltas_by_operation,
    networkdeltas_by_metaoperation,
) = lazy_load_network_edits(root_id, client=client)

# %%

operation_to_metaoperation = get_operation_metaoperation_map(
    networkdeltas_by_metaoperation
)

# %%

edit_stats, modified_level2_nodes = collate_edit_info(
    networkdeltas_by_operation, operation_to_metaoperation, root_id, client
)

# %%

edit_stats["datetime"] = pd.to_datetime(edit_stats["timestamp"], format="ISO8601")
edit_stats["time"] = edit_stats["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")


# %%

metaoperation_stats = edit_stats.groupby("metaoperation_id").agg(
    {
        "centroid_x": "mean",
        "centroid_y": "mean",
        "centroid_z": "mean",
        "centroid_distance_to_nuc_um": "min",
        "datetime": "max",  # using the latest edit in a bunch as the time
    }
)

metaoperation_stats["time"] = metaoperation_stats["datetime"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)


# %%

initial_nf = get_initial_network(root_id, client, positions=False)

# %%

nf = initial_nf.copy()

for col in [
    "operation_added",
    "operation_removed",
    "metaoperation_added",
    "metaoperation_removed",
]:
    nf.nodes[col] = -1
    nf.nodes[col] = nf.nodes[col].astype(int)
    nf.edges[col] = -1
    nf.edges[col] = nf.edges[col].astype(int)

nf.edges.set_index(["source", "target"], inplace=True, drop=False)

# %%

# go through all of the edits/metaedits
# add nodes that were added, but don't remove any nodes
# mark nodes/edges with when they were added/removed
# things that were never removed/added get -1
for networkdelta in tqdm(networkdeltas_by_operation.values()):
    # this step is necessary to match the indexing set above
    networkdelta.added_edges = networkdelta.added_edges.set_index(
        ["source", "target"], drop=False
    )
    networkdelta.removed_edges = networkdelta.removed_edges.set_index(
        ["source", "target"], drop=False
    )

    # TODO what happens here for multiple operations in the same metaoperation?
    # since they by definition are touching some of the same nodes?
    operation_id = networkdelta.metadata["operation_id"]
    pseudo_apply_edit(
        nf,
        networkdelta,
        operation_label=operation_id,
        metaoperation_label=operation_to_metaoperation[operation_id],
    )

# TODO is it worth just caching the whole networkframe at this stage?

# %%
nuc_supervoxel = nuc_row["pt_supervoxel_id"].values[0]
current_nuc_level2 = client.chunkedgraph.get_roots([nuc_supervoxel], stop_layer=2)[0]
current_nuc_level2 in nf.nodes.index

# %%

# apply positions to the final node-set
node_positions = get_positions(list(nf.nodes.index), client)
nf.nodes[["rep_coord_nm", "x", "y", "z"]] = node_positions[
    ["rep_coord_nm", "x", "y", "z"]
]


# %%

# give the edges info about when those nodes were added
nf.apply_node_features("operation_added", inplace=True)
nf.apply_node_features("metaoperation_added", inplace=True)

nf.edges["was_removed"] = nf.edges["operation_removed"] != -1
nf.nodes["was_removed"] = nf.nodes["operation_removed"] != -1

# %%

pre_synapses, post_synapses = apply_synapses(
    nf,
    networkdeltas_by_operation,
    root_id,
    client,
)

apply_nucleus(nf, root_id, client)

# %%

# generate edit selections, ordered by distance

metaoperation_stats.sort_values("centroid_distance_to_nuc_um", inplace=True)
metaoperation_stats

edit_selections = {}
for i in range(len(metaoperation_stats) + 1):
    edit_selections[i] = metaoperation_stats.index[:i].tolist()


# %%


def resolve_synapses_from_edit_order(
    nf: NetworkFrame,
    edit_selections: dict,
    root_id: int,
    client: cc.CAVEclient,
    prefix="meta",
):
    """
    Assumes that several steps have been run prior to this

    - operation_added has been set on nodes and edges
    """

    resolved_pre_synapses = {}
    resolved_post_synapses = {}

    nuc = client.materialize.query_table(
        "nucleus_detection_v0",
        filter_equal_dict={"pt_root_id": root_id},
        select_columns=["pt_supervoxel_id", "pt_root_id", "pt_position"],
    ).set_index("pt_root_id")
    nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]
    current_nuc_level2 = client.chunkedgraph.get_roots([nuc_supervoxel], stop_layer=2)[
        0
    ]

    choice_time = 0
    query_time = 0
    find_component_time = 0
    record_time = 0
    first = True
    for selection_name, edit_selection in tqdm(edit_selections.items()):
        if not isinstance(edit_selection, list):
            edit_selection = list(edit_selection)

        # -1 for the initial state of the neuron
        if -1 not in edit_selection:
            edit_selection.append(-1)

        if first:
            print("Querying nodes...")
            print("Edit selection: ", edit_selection)
        t = time.time()
        sub_nf = nf.query_nodes(
            f"{prefix}operation_added.isin(@edit_selection)",
            local_dict=locals(),
        ).query_edges(
            f"{prefix}operation_added.isin(@edit_selection)",
            local_dict=locals(),
        )
        query_time += time.time() - t

        t = time.time()
        if first:
            print("Finding component...")
        # this takes up 90% of the time
        # i think it's from the operation of cycling through connected components
        instance_neuron_nf = find_component_by_l2_id(sub_nf, current_nuc_level2)

        if instance_neuron_nf is None:
            print("Missing nucleus component, assuming no synapses...")
            # this can happen if the lack of edits means the nucleus is no longer
            # connected
            found_pre_synapses = []
            found_post_synapses = []
        else:
            found_pre_synapses = []
            for synapses in instance_neuron_nf.nodes["pre_synapses"]:
                found_pre_synapses.extend(synapses)

            found_post_synapses = []
            for synapses in instance_neuron_nf.nodes["post_synapses"]:
                found_post_synapses.extend(synapses)

        find_component_time += time.time() - t

        t = time.time()

        resolved_pre_synapses[selection_name] = found_pre_synapses
        resolved_post_synapses[selection_name] = found_post_synapses

        record_time += time.time() - t

        first = False

    total_time = choice_time + query_time + find_component_time + record_time
    print(f"Total time: {total_time}")
    print(f"Choice time: {choice_time / total_time}")
    print(f"Query time: {query_time / total_time}")
    print(f"Find component time: {find_component_time / total_time}")
    print(f"Record time: {record_time / total_time}")

    return resolved_pre_synapses, resolved_post_synapses


resolved_pre_synapses, resolved_post_synapses = resolve_synapses_from_edit_order(
    nf, edit_selections, root_id, client
)

# %%
mtypes = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")
root_id_counts = mtypes["pt_root_id"].value_counts()
root_id_singles = root_id_counts[root_id_counts == 1].index
mtypes = mtypes.query("pt_root_id in @root_id_singles")
mtypes.set_index("pt_root_id", inplace=True)

mtypes

# %%
pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(mtypes["cell_type"])

post_synapses["pre_mtype"] = post_synapses["pre_pt_root_id"].map(mtypes["cell_type"])

# %%


def count_synapses_by_sample(
    synapses: pd.DataFrame, resolved_pre_synapses: dict, by: str
) -> pd.DataFrame:
    """
    Count number of synapses belonging to some group (`by`) for each sample.

    Parameters
    ----------
    synapses : pd.DataFrame
        Synapses table.
    resolved_pre_synapses : dict
        Dictionary of resolved synapses by edit selection. Keys are edit selection
        identifiers, values are lists of synapse ids.
    by : str
        Column name to group by. Synapses will be grouped by this column, within each
        sample.
    """
    counts_by_sample = []
    for i, key in enumerate(resolved_pre_synapses.keys()):
        sample_resolved_synapses = resolved_pre_synapses[key]

        sample_counts = synapses.loc[sample_resolved_synapses].groupby(by).size()
        sample_counts.name = i
        counts_by_sample.append(sample_counts)

    count = pd.concat(counts_by_sample, axis=1).fillna(0).astype(int).T
    count.index.name = "sample"
    count
    return count


post_mtype_counts = count_synapses_by_sample(
    pre_synapses, resolved_pre_synapses, "post_mtype"
)
# %%
post_mtype_stats_tidy = post_mtype_counts.reset_index().melt(
    var_name="post_mtype", value_name="count", id_vars="sample"
)
post_mtype_stats_tidy["metaoperation"] = (
    (post_mtype_stats_tidy["sample"] - 1)
    .map(metaoperation_stats.index.to_series().reset_index(drop=True))
    .fillna(-1)
    .astype(int)
)
post_mtype_stats_tidy["distance_to_nuc_um"] = (
    post_mtype_stats_tidy["metaoperation"]
    .map(metaoperation_stats["centroid_distance_to_nuc_um"])
    .fillna(0)
)
post_mtype_stats_tidy
# %%

fig, ax = plt.subplots(figsize=(6, 6))
sns.set_context("talk")
sns.lineplot(
    data=post_mtype_stats_tidy,
    x="sample",
    y="count",
    hue="post_mtype",
    ax=ax,
    legend=False,
)
savefig(
    f"post_mtype_counts_vs_sample-root_id={root_id}", fig, folder="edit_replay_ordering"
)

fig, ax = plt.subplots(figsize=(6, 6))
sns.set_context("talk")
sns.lineplot(
    data=post_mtype_stats_tidy,
    x="distance_to_nuc_um",
    y="count",
    hue="post_mtype",
    ax=ax,
    legend=False,
    linewidth=1,
)
sns.scatterplot(
    data=post_mtype_stats_tidy,
    x="distance_to_nuc_um",
    y="count",
    hue="post_mtype",
    ax=ax,
    legend=False,
    s=10,
)
savefig(
    f"post_mtype_counts_vs_sample-root_id={root_id}", fig, folder="edit_replay_ordering"
)


# %%
post_mtype_probs = post_mtype_counts / post_mtype_counts.sum(axis=1).values[:, None]
post_mtype_probs_tidy = post_mtype_probs.reset_index().melt(
    var_name="post_mtype", value_name="prob", id_vars="sample"
)
post_mtype_probs_tidy["metaoperation"] = (
    (post_mtype_probs_tidy["sample"] - 1)
    .map(metaoperation_stats.index.to_series().reset_index(drop=True))
    .fillna(-1)
    .astype(int)
)
post_mtype_probs_tidy["distance_to_nuc_um"] = (
    post_mtype_probs_tidy["metaoperation"]
    .map(metaoperation_stats["centroid_distance_to_nuc_um"])
    .fillna(0)
)
post_mtype_probs_tidy

fig, ax = plt.subplots(figsize=(6, 6))
sns.set_context("talk")
sns.lineplot(
    data=post_mtype_probs_tidy,
    x="sample",
    y="prob",
    hue="post_mtype",
    ax=ax,
    legend=False,
)

savefig(
    f"post_mtype_probs_vs_sample-root_id={root_id}", fig, folder="edit_replay_ordering"
)


fig, ax = plt.subplots(figsize=(6, 6))
sns.set_context("talk")
sns.lineplot(
    data=post_mtype_probs_tidy,
    x="distance_to_nuc_um",
    y="prob",
    hue="post_mtype",
    ax=ax,
    legend=False,
    linewidth=1,
)
sns.scatterplot(
    data=post_mtype_probs_tidy,
    x="distance_to_nuc_um",
    y="prob",
    hue="post_mtype",
    ax=ax,
    legend=False,
    s=10,
)
savefig(
    f"post_mtype_prob_vs_sample-root_id={root_id}", fig, folder="edit_replay_ordering"
)


# %%
metaoperation_stats = metaoperation_stats.sort_values("time")

# %%
edit_selections = {}
for i in range(len(metaoperation_stats) + 1):
    edit_selections[i] = metaoperation_stats.index[:i].tolist()

# %%
resolved_pre_synapses, resolved_post_synapses = resolve_synapses_from_edit_order(
    nf, edit_selections, root_id, client
)

# %%
post_mtype_counts = count_synapses_by_sample(
    pre_synapses, resolved_pre_synapses, "post_mtype"
)

post_mtype_stats_tidy = post_mtype_counts.reset_index().melt(
    var_name="post_mtype", value_name="count", id_vars="sample"
)

post_mtype_probs = post_mtype_counts / post_mtype_counts.sum(axis=1).values[:, None]
post_mtype_probs.fillna(0, inplace=True)
post_mtype_probs_tidy = post_mtype_probs.reset_index().melt(
    var_name="post_mtype", value_name="prob", id_vars="sample"
)

post_mtype_stats_tidy["metaoperation_added"] = (
    (post_mtype_stats_tidy["sample"] - 1)
    .map(metaoperation_stats.index.to_series().reset_index(drop=True))
    .fillna(-1)
    .astype(int)
)

post_mtype_stats_tidy["prob"] = post_mtype_probs_tidy["prob"]

# %%
operation_feature_key = "time"
operation_key = "metaoperation_added"
fillna = metaoperation_stats[operation_feature_key].min()

post_mtype_stats_tidy["time"] = (
    post_mtype_stats_tidy[operation_key]
    .map(metaoperation_stats[operation_feature_key])
    .fillna(fillna)
)

if "time" in operation_feature_key:
    post_mtype_stats_tidy[operation_feature_key] = pd.to_datetime(
        post_mtype_stats_tidy[operation_feature_key]
    )


# %%
sns.set_context("talk")

name_map = {
    "count": "Synapse count",
    "prob": "Output proportion",
    "time": "Time",
}


def apply_name(name):
    if name in name_map:
        return name_map[name]
    else:
        return name


def editplot(stats, x, y, hue="post_mtype", figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=stats,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        legend=False,
        linewidth=1,
    )
    sns.scatterplot(
        data=stats,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        legend=False,
        s=10,
    )

    ax.set_xlabel(apply_name(ax.get_xlabel()))
    ax.set_ylabel(apply_name(ax.get_ylabel()))

    return fig, ax


def rotate_set_ticks(ax):
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")


root_id_time_map = {
    864691135995711402: (pd.to_datetime("2023-04-17"), pd.to_datetime("2023-04-27"))
}

if root_id in root_id_time_map:
    spans = [None, root_id_time_map[root_id]]
else:
    spans = [None]

x = "time"
hue = "post_mtype"

for y in ["count", "prob"]:
    for span in spans:
        name = f"{y}_vs_{x}_by_{hue}-root_id={root_id}"
        fig, ax = editplot(post_mtype_stats_tidy, x, y, hue=hue)
        if span is not None:
            ax.set_xlim(*span)
            name += "-span"
        rotate_set_ticks(ax)
        savefig(name, fig, folder="edit_replay_ordering")
