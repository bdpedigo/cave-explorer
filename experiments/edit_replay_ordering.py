# %%
import os

import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pkg.edits import (
    apply_edit_history,
    apply_synapses,
    collate_edit_info,
    count_synapses_by_sample,
    get_initial_network,
    get_operation_metaoperation_map,
    lazy_load_network_edits,
    resolve_synapses_from_edit_selections,
)
from pkg.morphology import (
    apply_nucleus,
    apply_positions,
)
from pkg.plot import rotate_set_labels, savefig

# %%

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"


# %%

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
query_neurons.sort_values("id", inplace=True)

# 10 looked "unstable"
# 11 looked "stable"
root_id = query_neurons["pt_root_id"].values[18]

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
edit_stats, metaoperation_stats, modified_level2_nodes = collate_edit_info(
    networkdeltas_by_operation, operation_to_metaoperation, root_id, client
)


# %%
initial_nf = get_initial_network(root_id, client, positions=False)

# %%


# go through all of the edits/metaedits
# add nodes that were added, but don't remove any nodes
# mark nodes/edges with when they were added/removed
# things that were never removed/added get -1


nf = initial_nf.copy()

apply_edit_history(nf, networkdeltas_by_operation, operation_to_metaoperation)

# TODO is it worth just caching the whole networkframe at this stage?


# %%

apply_positions(nf, client)


# %%

pre_synapses, post_synapses = apply_synapses(
    nf,
    networkdeltas_by_operation,
    root_id,
    client,
)

apply_nucleus(nf, root_id, client)

# %%
mtypes = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")
root_id_counts = mtypes["pt_root_id"].value_counts()
root_id_singles = root_id_counts[root_id_counts == 1].index
mtypes = mtypes.query("pt_root_id in @root_id_singles")
mtypes.set_index("pt_root_id", inplace=True)

# %%
pre_synapses["post_mtype"] = pre_synapses["post_pt_root_id"].map(mtypes["cell_type"])
post_synapses["pre_mtype"] = post_synapses["pre_pt_root_id"].map(mtypes["cell_type"])

# %%
metaoperation_stats = metaoperation_stats.sort_values("time")

# %%
edit_selections = {}
for i in range(len(metaoperation_stats) + 1):
    edit_selections[i] = metaoperation_stats.index[:i].tolist()

# %%
resolved_pre_synapses, resolved_post_synapses = resolve_synapses_from_edit_selections(
    nf, edit_selections, root_id, client
)

# %%
post_mtype_counts = count_synapses_by_sample(
    pre_synapses, resolved_pre_synapses, "post_mtype"
)

operation_feature_key = "time"
operation_key = "metaoperation_added"
fillna = metaoperation_stats[operation_feature_key].min()


def wrangle_counts_by_edit_sample(
    metaoperation_stats, counts_table, operation_feature_key, operation_key, fillna
):
    post_mtype_stats_tidy = counts_table.reset_index().melt(
        var_name="post_mtype", value_name="count", id_vars="sample"
    )

    post_mtype_probs = counts_table / counts_table.sum(axis=1).values[:, None]
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

    post_mtype_stats_tidy["time"] = (
        post_mtype_stats_tidy[operation_key]
        .map(metaoperation_stats[operation_feature_key])
        .fillna(fillna)
    )

    if "time" in operation_feature_key:
        post_mtype_stats_tidy[operation_feature_key] = pd.to_datetime(
            post_mtype_stats_tidy[operation_feature_key]
        )

    return post_mtype_stats_tidy


post_mtype_stats_tidy = wrangle_counts_by_edit_sample(
    metaoperation_stats, post_mtype_counts, operation_feature_key, operation_key, fillna
)


# %%
sns.set_context("talk")

name_map = {
    "count": "Synapse count",
    "prob": "Output proportion",
    "time": "Time",
    "centroid_distance_to_nuc_um": "Distance to nucleus (um)",
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

    ax.spines[['right', 'top']].set_visible(False)

    return fig, ax

#%%
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
        rotate_set_labels(ax)
        savefig(name, fig, folder="edit_replay_ordering")


# %%

# generate edit selections, ordered by distance

metaoperation_stats.sort_values("centroid_distance_to_nuc_um", inplace=True)
metaoperation_stats

edit_selections = {}
for i in range(len(metaoperation_stats) + 1):
    edit_selections[i] = metaoperation_stats.index[:i].tolist()


# %%
resolved_pre_synapses, resolved_post_synapses = resolve_synapses_from_edit_selections(
    nf, edit_selections, root_id, client
)

post_mtype_counts = count_synapses_by_sample(
    pre_synapses, resolved_pre_synapses, "post_mtype"
)

# %%

operation_feature_key = "centroid_distance_to_nuc_um"
operation_key = "metaoperation_added"
fillna = metaoperation_stats[operation_feature_key].min()


def wrangle_counts_by_edit_sample(
    metaoperation_stats, counts_table, operation_feature_key, operation_key, fillna
):
    post_mtype_stats_tidy = counts_table.reset_index().melt(
        var_name="post_mtype", value_name="count", id_vars="sample"
    )

    post_mtype_probs = counts_table / counts_table.sum(axis=1).values[:, None]
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

    post_mtype_stats_tidy[operation_feature_key] = (
        post_mtype_stats_tidy[operation_key]
        .map(metaoperation_stats[operation_feature_key])
        .fillna(fillna)
    )

    if "time" in operation_feature_key:
        post_mtype_stats_tidy[operation_feature_key] = pd.to_datetime(
            post_mtype_stats_tidy[operation_feature_key]
        )

    return post_mtype_stats_tidy


post_mtype_stats_tidy = wrangle_counts_by_edit_sample(
    metaoperation_stats, post_mtype_counts, operation_feature_key, operation_key, fillna
)
post_mtype_stats_tidy

# %%

x = "centroid_distance_to_nuc_um"
hue = "post_mtype"
spans = [None]

for y in ["count", "prob"]:
    for span in spans:
        name = f"{y}_vs_{x}_by_{hue}-root_id={root_id}"
        fig, ax = editplot(post_mtype_stats_tidy, x, y, hue=hue)
        if span is not None:
            ax.set_xlim(*span)
            name += "-span"
        rotate_set_labels(ax)
        savefig(name, fig, folder="edit_replay_ordering")
