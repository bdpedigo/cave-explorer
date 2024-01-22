# %%
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pkg.edits import (
    collate_edit_info,
    get_initial_network,
    get_operation_metaoperation_map,
    lazy_load_network_edits,
    pseudo_apply_edit,
)
from pkg.morphology import (
    apply_nucleus,
    apply_synapses,
    find_nucleus_component,
)
from pkg.plot import savefig
from pkg.utils import get_positions
from tqdm.auto import tqdm

import caveclient as cc
from networkframe import NetworkFrame

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
client.materialize.query_table(
    "nucleus_detection_v0",
    filter_equal_dict={"pt_root_id": root_id},
    # select_columns=["pt_supervoxel_id"],
)

# %%
type(client.materialize)


# %%

operation_to_metaoperation = get_operation_metaoperation_map(
    networkdeltas_by_metaoperation
)

# %%

edit_stats, modified_level2_nodes = collate_edit_info(
    networkdeltas_by_operation, operation_to_metaoperation, root_id, client
)

# %%

metaoperation_stats = edit_stats.groupby("metaoperation_id").agg(
    {
        "centroid_x": "mean",
        "centroid_y": "mean",
        "centroid_z": "mean",
        "centroid_distance_to_nuc_um": "min",
    }
)

metaoperation_stats.sort_values("centroid_distance_to_nuc_um", inplace=True)
metaoperation_stats


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

# generate edit selections

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

    choice_time = 0
    query_time = 0
    find_component_time = 0
    record_time = 0
    for selection_name, edit_selection in tqdm(edit_selections.items()):
        if not isinstance(edit_selection, list):
            edit_selection = list(edit_selection)

        # -1 for the initial state of the neuron
        if -1 not in edit_selection:
            edit_selection.append(-1)

        print("At query nodes...")
        t = time.time()
        sub_nf = nf.query_nodes(
            f"{prefix}operation_added.isin(@edit_selection)", local_dict=locals()
        ).query_edges(
            f"{prefix}operation_added.isin(@edit_selection)", local_dict=locals()
        )
        query_time += time.time() - t

        t = time.time()
        print("At find component...")
        # this takes up 90% of the time
        # i think it's from the operation of cycling through connected components
        instance_neuron_nf = find_nucleus_component(sub_nf, root_id, client)

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


post_mtype_counts_by_sample = []
for i, key in enumerate(resolved_pre_synapses.keys()):
    sample_resolved_synapses = resolved_pre_synapses[key]

    sample_post_mtype_counts = (
        pre_synapses.loc[sample_resolved_synapses].groupby("post_mtype").size()
    )
    sample_post_mtype_counts.name = i
    post_mtype_counts_by_sample.append(sample_post_mtype_counts)

post_mtype_counts = (
    pd.concat(post_mtype_counts_by_sample, axis=1).fillna(0).astype(int).T
)
post_mtype_counts.index.name = "sample"
post_mtype_counts
# %%
post_mtype_counts_tidy = post_mtype_counts.reset_index().melt(
    var_name="post_mtype", value_name="count", id_vars="sample"
)
post_mtype_counts_tidy["metaoperation"] = (
    (post_mtype_counts_tidy["sample"] - 1)
    .map(metaoperation_stats.index.to_series().reset_index(drop=True))
    .fillna(-1)
    .astype(int)
)
post_mtype_counts_tidy["distance_to_nuc_um"] = (
    post_mtype_counts_tidy["metaoperation"]
    .map(metaoperation_stats["centroid_distance_to_nuc_um"])
    .fillna(0)
)
post_mtype_counts_tidy
# %%

fig, ax = plt.subplots(figsize=(6, 6))
sns.set_context("talk")
sns.lineplot(
    data=post_mtype_counts_tidy,
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
    data=post_mtype_counts_tidy,
    x="distance_to_nuc_um",
    y="count",
    hue="post_mtype",
    ax=ax,
    legend=False,
    linewidth=1,
)
sns.scatterplot(
    data=post_mtype_counts_tidy,
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
