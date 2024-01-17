# %%
import os
import time

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from pkg.edits import (
    collate_edit_info,
    find_supervoxel_component,
    get_initial_network,
    get_operation_metaoperation_map,
    lazy_load_network_edits,
    pseudo_apply_edit,
)
from pkg.morphology import (
    apply_nucleus,
    apply_synapses,
)
from pkg.plot import savefig
from pkg.utils import get_positions

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
query_neurons.sort_values("id", inplace=True)

nuc = client.materialize.query_table("nucleus_detection_v0").set_index("pt_root_id")

# %%

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"

# %%

root_id = query_neurons["pt_root_id"].values[10]

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

# label edges which cross operations/metaoperations
nf.edges["cross_operation"] = (
    nf.edges["source_operation_added"] != nf.edges["target_operation_added"]
)
nf.edges["cross_metaoperation"] = (
    nf.edges["source_metaoperation_added"] != nf.edges["target_metaoperation_added"]
)
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

# NOTE: this is the "random sets of synapses" approach
all_metaoperations = list(networkdeltas_by_metaoperation.keys())

resolved_synapses_by_sample = {}

prefix = "meta"
# still only takes ~ 0.5 mins for a neuron, to do 100 samples
# so 1000 samples would take 5 mins... same order as extracting the edits
choice_time = 0
query_time = 0
find_component_time = 0
record_time = 0
for i in tqdm(range(100)):
    t = time.time()
    metaoperation_set = np.random.choice(all_metaoperations, size=10, replace=False)
    metaoperation_set = list(metaoperation_set)
    # add -1; this denotes the initial state of the neuron i.e. things not added
    metaoperation_set.append(-1)
    choice_time += time.time() - t

    t = time.time()
    sub_nf = nf.query_nodes(
        f"{prefix}operation_added.isin(@metaoperation_set)", local_dict=locals()
    ).query_edges(
        f"{prefix}operation_added.isin(@metaoperation_set)", local_dict=locals()
    )
    query_time += time.time() - t

    t = time.time()
    # this takes up 90% of the time
    # i think it's from the operation of cycling through connected components
    nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]
    instance_neuron_nf = find_supervoxel_component(nuc_supervoxel, sub_nf, client)
    find_component_time += time.time() - t

    t = time.time()
    found_synapses = []
    for synapses in instance_neuron_nf.nodes["synapses"]:
        found_synapses.extend(synapses)

    metaoperation_set = metaoperation_set[:-1]
    metaoperation_set = tuple(np.unique(metaoperation_set))
    resolved_synapses_by_sample[metaoperation_set] = found_synapses
    record_time += time.time() - t

total_time = choice_time + query_time + find_component_time + record_time
print(f"Total time: {total_time}")
print(f"Choice time: {choice_time / total_time}")
print(f"Query time: {query_time / total_time}")
print(f"Find component time: {find_component_time / total_time}")
print(f"Record time: {record_time / total_time}")

# %%

# TODO write function to do this as a function of an input ordering of metaoperations

# NOTE: this is the synapses ordered by distance to nucleus approach

resolved_pre_synapses_by_sample = {}
resolved_post_synapses_by_sample = {}


prefix = "meta"
# still only takes ~ 0.5 mins for a neuron, to do 100 samples
# so 1000 samples would take 5 mins... same order as extracting the edits
choice_time = 0
query_time = 0
find_component_time = 0
record_time = 0
for i in tqdm(range(len(metaoperation_stats) + 1)):
    t = time.time()
    metaoperation_set = metaoperation_stats.index[:i]
    metaoperation_set = list(metaoperation_set)
    # add -1; this denotes the initial state of the neuron i.e. things not added
    metaoperation_set.append(-1)
    choice_time += time.time() - t

    t = time.time()
    sub_nf = nf.query_nodes(
        f"{prefix}operation_added.isin(@metaoperation_set)", local_dict=locals()
    ).query_edges(
        f"{prefix}operation_added.isin(@metaoperation_set)", local_dict=locals()
    )
    query_time += time.time() - t

    t = time.time()
    # this takes up 90% of the time
    # i think it's from the operation of cycling through connected components
    nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]
    instance_neuron_nf = find_supervoxel_component(nuc_supervoxel, sub_nf, client)
    find_component_time += time.time() - t

    t = time.time()

    found_pre_synapses = []
    for synapses in instance_neuron_nf.nodes["pre_synapses"]:
        found_pre_synapses.extend(synapses)

    found_post_synapses = []
    for synapses in instance_neuron_nf.nodes["post_synapses"]:
        found_post_synapses.extend(synapses)

    metaoperation_set = metaoperation_set[:-1]
    metaoperation_set = tuple(np.unique(metaoperation_set))

    resolved_synapses_by_sample[metaoperation_set] = found_synapses
    resolved_pre_synapses_by_sample[metaoperation_set] = found_pre_synapses
    resolved_post_synapses_by_sample[metaoperation_set] = found_post_synapses

    record_time += time.time() - t

total_time = choice_time + query_time + find_component_time + record_time
print(f"Total time: {total_time}")
print(f"Choice time: {choice_time / total_time}")
print(f"Query time: {query_time / total_time}")
print(f"Find component time: {find_component_time / total_time}")
print(f"Record time: {record_time / total_time}")


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
for i, key in enumerate(resolved_pre_synapses_by_sample.keys()):
    sample_resolved_synapses = resolved_pre_synapses_by_sample[key]

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
savefig("post_mtype_counts_vs_sample", fig, folder="edit_replay_ordering")

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
savefig("post_mtype_counts_vs_sample", fig, folder="edit_replay_ordering")


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

savefig("post_mtype_probs_vs_sample", fig, folder="edit_replay_ordering")


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
savefig("post_mtype_prob_vs_sample", fig, folder="edit_replay_ordering")


# %%
