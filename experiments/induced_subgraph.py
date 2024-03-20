# %%

import time

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cloudfiles import CloudFiles
from graspologic.utils import pass_to_ranks
from joblib import Parallel, delayed
from networkframe import NetworkFrame

from pkg.neuronframe import load_neuronframe
from pkg.sequence import create_time_ordered_sequence

# %%
client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v795")

root_options = query_neurons["pt_root_id"].values

# %%

nodes = pd.DataFrame()
nodes["working_root_id"] = root_options

# take my list of root IDs
# make sure I have the latest root ID for each, using `get_latest_roots`
is_current_mask = client.chunkedgraph.is_latest_roots(root_options)
outdated_roots = root_options[~is_current_mask]
root_map = dict(zip(root_options[is_current_mask], root_options[is_current_mask]))
for outdated_root in outdated_roots:
    latest_roots = client.chunkedgraph.get_latest_roots(outdated_root)
    sub_nucs = client.materialize.query_table(
        "nucleus_detection_v0", filter_in_dict={"pt_root_id": latest_roots}
    )
    if len(sub_nucs) == 1:
        root_map[outdated_root] = sub_nucs.iloc[0]["pt_root_id"]
    else:
        print(f"Multiple nuc roots for {outdated_root}")

updated_root_options = np.array([root_map[root] for root in root_options])
nodes["current_root_id"] = updated_root_options

# map to nucleus IDs
current_nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"pt_root_id": updated_root_options},
    # select_columns=["id", "pt_root_id"],
).set_index("pt_root_id")["id"]
nodes["target_id"] = nodes["current_root_id"].map(current_nucs)


# %%
timestamp = pd.to_datetime("2021-07-01 00:00:00", utc=True)

nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": nodes["target_id"].to_list()},
).set_index("id")
nodes["pt_supervoxel_id"] = nodes["target_id"].map(nucs["pt_supervoxel_id"])
nodes["timestamp_root_from_chunkedgraph"] = client.chunkedgraph.get_roots(
    nodes["pt_supervoxel_id"], timestamp=timestamp
)
nodes["nuc_depth"] = nodes["target_id"].map(nucs["pt_position"].apply(lambda x: x[1]))

past_nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": nodes["target_id"].to_list()},
    # select_columns=["id", "pt_root_id"],
    timestamp=timestamp,
).set_index("id")["pt_root_id"]
nodes["timestamp_root_from_table"] = nodes["target_id"].map(past_nucs)

mtypes = client.materialize.query_table(
    "allen_column_mtypes_v2", filter_in_dict={"target_id": nodes["target_id"].to_list()}
)
nodes["mtype"] = nodes["target_id"].map(mtypes.set_index("target_id")["cell_type"])

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

root_options = files_finished["root_id"].unique()

# %%
nodes["has_sequence"] = nodes["current_root_id"].isin(root_options)
nodes["ctype"] = nodes["target_id"].map(
    query_neurons.set_index("target_id")["cell_type"]
)
# %%
root_options = nodes.query("has_sequence")["working_root_id"]

# %%
# timestamps = pd.date_range("2022-07-01", "2024-01-01", freq="M", tz="UTC")


# loop over the entire set of nodes to consider and get the collection of pre
# and post synapses for each, across time


def get_pre_post_synapse_ids(root_id):
    neuron = load_neuronframe(root_id, client, cache_verbose=False)
    return neuron.pre_synapses.index, neuron.post_synapses.index


currtime = time.time()
pre_posts = Parallel(n_jobs=8, verbose=10)(
    delayed(get_pre_post_synapse_ids)(root_id) for root_id in root_options
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

all_pre_synapses = []
all_post_synapses = []
for pre, post in pre_posts:
    all_pre_synapses.extend(pre)
    all_post_synapses.extend(post)

# %%

# the induced synapses is the set of all synapses which show up at least once pre, and
# at least once post across all time

induced_synapses = np.intersect1d(all_pre_synapses, all_post_synapses)

print("Number of induced synapses:", len(induced_synapses))

# %%

# now, extract the synapses that are in this induced set for each neuron.


def process_resolved_synapses(sequence, root_id, which="pre"):
    synapse_sets = sequence.sequence_info[f"{which}_synapses"]
    synapse_sets = synapse_sets.apply(
        lambda x: np.intersect1d(x, induced_synapses)
    ).to_frame()
    synapse_sets["time"] = sequence.edits["time"]
    synapse_sets["time"] = synapse_sets["time"].fillna("2019-07-01 00:00:00")
    synapse_sets["datetime"] = pd.to_datetime(synapse_sets["time"], utc=True)
    synapse_sets[f"{which}_root_id"] = root_id

    breaks = list(synapse_sets["datetime"])
    breaks.append(
        pd.to_datetime("2070-01-01 00:00:00", utc=True)
    )  # TODO could make this explicit about now
    intervals = pd.IntervalIndex.from_breaks(breaks, closed="left")
    synapse_sets["interval"] = intervals

    synapse_sets = synapse_sets.reset_index(drop=False)
    synapses = synapse_sets.explode(f"{which}_synapses").rename(
        columns={f"{which}_synapses": "synapse_id"}
    )
    return synapses


def get_info_by_time(root_id):
    neuron = load_neuronframe(root_id, client, cache_verbose=False)
    sequence = create_time_ordered_sequence(neuron, root_id)
    pre_synapse_sets = process_resolved_synapses(sequence, root_id, which="pre")
    post_synapse_sets = process_resolved_synapses(sequence, root_id, which="post")
    edits = neuron.edits.copy()
    edits["root_id"] = root_id
    return pre_synapse_sets, post_synapse_sets, edits


outs = Parallel(n_jobs=8, verbose=10)(
    delayed(get_info_by_time)(root_id) for root_id in root_options
)


pre_synapselist = []
post_synapselist = []
all_edit_tables = []

for pre, post, edit_table in outs:
    pre_synapselist.append(pre)
    post_synapselist.append(post)
    all_edit_tables.append(edit_table)

pre_synapses = pd.concat(pre_synapselist)
post_synapses = pd.concat(post_synapselist)
all_edits = pd.concat(all_edit_tables)


# %%
def synapselist_at_time(timestamp, remove_loops=True):
    # pre_synapses_at_time = pre_synapses.query("@timestamp in interval")
    # post_synapses_at_time = post_synapses.query("@timestamp in interval")
    pre_synapses_at_time = pre_synapses[
        pd.IntervalIndex(pre_synapses.interval).contains(timestamp)
    ]
    post_synapses_at_time = post_synapses[
        pd.IntervalIndex(post_synapses.interval).contains(timestamp)
    ]

    pre_synapses_at_time = pre_synapses_at_time.set_index("synapse_id")
    post_synapses_at_time = post_synapses_at_time.set_index("synapse_id")
    synapselist = pre_synapses_at_time.join(
        post_synapses_at_time, how="inner", lsuffix="_pre", rsuffix="_post"
    )
    synapselist["source"] = synapselist["pre_root_id"]
    synapselist["target"] = synapselist["post_root_id"]
    if remove_loops:
        synapselist = synapselist.query("source != target")
    return synapselist


synapselist = synapselist_at_time(pd.to_datetime("2021-07-01 00:00:00", utc=True))

# %%

# nodes = pd.DataFrame()
# nodes.index = root_options
timestamps = pd.date_range("2020-07-01", "2024-01-01", freq="M", tz="UTC")

used_nodes = nodes.query("working_root_id in @root_options")
nfs_by_time = {}
for timestamp in timestamps:
    synapselist = synapselist_at_time(timestamp)
    nf = NetworkFrame(used_nodes.set_index("working_root_id"), synapselist)
    nfs_by_time[timestamp] = nf

# %%
final_nf = nf

ordering = {"PeriTC": 0, "DistTC": 1, "SparTC": 2, "InhTC": 3}
final_nf.nodes["mtype_order"] = final_nf.nodes["mtype"].map(ordering)
final_nf.nodes = final_nf.nodes.sort_values(["mtype_order", "ctype", "nuc_depth"])

# %%

fig, axs = plt.subplots(3, 5, figsize=(15, 9), constrained_layout=True)
for i, (timestamp, nf) in enumerate(nfs_by_time.items()):
    if i > 14:
        break
    nf.nodes = nf.nodes.loc[final_nf.nodes.index]
    adj = nf.to_sparse_adjacency(weight_col=None).todense()
    adj = pass_to_ranks(adj)
    ax = axs.flatten()[i]
    sns.heatmap(
        adj,
        ax=ax,
        square=True,
        cbar=False,
        cmap="RdBu_r",
        center=0,
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title(timestamp.strftime("%Y-%m"))

# %%
diffs_by_time = pd.DataFrame(index=timestamps, columns=timestamps, dtype=float)
for i, timestamp1 in enumerate(timestamps):
    adj1 = nfs_by_time[timestamp1].to_sparse_adjacency(weight_col=None)
    adj1 = adj1.todense()
    for j, timestamp2 in enumerate(timestamps):
        adj2 = nfs_by_time[timestamp2].to_sparse_adjacency(weight_col=None)
        adj2 = adj2.todense()
        diff = np.linalg.norm(adj1 - adj2, ord="fro")
        diffs_by_time.loc[timestamps[i], timestamps[j]] = diff
        diffs_by_time.loc[timestamps[j], timestamps[i]] = diff

diffs_by_time.fillna(0.0, inplace=True)

# %%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(
    diffs_by_time, cmap="RdBu_r", center=0, square=True, ax=ax, cbar_kws={"shrink": 0.5}
)
labels = diffs_by_time.columns.strftime("%Y-%m")
ax.set_yticklabels(labels, rotation=0)
ax.set_xticklabels([])
ax.set_xticks([])

# %%
from graspologic.embed import ClassicalMDS

cmds = ClassicalMDS(n_components=2, dissimilarity="precomputed")
embedding = cmds.fit_transform(diffs_by_time.values)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(embedding[:, 0], embedding[:, 1])
for i, timestamp in enumerate(timestamps):
    ax.text(embedding[i, 0], embedding[i, 1], timestamp.strftime("%Y-%m"))


# %%

synapselist
# %%

timestamp = pd.to_datetime("2021-07-01 00:00:00", utc=True)

# convert this timestamp to a utc timestamp
# timestamp = timestamp.tz_localize("UTC")

# synapse_table = client.materialize.query_table(
#     "synapses_pni_2",
#     filter_in_dict={"pre_pt_root_id": root_options, "post_pt_root_id": root_options},
#     timestamp=timestamp,
# )
# synapse_table.query("pre_pt_root_id != post_pt_root_id")

# %%
nuc_table = client.materialize.query_table(
    "nucleus_detection_v0", filter_in_dict={"pt_root_id": root_options}
)


# %%

nodes = pd.DataFrame()
nodes["working_root_id"] = root_options

# take my list of root IDs
# make sure I have the latest root ID for each, using `get_latest_roots`
is_current_mask = np.isin(root_options, nuc_table.pt_root_id.unique())
outdated_roots = root_options[~is_current_mask]
root_map = dict(zip(root_options[is_current_mask], root_options[is_current_mask]))
for outdated_root in outdated_roots:
    latest_roots = client.chunkedgraph.get_latest_roots(outdated_root)
    for latest_root in latest_roots:
        root_map[outdated_root] = latest_root
updated_root_options = np.array([root_map[root] for root in root_options])
nodes["current_root_id"] = updated_root_options

# map to nucleus IDs
current_nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"pt_root_id": updated_root_options},
    # select_columns=["id", "pt_root_id"],
).set_index("pt_root_id")["id"]
nodes["target_id"] = nodes["current_root_id"].map(current_nucs)

# for those nucleus IDs, get previous root IDs
past_nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": nodes["target_id"].to_list()},
    # select_columns=["id", "pt_root_id"],
    timestamp=timestamp,
).set_index("id")["pt_root_id"]
nodes["past_root_id"] = nodes["target_id"].map(past_nucs)

# for those past root IDs, get the synapses
client_synapse_table = client.materialize.synapse_query(
    pre_ids=past_nucs,
    post_ids=past_nucs,
    timestamp=timestamp,
    remove_autapses=True,
).set_index("id")

# %%
client_synapse_table.index.difference(synapselist.index)

# %%
times = sequence.sequence_info["datetime"].copy()
times.fillna(pd.to_datetime("2019-07-01 00:00:00", utc=True), inplace=True)
times = times.sort_values()
found_iloc = times.searchsorted(timestamp, side="left") - 1
found_operation = sequence.sequence_info.index[found_iloc]

sequence.sequence_info.loc[found_operation, "timestamp"]

# synapse_sets["time"] = synapse_sets["time"].fillna("2019-07-01 00:00:00")
# synapse_sets["datetime"] = pd.to_datetime(synapse_sets["time"], utc=True)

# %%
edits_to_apply = times.loc[:found_operation].index.dropna()

# %%
test_neuron = (
    neuron.set_edits(edits_to_apply, inplace=False)
    .select_nucleus_component(inplace=False)
    .remove_unused_synapses(inplace=False)
)

# %%
old_root = neuron.edits.loc[242946, "after_root_ids"][0]

# %%

old_l2_graph = client.chunkedgraph.level2_chunk_graph(old_root)

# %%
old_post_synapses = client.materialize.synapse_query(
    post_ids=old_root, timestamp=timestamp
)

# %%
old_pre_synapses
# %%
client.materialize.query_table(
    "synapses_pni_2", filter_equal_dict={"p_pt_root_id": old_root}, timestamp=timestamp
)

# %%

# find the nucleus supervoxel ID that anchors each of these
relevant_nuc_table = nuc_table.set_index("pt_root_id").loc[updated_root_options]

# now for the timestamp I am interested in, get the root IDs of the nucleus at that time
old_root_ids = client.chunkedgraph.get_roots(
    relevant_nuc_table["pt_supervoxel_id"], timestamp=timestamp
)
relevant_nuc_table["old_root_id"] = old_root_ids

# for those root IDs, do a synapse lookup
# TODO use synapse query here
# synapse table has a lot of false positives at nuclei
#
old_synapse_table = client.materialize.query_table(
    "synapses_pni_2",
    filter_in_dict={"pre_pt_root_id": old_root_ids, "post_pt_root_id": old_root_ids},
    timestamp=timestamp,
)

# remove loops
old_synapse_table = old_synapse_table.query("pre_pt_root_id != post_pt_root_id")

# formatting
old_synapse_table = old_synapse_table.set_index("id")
old_synapse_table.index.name = "synapse_id"

# %%
synapselist

# %%

old_synapse_table["pre_pt_root_id"].isin(old_root_ids).all()

# %%
old_synapse_table["post_pt_root_id"].isin(old_root_ids).all()

# %%
# all of my synapses are present in the "correct" synapse table
mask = synapselist.index.isin(old_synapse_table.index)
mask.mean()

# %%

# .99 of these are even in the induced synapses table
mask = old_synapse_table.index.isin(synapselist.index)
print(mask.mean())

old_synapse_table[~mask]

# %%
old_synapse_table.index.isin(induced_synapses).mean()

# %%
old_synapse_table[~old_synapse_table.index.isin(induced_synapses)]

# %%
# 1/3 of synapses from the true version are not even in the intersection of pre and post
# synapse IDs across all time for this collection of neurons

query = root_options[0]
new_query = relevant_nuc_table.loc[query, "old_root_id"]
query_pre_synapses = old_synapse_table.query("pre_pt_root_id == @new_query")

# %%
from pkg.morphology.synapses import get_alltime_synapses

for root_id in root_options[:1]:
    pre_syns, post_syns = get_alltime_synapses(root_id, client)

# %%
query_pre_synapses["id"].isin(pre_syns.index).mean()

# %%

import datetime

timestamp = datetime.datetime.now(datetime.timezone.utc)

root_id = 864691134886015738
original_roots = client.chunkedgraph.get_original_roots(root_id)
latest_roots = client.chunkedgraph.get_latest_roots(original_roots)
# timestamp = pd.to_datetime("2021-07-01 00:00:00")


side = "pre"
synapse_table = client.info.get_datastack_info()["synapse_table"]
candidate_synapses = client.materialize.query_table(
    synapse_table,
    filter_in_dict={f"{side}_pt_root_id": latest_roots},
)

# %%
# everything in the correct answer ends up in this candidate pool
query_pre_synapses["id"].isin(candidate_synapses["id"]).mean()

# %%
query_pre_synapses["id"].isin(neuron.pre_synapses.index).mean()

# %%
query_pre_synapses.query("pre_pt_root_id != post_pt_root_id")
