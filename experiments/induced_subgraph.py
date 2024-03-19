# %%

import time

import caveclient as cc
import numpy as np
import pandas as pd
from cloudfiles import CloudFiles
from joblib import Parallel, delayed

from pkg.neuronframe import load_neuronframe
from pkg.sequence import create_time_ordered_sequence

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


client = cc.CAVEclient("minnie65_phase3_v1")


# %%
# all_root_ids = []
# relevant_synapses = []
# for root_id in tqdm(root_options[:]):
#     pass
#     # neuron = load_neuronframe(root_id, client, cache_verbose=False)


def get_pre_post_synapse_ids(root_id):
    neuron = load_neuronframe(root_id, client, cache_verbose=False)
    return neuron.pre_synapses.index, neuron.post_synapses.index


currtime = time.time()
pre_posts = Parallel(n_jobs=8, verbose=10)(
    delayed(get_pre_post_synapse_ids)(root_id) for root_id in root_options[:]
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

all_pre_synapses = []
all_post_synapses = []
for pre, post in pre_posts:
    all_pre_synapses.extend(pre)
    all_post_synapses.extend(post)

# %%


currtime = time.time()
induced_synapses = np.intersect1d(all_pre_synapses, all_post_synapses)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

print("number of induced synapses:", len(induced_synapses))

# %%
root_id = root_options[0]
neuron = load_neuronframe(root_id, client, cache_verbose=False)
sequence = create_time_ordered_sequence(neuron, root_id)
# %%


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


pre_synapse_sets = process_resolved_synapses(sequence, root_id, which="pre")
post_synapse_sets = process_resolved_synapses(sequence, root_id, which="post")

# %%


def get_pre_post_synapses_by_time(root_id):
    neuron = load_neuronframe(root_id, client, cache_verbose=False)
    sequence = create_time_ordered_sequence(neuron, root_id)
    pre_synapse_sets = process_resolved_synapses(sequence, root_id, which="pre")
    post_synapse_sets = process_resolved_synapses(sequence, root_id, which="post")
    return pre_synapse_sets, post_synapse_sets


pre_post_synapses = Parallel(n_jobs=8, verbose=10)(
    delayed(get_pre_post_synapses_by_time)(root_id) for root_id in root_options[:]
)

# %%

pre_synapselist = []
post_synapselist = []

for pre, post in pre_post_synapses:
    pre_synapselist.append(pre)
    post_synapselist.append(post)

pre_synapses = pd.concat(pre_synapselist)
post_synapses = pd.concat(post_synapselist)


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

client = cc.CAVEclient("minnie65_phase3_v1")

root_options = np.array(
    [
        864691135323181212,
        864691135383669466,
        864691135416719546,
        864691135571546917,
        864691135660772080,
        864691135772774651,
        864691135808473885,
        864691135953216547,
        864691136005322698,
        864691136389585015,
        864691137197468481,
    ]
)
object_table = pd.DataFrame()
object_table["working_root_id"] = root_options

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
object_table["current_root_id"] = updated_root_options

# map to nucleus IDs
current_nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"pt_root_id": updated_root_options},
    select_columns=["id", "pt_root_id"],
).set_index("pt_root_id")["id"]
object_table["target_id"] = object_table["current_root_id"].map(current_nucs)

# for those nucleus IDs, get previous root IDs
past_nucs = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": object_table["target_id"].to_list()},
    # select_columns=["id", "pt_root_id"],
    timestamp=timestamp,
)

# .set_index("id")["pt_root_id"]
# object_table["past_root_id"] = object_table["target_id"].map(past_nucs)

# # for those past root IDs, get the synapses
# client_synapse_table = client.materialize.synapse_query(
#     pre_ids=past_nucs,
#     post_ids=past_nucs,
#     timestamp=timestamp,
#     remove_autapses=True,
# )

object_table

# %%
import caveclient as cc
import numpy as np
import pandas as pd

client = cc.CAVEclient("minnie65_phase3_v1")

root_options = np.array(
    [
        864691135323181212,
        864691135383669466,
        864691135416719546,
        864691135571546917,
        864691135660772080,
        864691135772774651,
        864691135808473885,
        864691135953216547,
        864691136005322698,
        864691136389585015,
        864691137197468481,
    ]
)
object_table = pd.DataFrame()
object_table["working_root_id"] = root_options

nuc_table = client.materialize.tables.nucleus_detection_v0(
    pt_root_id=object_table["working_root_id"]
).query()

object_table = object_table.merge(
    nuc_table[["pt_root_id", "id"]].rename(
        columns={"pt_root_id": "working_root_id", "id": "soma_id"}
    ),
    on="working_root_id",
)

timestamp = pd.to_datetime("2021-07-01 00:00:00", utc=True)

old_roots = client.materialize.tables.nucleus_detection_v0(
    id=object_table["soma_id"]
).query(timestamp=timestamp)

object_table = object_table.merge(
    old_roots[["id", "pt_root_id"]].rename(
        columns={"id": "soma_id", "pt_root_id": "old_root_id"}
    ),
    on="soma_id",
)

# client_synapse_table = client.materialize.synapse_query(
#     pre_ids=object_table["old_root_id"],
#     post_ids=object_table["old_root_id"],
#     timestamp=timestamp,
#     remove_autapses=True,
# )

object_table

# %%

import caveclient as cc
import numpy as np
import pandas as pd

client = cc.CAVEclient("minnie65_phase3_v1")

timestamp = pd.to_datetime("2021-07-01 00:00:00", utc=True)

soma_ids = [
    292864,
    291116,
    303149,
    264824,
    292670,
    260541,
    301085,
    294825,
    292649,
    298937,
    262678,
]

old_roots = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": soma_ids},
    timestamp=timestamp,
).set_index("id")["pt_root_id"]
print(old_roots)

old_roots_w_select = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": soma_ids},
    select_columns=["id", "pt_root_id"],
    timestamp=timestamp,
).set_index("id")["pt_root_id"]
print(old_roots_w_select)

old_roots_w_select_no_time = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"id": soma_ids},
    select_columns=["id", "pt_root_id"],
).set_index("id")["pt_root_id"]
print(old_roots_w_select_no_time)


# %%
from datetime import timedelta

start = timestamp - timedelta(microseconds=1)
end = timestamp + timedelta(microseconds=1)
client.chunkedgraph.is_valid_nodes(object_table["past_root_id"].to_list(), start, end)


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
