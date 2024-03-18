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
    synapse_sets["time"] = synapse_sets["time"].fillna("2020-07-01 00:00:00")
    synapse_sets["datetime"] = pd.to_datetime(synapse_sets["time"])
    synapse_sets[f"{which}_root_id"] = root_id

    breaks = list(synapse_sets["datetime"])
    breaks.append(
        pd.to_datetime("2070-01-01 00:00:00")
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
