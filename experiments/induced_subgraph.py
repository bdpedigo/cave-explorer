# %%

import time

import caveclient as cc
import pandas as pd
from cloudfiles import CloudFiles
from joblib import Parallel, delayed

from pkg.neuronframe import load_neuronframe

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

import numpy as np

currtime = time.time()
induced_synapses = np.intersect1d(all_pre_synapses, all_post_synapses)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

print("number of induced synapses:", len(induced_synapses))

#%%



# %%


def get_relevant_roots(root_id):
    lineage_graph = client.chunkedgraph.get_lineage_graph(root_id, as_nx_graph=True)
    nodes = list(lineage_graph.nodes)
    return nodes


currtime = time.time()
relevant_roots = Parallel(n_jobs=8, verbose=10)(
    delayed(get_relevant_roots)(root_id) for root_id in root_options[:]
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

relevant_roots = [item for sublist in relevant_roots for item in sublist]

# %%


all_relevant_pre_synapses = []
all_relevant_post_synapses = []


def get_relevant_synapses(root_id, relevant_roots):
    neuron = load_neuronframe(root_id, client, cache_verbose=False)
    relevant_pre_synapses = neuron.pre_synapses.query(
        "post_pt_root_id in @relevant_roots"
    ).copy()
    relevant_pre_synapses["root_id"] = root_id
    relevant_post_synapses = neuron.post_synapses.query(
        "pre_pt_root_id in @relevant_roots"
    ).copy()
    relevant_post_synapses["root_id"] = root_id
    return relevant_pre_synapses, relevant_post_synapses


all_relevant = Parallel(n_jobs=8, verbose=10)(
    delayed(lambda x: get_relevant_synapses(x, relevant_roots))(root_id)
    for root_id in root_options[:]
)

all_relevant_pre_synapses = [item[0] for item in all_relevant]
all_relevant_post_synapses = [item[1] for item in all_relevant]
# all_relevant_pre_synapses = pd.concat(all_relevant_pre_synapses)
# all_relevant_post_synapses = pd.concat(all_relevant_post_/synapses)

# %%

import numpy as np
from tqdm.auto import tqdm

from pkg.sequence import create_time_ordered_sequence

for root_id in tqdm(root_options[:10]):
    neuron = load_neuronframe(root_id, client)
    sequence = create_time_ordered_sequence(neuron, root_id)

    pre_synapse_sets = sequence.sequence_info["pre_synapses"]
    pre_synapse_sets = pre_synapse_sets.apply(
        lambda x: np.intersect1d(x, all_relevant_pre_synapses.index)
    ).to_frame()
    pre_synapse_sets["time"] = sequence.edits["time"]
    # for _, row in pre_synapse_set


# %%

pre_synapse_sets.explode("pre_synapses")
