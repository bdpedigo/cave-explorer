# %%
import os
import pickle

import caveclient as cc

# %%

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"


# %%

client = cc.CAVEclient("minnie65_phase3_v1")

query_neurons = client.materialize.query_table("connectivity_groups_v507")
query_neurons.sort_values("id", inplace=True)

# 10 looked "unstable"
# 11 looked "stable"
root_id = query_neurons["pt_root_id"].values[10]
# root_id = 864691135865971164
# root_id = 864691135992790209

# %%
from pkg.neuronframe import load_neuronframe

full_neuron = load_neuronframe(root_id, client)

# %%
palette_file = "/Users/ben.pedigo/code/skedits/skedits-app/skedits/data/ctype_hues.pkl"

with open(palette_file, "rb") as f:
    ctype_hues = pickle.load(f)

ctype_hues = {ctype: tuple(ctype_hues[ctype]) for ctype in ctype_hues.keys()}

# %%
full_neuron.edits.groupby("metaoperation_id").groups
# %%
metaedits = full_neuron.metaedits.sort_values("time")

# %%


# %%

from tqdm.auto import tqdm

prefix = "meta"
pure_split_metaedits = metaedits.query("~has_merge")

merge_metaedits = metaedits.query("has_merge")


merge_op_ids = merge_metaedits.index
split_op_ids = pure_split_metaedits.index
applied_op_ids = list(split_op_ids)


neuron_list = []
applied_merges = []
for i in tqdm(range(len(merge_op_ids) + 1)):
    # apply the next operation
    current_neuron = full_neuron.set_edits(applied_op_ids, inplace=False, prefix=prefix)
    current_neuron.select_nucleus_component(inplace=True)
    current_neuron.remove_unused_synapses(inplace=True)
    neuron_list.append(current_neuron)

    # select the next operation to apply
    out_edges = full_neuron.edges.query(
        "source.isin(@current_neuron.nodes.index) | target.isin(@current_neuron.nodes.index)"
    )
    # print(len(out_edges), "out edges")

    out_edges = out_edges.drop(current_neuron.edges.index)

    # print(len(out_edges), "out edges after removing current edges")

    possible_operations = out_edges[f"{prefix}operation_added"].unique()
    # print(len(possible_operations), "possible operations")

    ordered_ops = merge_op_ids[merge_op_ids.isin(possible_operations)]

    # HACK
    ordered_ops = ordered_ops[~ordered_ops.isin(applied_merges)]

    if len(ordered_ops) == 0:
        print(f"no remaining merges, stopping ({i / len(merge_op_ids):.2f})")
        break

    applied_op_ids.append(ordered_ops[0])
    applied_merges.append(ordered_ops[0])

current_neuron.generate_neuroglancer_link(client)

# %%
import numpy as np

# TODO i bet these are all things such that filtered = False
metaedits.loc[merge_op_ids[~np.isin(merge_op_ids, applied_merges)]]

# %%
# TODO check if current neuron at the end of this is equal to the final one with ALL
# edits applied

# %%
final_neuron = full_neuron.set_edits(full_neuron.edits.index, inplace=False)
final_neuron.select_nucleus_component(inplace=True)
final_neuron.remove_unused_synapses(inplace=True)
# %%
final_neuron == current_neuron

# %%
final_neuron.nodes.index.sort_values().equals(current_neuron.nodes.index.sort_values())

# %%
final_neuron.edges.index.sort_values().equals(current_neuron.edges.index.sort_values())

