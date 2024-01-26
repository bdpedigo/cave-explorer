# %%
import os
import pickle

import caveclient as cc
import pandas as pd
from tqdm.auto import tqdm

from pkg.edits import (
    apply_edit_history,
    apply_synapses,
    collate_edit_info,
    get_initial_network,
    get_operation_metaoperation_map,
    lazy_load_network_edits,
)
from pkg.morphology import (
    apply_nucleus,
    apply_positions,
)

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
palette_file = "/Users/ben.pedigo/code/skedits/skedits-app/skedits/data/ctype_hues.pkl"

with open(palette_file, "rb") as f:
    ctype_hues = pickle.load(f)

ctype_hues = {ctype: tuple(ctype_hues[ctype]) for ctype in ctype_hues.keys()}


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


apply_positions(nf, client)


# %%

pre_synapses, post_synapses = apply_synapses(
    nf,
    networkdeltas_by_operation,
    root_id,
    client,
)

# %%

nuc_level2_id = apply_nucleus(nf, root_id, client)

# TODO is it worth just caching the whole networkframe at this stage?

# %%

# linkages between dataframes to consider:
#
# nodes index <-> edges "source"/"target" (also its index)
# one to many
#
# nodes index <-> pre_synapses "pre_pt_root_id"
# one to many
#
# nodes index <-> post_synapses "post_pt_root_id"
# one to many
#
# nodes index <-> modified_level2_nodes index
# one to one
#
# nodes index <-> edit_stats "modified_nodes"
# many to one


# %%
from pkg.neuronframe import NeuronFrame

full_neuron = NeuronFrame(
    nodes=nf.nodes,
    edges=nf.edges,
    nucleus_id=nuc_level2_id,
    neuron_id=root_id,
    pre_synapses=pre_synapses,
    post_synapses=post_synapses,
    edits=edit_stats,
)
full_neuron

# %%
edited_neuron = full_neuron.set_edits(edit_stats.index[:20], inplace=False)
edited_neuron.select_nucleus_component(inplace=True)
edited_neuron.remove_unused_synapses(inplace=True)
edited_neuron.generate_neuroglancer_link(client)

# %%

sub = edited_neuron.select_by_ball(100_000, inplace=False)
sub.generate_neuroglancer_link(client)

# %%

# label edges which cross operations/metaoperations
full_neuron.edges["cross_operation"] = (
    full_neuron.edges["source_operation_added"]
    != full_neuron.edges["target_operation_added"]
)
full_neuron.edges["cross_metaoperation"] = (
    full_neuron.edges["source_metaoperation_added"]
    != full_neuron.edges["target_metaoperation_added"]
)
full_neuron.edges["was_removed"] = full_neuron.edges["operation_removed"] != -1
full_neuron.nodes["was_removed"] = full_neuron.nodes["operation_removed"] != -1

meta = True
if meta:
    prefix = "meta"
else:
    prefix = ""

# now, create a view of the graph such that we are only looking at edges which go
# between nodes that were added at the same time AND which were never removed later.
# this should give us a set of connected components which are meaningful "chunks" of
# neuron that share the same edit history/relationship to the nucleus in terms of
# operations.

full_neuron.apply_node_features("component_label", inplace=True)

cross_neuron = full_neuron.query_edges(f"cross_{prefix}operation").remove_unused_nodes()

component_edges = cross_neuron.edges.copy()
component_edges.reset_index(drop=True, inplace=True)
component_edges.rename(
    columns={
        "source_component_label": "source",
        "target_component_label": "target",
        "source": "source_l2_id",
        "target": "target_l2_id",
    },
    inplace=True,
)
component_nodelist = full_neuron.nodes["component_label"].unique()
component_nodes = pd.DataFrame(index=component_nodelist)

component_nodes[["x", "y", "z"]] = full_neuron.nodes.groupby("component_label")[
    ["x", "y", "z"]
].mean()
component_nodes["rep_coord_nm"] = [
    [x, y, z]
    for x, y, z in zip(component_nodes["x"], component_nodes["y"], component_nodes["z"])
]

nuc_component_id = full_neuron.nodes.loc[full_neuron.nucleus_id, "component_label"]

component_neuron = NeuronFrame(
    nodes=component_nodes,
    edges=component_edges,
    nucleus_id=nuc_component_id,
    neuron_id=root_id,
    edits=edit_stats,
)
component_neuron.select_by_ball(100_000).generate_neuroglancer_link(client)


# %%
full_neuron.select_nucleus_component().select_by_ball(
    50_000
).generate_neuroglancer_link(client)
# %%
og_neuron = full_neuron.set_edits([], inplace=False)
og_neuron.select_nucleus_component().select_by_ball(50_000).generate_neuroglancer_link(
    client
)

# %%
new_neuron = full_neuron.set_edits(full_neuron.edits.index[:50], inplace=False)
new_neuron.select_nucleus_component().select_by_ball(50_000).generate_neuroglancer_link(
    client
)

# %%

prefix = ""
no_cross_neuron = full_neuron.query_edges(
    f"(~cross_{prefix}operation) & (~was_removed)"
)  # .query_nodes("~was_removed")
n_connected_components = no_cross_neuron.n_connected_components()

full_neuron.nodes["component_label"] = 0

i = 2
for component in tqdm(
    no_cross_neuron.connected_components(), total=n_connected_components
):
    if (component.nodes[f"{prefix}operation_added"] == -1).all():
        label = -1 * i
        i += 1
    else:
        label = component.nodes[f"{prefix}operation_added"].iloc[0]
    if not (
        component.nodes[f"{prefix}operation_added"]
        == component.nodes[f"{prefix}operation_added"].iloc[0]
    ).all():
        print(component.nodes[f"{prefix}operation_added"])

    full_neuron.nodes.loc[component.nodes.index, "component_label"] = label

# nodes that get removed later have 0
# nodes that are not touched have a negative number
# nodes that are added have a positive number corresponding to the operation

# %%
full_neuron.nodes.query("component_label == 0")["was_removed"].mean()

# %%
full_neuron.select_by_ball(40_000).generate_neuroglancer_link_by_component(client)


# %%

full_neuron.edits.sort_values("datetime", inplace=True)

merge_op_ids = full_neuron.edits.query("is_merge").index
split_op_ids = full_neuron.edits.query("~is_merge").index
applied_op_ids = list(split_op_ids)

pruned_neuron = full_neuron.set_edits(split_op_ids, inplace=False)

neuron_list = []
applied_merges = []
for i in range(len(merge_op_ids) + 1):
    # apply the next operation
    current_neuron = full_neuron.set_edits(applied_op_ids, inplace=False)
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

    possible_operations = out_edges["operation_added"].unique()
    # print(len(possible_operations), "possible operations")

    ordered_ops = merge_op_ids[merge_op_ids.isin(possible_operations)]
    applied_op_ids.append(ordered_ops[0])
    applied_merges.append(ordered_ops[0])
    print(ordered_ops[0], "applied operation")
    if len(possible_operations) == 64:
        break

# %%
import numpy as np

full_neuron.edits.loc[np.unique(applied_op_ids), "metaoperation_id"]

metaop_counts = (
    full_neuron.edits.loc[np.unique(applied_op_ids)].groupby("metaoperation_id").size()
)

metaop_counts[metaop_counts > 1]

# issue - when doing this with the operations and not metaoperations, some things get
# messy when played out of order

# not sure whether to switch to metaoperations, or actually deal with that messiness

# one option is to just add nodes when i encounter them in the branching process


# %%
current_neuron.generate_neuroglancer_link(client)

# %%

# import the display function for ipython
from IPython.display import display

for neuron in neuron_list:
    display(neuron.generate_neuroglancer_link(client))

# %%
is_merge = full_neuron.edges["operation_added"].map(full_neuron.edits["is_merge"])

operation_type = pd.Series(index=full_neuron.edges.index, dtype="object")
operation_type[is_merge == True] = "merge"
operation_type[is_merge == False] = "split"
operation_type[is_merge.isna()] = "original"

full_neuron.edges["operation_added_type"] = operation_type
full_neuron.edges

# %%
component_labels = no_cross_neuron.component_labels()

full_neuron.nodes["component_label"] = component_labels

og_neuron = full_neuron.set_edits([], inplace=False)
og_neuron.select_nucleus_component().select_by_ball(50_000).generate_neuroglancer_link(
    client
)

# TODO figure out the correct logic here for generating the connected components


# %%
cross_neuron = full_neuron.query_edges(f"cross_{prefix}operation").remove_unused_nodes()
cross_neuron
# %%
