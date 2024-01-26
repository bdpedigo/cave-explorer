# %%
import os
import pickle

import caveclient as cc

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
    nf.edges["source_operation_added"] != full_neuron.edges["target_operation_added"]
)
full_neuron.edges["cross_metaoperation"] = (
    nf.edges["source_metaoperation_added"]
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
prefix = ""
no_cross_neuron = full_neuron.query_edges(
    f"(~cross_{prefix}operation) & (~was_removed)"
).query_nodes("~was_removed")
n_connected_components = no_cross_neuron.n_connected_components()

# TODO figure out the correct logic here for generating the connected components
# %%
cross_neuron = full_neuron.query_edges(f"cross_{prefix}operation").remove_unused_nodes()
cross_neuron