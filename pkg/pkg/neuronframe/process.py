# %%

import caveclient as cc
import pandas as pd

from pkg.edits import (
    apply_edit_history,
    apply_synapses,
    collate_edit_info,
    get_initial_network,
    get_network_edits,
    get_network_metaedits,
    get_operation_metaoperation_map,
)
from pkg.io import lazycloud
from pkg.morphology import (
    apply_nucleus,
    apply_positions,
)
from pkg.neuronframe import NeuronFrame
from pkg.utils import get_level2_nodes_edges


@lazycloud(
    cloud_bucket="allen-minnie-phase3",
    folder="edit_neuronframes",
    file_suffix="neuronframe.pkl",
    arg_keys=[0],
)
def load_neuronframe(
    root_id: int,
    client: cc.CAVEclient,
    bounds_halfwidth: int = 20_000,
    cache_verbose: bool = False,
    use_cache: bool = True,
    only_load: bool = False,
) -> NeuronFrame:
    print("Loading level 2 network edits...")
    networkdeltas_by_operation = get_network_edits(
        root_id,
        client,
        bounds_halfwidth=bounds_halfwidth,
        use_cache=use_cache,
        cache_verbose=cache_verbose,
    )
    networkdeltas_by_metaoperation = get_network_metaedits(
        networkdeltas_by_operation,
        root_id,
        client,
        use_cache=use_cache,
        cache_verbose=cache_verbose,
    )

    print("Collating edit info...")
    operation_to_metaoperation = get_operation_metaoperation_map(
        networkdeltas_by_metaoperation
    )
    edit_stats, _, _ = collate_edit_info(
        networkdeltas_by_operation, operation_to_metaoperation, root_id, client
    )

    print("Loading initial network state...")
    # TODO this is super lazy and not optimized, just gets ALL of the initial states
    # for any neuron related to this one, but doesn't check whether parts of that
    # neuron can actually connect to this one.
    nf = get_initial_network(root_id, client, positions=False)

    # go through all of the edits/metaedits
    # add nodes that were added, but don't remove any nodes
    # mark nodes/edges with when they were added/removed
    # things that were never removed/added get -1

    print("Applying edit history to frame...")
    apply_edit_history(nf, networkdeltas_by_operation, operation_to_metaoperation)

    print("Applying positions...")
    apply_positions(nf, client, skip=True)

    print("Applying synapses...")
    pre_synapses, post_synapses = apply_synapses(
        nf,
        networkdeltas_by_operation,
        root_id,
        client,
    )

    print("Applying nucleus...")
    nuc_level2_id = apply_nucleus(nf, root_id, client)

    print("Creating full neuronframe...")
    full_neuron = NeuronFrame(
        nodes=nf.nodes,
        edges=nf.edges,
        nucleus_id=nuc_level2_id,
        neuron_id=root_id,
        pre_synapses=pre_synapses,
        post_synapses=post_synapses,
        edits=edit_stats,
    )

    print("Comparing to final neuron state...")
    edited_neuron = full_neuron.set_edits(
        full_neuron.edits.index, inplace=False
    ).select_nucleus_component(inplace=False)
    final_nodes, final_edges = get_level2_nodes_edges(root_id, client)
    final_neuron = NeuronFrame(nodes=final_nodes, edges=final_edges)
    final_neuron.n_connected_components()

    check = True
    if not edited_neuron.nodes.index.sort_values().equals(
        final_neuron.nodes.index.sort_values()
    ):
        print("Nodes do not match final state")
        print(edited_neuron.nodes.index)
        print(final_neuron.nodes.index)
        print()
        check = False

    edited_edges = pd.MultiIndex.from_frame(
        edited_neuron.edges[["source", "target"]]
    ).sort_values()
    final_edges = pd.MultiIndex.from_frame(
        final_neuron.edges[["source", "target"]]
    ).sort_values()

    if not edited_edges.equals(final_edges):
        print("Edges do not match final state")
        print(edited_edges.difference(final_edges))
        print(final_edges.difference(edited_edges))
        print()
        check = False

    if not check:
        raise ValueError("Edited neuron does not match final state.")

    full_neuron.apply_edge_lengths(inplace=True)

    return full_neuron
