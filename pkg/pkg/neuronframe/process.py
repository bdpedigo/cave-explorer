# %%

import caveclient as cc

from pkg.edits import (
    apply_edit_history,
    apply_synapses,
    collate_edit_info,
    get_initial_network,
    get_operation_metaoperation_map,
    lazy_load_network_edits,
)
from pkg.io import lazycloud
from pkg.morphology import (
    apply_nucleus,
    apply_positions,
)
from pkg.neuronframe import NeuronFrame


@lazycloud(
    cloud_bucket="allen-minnie-phase3",
    folder="edit_neuronframes",
    file_suffix="neuronframe.pkl",
    arg_keys=[0],
)
def load_neuronframe(
    root_id: int, client: cc.CAVEclient, cache_verbose: bool = False
) -> NeuronFrame:
    cache_verbose

    print("Loading level 2 network edits...")
    (
        networkdeltas_by_operation,
        networkdeltas_by_metaoperation,
    ) = lazy_load_network_edits(root_id, client=client)

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
    apply_positions(nf, client)

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

    full_neuron.apply_edge_lengths(inplace=True)

    return full_neuron
