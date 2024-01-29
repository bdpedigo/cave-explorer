# %%
import os

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

os.environ["SKEDITS_USE_CLOUD"] = "True"
os.environ["SKEDITS_RECOMPUTE"] = "False"


@lazycloud("allen-minnie-phase3", "edit_neuronframes", "neuronframe.pkl", arg_key=0)
def load_neuronframe(root_id: int, client: cc.CAVEclient):
    (
        networkdeltas_by_operation,
        networkdeltas_by_metaoperation,
    ) = lazy_load_network_edits(root_id, client=client)

    operation_to_metaoperation = get_operation_metaoperation_map(
        networkdeltas_by_metaoperation
    )

    edit_stats, metaoperation_stats, modified_level2_nodes = collate_edit_info(
        networkdeltas_by_operation, operation_to_metaoperation, root_id, client
    )

    initial_nf = get_initial_network(root_id, client, positions=False)

    # go through all of the edits/metaedits
    # add nodes that were added, but don't remove any nodes
    # mark nodes/edges with when they were added/removed
    # things that were never removed/added get -1

    nf = initial_nf.copy()

    apply_edit_history(nf, networkdeltas_by_operation, operation_to_metaoperation)

    apply_positions(nf, client)

    pre_synapses, post_synapses = apply_synapses(
        nf,
        networkdeltas_by_operation,
        root_id,
        client,
    )

    nuc_level2_id = apply_nucleus(nf, root_id, client)

    full_neuron = NeuronFrame(
        nodes=nf.nodes,
        edges=nf.edges,
        nucleus_id=nuc_level2_id,
        neuron_id=root_id,
        pre_synapses=pre_synapses,
        post_synapses=post_synapses,
        edits=edit_stats,
    )
    return full_neuron
