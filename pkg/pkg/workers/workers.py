import numpy as np
from taskqueue import queueable

import caveclient as cc
from pkg.edits import lazy_load_initial_network, lazy_load_network_edits
from pkg.neuronframe import load_neuronframe
from pkg.sequence import create_merge_and_clean_sequence, create_time_ordered_sequence


@queueable
def extract_edit_info(root_id):
    client = cc.CAVEclient("minnie65_phase3_v1")

    lazy_load_network_edits(root_id, client)

    lazy_load_initial_network(root_id, client, positions="lazy")

    return 1


@queueable
def extract_initial_network(root_id):
    client = cc.CAVEclient("minnie65_phase3_v1")

    lazy_load_initial_network(root_id, client)

    return 1


@queueable
def create_neuronframe(root_id):
    client = cc.CAVEclient("minnie65_phase3_v1")

    load_neuronframe(root_id, client)

    return 1


@queueable
def create_sequences(root_id):
    client = cc.CAVEclient("minnie65_phase3_v1")

    neuron = load_neuronframe(root_id, client)

    create_time_ordered_sequence(neuron, root_id)

    create_merge_and_clean_sequence(neuron, root_id, order_by="time")

    rng = np.random.default_rng(8888)
    for i in range(10):
        seed = rng.integers(0, np.iinfo(np.int32).max, dtype=np.int32)
        create_merge_and_clean_sequence(
            neuron, root_id, order_by="random", random_seed=seed
        )

    return 1
