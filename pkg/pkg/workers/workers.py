import caveclient as cc
from taskqueue import queueable

from pkg.edits import lazy_load_initial_network, lazy_load_network_edits


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
