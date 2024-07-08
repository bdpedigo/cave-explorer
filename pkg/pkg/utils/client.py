from caveclient import CAVEclient

from pkg.constants import MATERIALIZATION_VERSION

def start_client():
    client = CAVEclient("minnie65_phase3_v1")
    client.materialize.version = MATERIALIZATION_VERSION
    return client

