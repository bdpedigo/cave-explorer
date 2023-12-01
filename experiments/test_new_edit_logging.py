import os

import caveclient as cc

from pkg.edits import (
    lazy_load_network_edits,
)

client = cc.CAVEclient("minnie65_phase3_v1")

root_id = 864691135082074103
os.environ["SKEDITS_RECOMPUTE"] = "True"
lazy_load_network_edits(root_id, client=client)
